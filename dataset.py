import dgl
import torch
from torch.utils import data


class RolloutBuffer:
    def __init__(self):

        #TODO supress dict
        self.actions_dict = []
        self.actions_global_idx = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_states = []
        self.value_states = []
        self.value_next_states = []
        self.advantages = []
        self.discounted_returns = []


    def compute_adv(self, gamma, lam):
        advs = []

        rev_vs = reversed(self.value_states)
        rev_v_next_state = reversed(self.value_next_states)
        re_reward = reversed(self.rewards)
        re_is_terminal = reversed(self.is_terminals)
        adv = 0
        for reward, vs_next, vs, is_terminal in zip(*(re_reward, rev_v_next_state, rev_vs, re_is_terminal)):
            if is_terminal:
                adv = 0
            delta = reward + gamma * vs_next - vs
            adv = delta + lam * gamma * adv
            advs.append(adv.item())

        self.advantages = advs[::-1]

    def clear(self):
        del self.actions_dict[:]
        del self.actions_global_idx[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_states[:]
        del self.value_states[:]
        del self.value_next_states[:]
        del self.advantages[:]
        del self.discounted_returns[:]

    def compute_returns(self):
        """

        advantages:  advantages estimate by GAE methods
        value_state: previous value state          At = rt + gamma*V(t+1) - V(t)
        :return: None
        """
        self.discounted_returns = (torch.tensor(self.advantages) + torch.tensor(self.value_states)).tolist()




class PPO_Dataset(data.Dataset):

    def __init__(self, buffer):
        super(PPO_Dataset, self).__init__()
        self.buffer = buffer


    def __len__(self):
        return len(self.buffer.states)

    def __getitem__(self, index):
        act_dict = self.buffer.actions_dict[index]
        act_global_idx = self.buffer.actions_global_idx[index]
        state = self.buffer.states[index]
        advantage = self.buffer.advantages[index]
        logprob = self.buffer.logprobs[index]
        discounted_return = self.buffer.discounted_returns[index]
        return act_dict, act_global_idx, state, advantage, logprob, discounted_return

    @staticmethod
    def collate_fn(batch):

        act_dicts, act_global_idxs, states, advantages, logprobs, discounted_returns = list(zip(*batch))

        actions_dict = {}
        for key in act_dicts[0].keys():
            actions_dict[key] = [t[key] for t in act_dicts]
            actions_dict[key] = torch.tensor(actions_dict[key]).long()

        states = dgl.batch(states)
        action_global_idx = torch.tensor(act_global_idxs).long()
        adv = torch.tensor(advantages).float()
        logprobs = torch.tensor(logprobs).float()
        discounted_returns = torch.tensor(discounted_returns).float()

        return action_global_idx.detach(), states, adv.detach(), logprobs.detach(), discounted_returns.detach(), actions_dict


if __name__ == '__main__':
    import random
    from math import *
    import numpy as np
    import torch
    from environment import Environement
    from actor_critic import Actor, Critic, ActorCritic

    #
    #
    # def random_select_action(state):
    #     nb_edges = state.edges()[0].shape[0]
    #     nb_nodes = state.number_of_nodes()
    #     act = random.randint(0, 1)
    #     if act :
    #         arm_idx = random.randint(0, 100 - 1)
    #         add_idx = random.randint(0, nb_nodes - 1)
    #         del_idx = 0
    #     else :
    #         arm_idx =0
    #         add_idx = 0
    #         del_idx = random.randint(0, nb_edges - 1)
    #     action = {
    #         'act': act,
    #         'del': del_idx,
    #         'add': add_idx,
    #         'arm': arm_idx
    #     }
    #     prob = 1/(nb_edges + 100* nb_nodes)
    #     logprob = log10(prob)
    #     return action, logprob


    ### run one episode

    env = Environement()
    buffer = RolloutBuffer()
    from actor_critic import Critic, Actor,ActorCritic

    config = {
        'device': 'cpu',
        'n_atom_feat': 17,
        'n_node_hidden': 64,
        'n_bond_feat': 5,
        'n_edge_hidden': 128,
        'n_layers': 6,
        'vocab_size': 1000,
        'batch_size': 128
    }
    critic = Critic(config)
    actor = Actor(config)
    actor_critic = ActorCritic(actor,critic)
    for i in range(10):
        env.reset()
        done = False
        state = env.current_graph

        while done == False:

            ### select action
            global_act_idx, global_logprob, action, logprob = actor_critic.act(state)
            print('compute', global_act_idx)
            value_state = critic(state)
            buffer.states.append(state)
            buffer.value_states.append(value_state)
            buffer.actions_dict.append(action)
            buffer.logprobs_dict.append(logprob)
            buffer.actions_global_idx.append(global_act_idx)
            buffer.logprobs.append(global_logprob)
            assert global_act_idx == (1-action['act'])*action['del'] + action['act']*(state.all_edges()[0].shape[0] + actor_critic.vocab_size * action['add'] + action['arm'])

            ##### next step env
            state, reward, done = env.next_step(action)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            buffer.next_states.append(state)
            value_state = critic(state)
            buffer.value_next_states.append(value_state)

    buffer.compute_adv(lam= 0.9, gamma =0.999)
    buffer.compute_returns()

    dict = buffer.__dict__
    for key in dict :
        print(key)
        print(dict[key])


    dataset = PPO_Dataset(buffer)
    loader = torch.utils.data.DataLoader(dataset, 5, collate_fn= PPO_Dataset.collate_fn)
    for g in buffer.states[:5]:
        print(g)

    print('endndndn')
    print(buffer.actions_global_idx)
    batch = next(iter(loader))
    global_act_idxs, states, vs_target, logprobs, discounted_returns, actions_dict =  batch

    from ppo import PPO
    print(logprobs)
    print(actor_critic.evaluate(states,global_act_idxs))
    print(actor_critic.critic(states))

