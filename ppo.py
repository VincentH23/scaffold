
############################### Import libraries ###############################
import copy


import torch
import torch.nn as nn

import dgl


from actor import ActorCritic, Actor, Critic
from dataset import PPO_Dataset, RolloutBuffer
from torch.utils.data import DataLoader
from tqdm import tqdm

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")




################################## PPO Policy ##################################




class PPO:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip, value_coeff, entropy_coeff, lam, batch_size, config):


        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = config['device']

        self.policy = ActorCritic(actor= Actor(config), critic= Critic(config)).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(actor= Actor(config), critic= Critic(config)).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.rollout = RolloutBuffer()
        self.batch_size = batch_size



    def select_action(self, state):

        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            state = state.to(self.device)

            global_act_idx, global_logprob, action_dict= self.policy_old.act(state)
            value_state = self.policy_old.critic(state)

        return global_act_idx, global_logprob, action_dict, value_state


    def compute_advantages(self):
        self.rollout.compute_adv(self.gamma, self.lam)

    def compute_returns(self):
        self.rollout.compute_returns()


    def train(self):
        ### create loader + data
        dataset = PPO_Dataset(self.rollout)
        loader = DataLoader(dataset, batch_size= self.batch_size, collate_fn= PPO_Dataset.collate_fn)

        for _ in range(self.K_epochs):

            for batch in tqdm(loader):

            #### data generate by old policy
                old_actions, old_states, advantages, old_logprobs, discounted_return, action_dict  = batch

                old_actions = old_actions.to(self.device)
                old_states = old_states.to(self.device)
                advantages = advantages.to(self.device)
                old_logprobs = old_logprobs.to(self.device)
                discounted_return = discounted_return.to(self.device)


                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, mask= False)


                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1- self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                discounted_return = discounted_return.unsqueeze(1).detach()
                loss = -torch.min(surr1, surr2) + self.value_coeff * self.MseLoss(state_values, discounted_return) - self.entropy_coeff * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        #### clear buffer
        self.rollout.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


if __name__ == '__main__':
    import random
    from math import *
    import numpy as np
    import torch
    from environment import Environement
    from actor_critic import Actor, Critic, ActorCritic
    from dataset import  RolloutBuffer


    env = Environement()
    buffer = RolloutBuffer()
    config = {
        'device': 'cpu',
        'n_atom_feat': 17,
        'n_node_hidden': 64,
        'n_bond_feat': 5,
        'n_edge_hidden': 128,
        'n_layers': 6,
        'vocab_size': 1000,
    }
    critic = Critic(config)
    actor = Actor(config)
    actor_critic = ActorCritic(actor, critic)
    mols = []
    new_mols = []
    ppo_agent = PPO(lr_actor= 0.001, lr_critic= 0.001, gamma =0.999, K_epochs= 3, eps_clip= 0.1, value_coeff= 0.5, entropy_coeff= 0.1 , lam =0.9, batch_size= 5, config= config)
    step = 0
    for i in range(50):
        env.reset()
        done = False
        state = env.current_graph
        print('episode_{}'.format(i+1))
        while done == False:
            step +=1
            mols.append(env.current_mol)
            global_act_idx, global_logprob, action_dict, logprob_dict, value_state = ppo_agent.select_action(state)


            ppo_agent.rollout.states.append(state)
            ppo_agent.rollout.value_states.append(value_state)
            ppo_agent.rollout.actions_dict.append(action_dict)
            ppo_agent.rollout.logprobs_dict.append(logprob_dict)
            ppo_agent.rollout.actions_global_idx.append(global_act_idx)
            ppo_agent.rollout.logprobs.append(global_logprob)
            assert global_act_idx == (1 - action_dict['act']) * action_dict['del'] + action_dict['act'] * (
                    state.all_edges()[0].shape[0] + actor_critic.vocab_size * action_dict['add'] + action_dict['arm'])

            ##### next step env
            # action_dict['act'] = 1
            state, reward, done = env.next_step(action_dict)
            ppo_agent.rollout.rewards.append(reward)
            ppo_agent.rollout.is_terminals.append(done)
            ppo_agent.rollout.next_states.append(state)
            value_next_state = critic(state)
            ppo_agent.rollout.value_next_states.append(value_next_state)
            new_mols.append(env.current_mol)
            # print(len(env.new_mol_dict['mols']))
    ###end collecting data
    ppo_agent.compute_advantages()
    ppo_agent.compute_returns()
    print('total step ', step)

    # old_params = []
    # for p in ppo_agent.policy_old.parameters():
    #     old_params.append(p)
    #
    # params = []
    # for p in ppo_agent.policy.parameters():
    #     params.append(copy.deepcopy(p))
    # for i,old_p in enumerate(old_params):
    #     print((old_p!=params[i]).sum())
    #     # print(old_p)
    #     # print(params[i])
    # ppo_agent.train()
    #
    # old_params = []
    # for p in ppo_agent.policy_old.parameters():
    #     old_params.append(p)
    #
    # params = []
    # for p in ppo_agent.policy.parameters():
    #     params.append(p)
    # for i, old_p in enumerate(old_params):
    #     print((old_p != params[i]).sum())
    # #
    # # dict =ppo_agent.rollout.__dict__
    # # for key in dict :
    # #     print(key)
    # #     print(dict[key])
    #
    # # dict = ppo_agent.rollout.__dict__
    # # # for key in dict:
    # # #     print(key)
    # # #     print(dict[key])
    # #
    # # dataset = PPO_Dataset(ppo_agent.rollout)
    # # loader = torch.utils.data.DataLoader(dataset, 5, collate_fn=PPO_Dataset.collate_fn)
    # # for g in ppo_agent.rollout.states[:5]:
    # #     print(g)
    # #
    # # print('endndndn')
    # # print(ppo_agent.rollout.actions_global_idx)
    # # batch = next(iter(loader))
    # # global_act_idxs, states, vs_target, logprobs, discounted_returns, actions_dict = batch
    # #
    # # from ppo import PPO
    # #
    # # print(logprobs)
    # # print(ppo_agent.policy_old.evaluate(states, global_act_idxs))
    # # print(ppo_agent.policy_old.critic(states))
    # #
    # print(mols)
    # print(new_mols)
    # from utils import draw_mol
    #
    # for i,mol in enumerate(mols):
    #     draw_mol(mol, 'mols/'+'mols_{}.png'.format(i))
    #     draw_mol(new_mols[i], 'mols/' + 'new_mols_{}.png'.format(i))
    #
    # a = ppo_agent.rollout.states
    # for i,g in enumerate(a):
    #     print(i)
    #     print(g.ndata['n_feat'][0])
    #
    # print(ppo_agent.rollout.actions_dict)
    # for i,s in enumerate(ppo_agent.rollout.states):
    #     print(ppo_agent.rollout.actions_dict[i])
    #     print(s.all_edges()[0][ppo_agent.rollout.actions_dict[i]['del']],s.all_edges()[1][ppo_agent.rollout.actions_dict[i]['del']])





