import torch

from environment import *
from actor import Actor, Critic, ActorCritic

from dataset import  RolloutBuffer
from ppo import PPO
from MARS.common.train import imitation_training


config = {
    'device': 'cpu',
    'n_atom_feat': 16,
    'n_node_hidden': 64,
    'n_bond_feat': 4,
    'n_edge_hidden': 128,
    'n_layers': 6,
    'vocab_size': 1000,
}

lr_actor = 0.0002
lr_critic = 0.0002
gamma = 0.999
K_epochs = 3
eps_clip = 0.2
value_coeff = 0.5
entropy_coeff = 0.01
lam = 0.95
batch_size = 64
objective_nb_mols = 25
train_episode_freq = 3


env = Environement()
buffer = RolloutBuffer()
critic = Critic(config)
actor = Actor(config)
actor_critic = ActorCritic(actor, critic)
ppo_agent = PPO(config=config, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, lam=lam, K_epochs=K_epochs, eps_clip=eps_clip, value_coeff=value_coeff, entropy_coeff=entropy_coeff, batch_size=batch_size)
nb_dicovered_mols_step = []
nb_dicovered_mols_step_episode = []
i =0
records_rewards = []
imitation_optimizer = torch.optim.Adam(ppo_agent.policy_old.actor.parameters(), lr=lr_actor)
for i in range(1000):

    env.reset()
    done = False
    state = env.current_graph
    # print('episode_{}'.format(env.nb_episode))
    print('game_{}'.format(i))
    r = 0
    R = 0
    with torch.no_grad():
        while done == False:

            global_act_idx, global_logprob, action_dict, value_state = ppo_agent.select_action(state)

            ppo_agent.rollout.states.append(state)
            ppo_agent.rollout.value_states.append(value_state)
            ppo_agent.rollout.actions_dict.append(action_dict)
            ppo_agent.rollout.actions_global_idx.append(global_act_idx)
            ppo_agent.rollout.logprobs.append(global_logprob)



            ##### next step env
            state, reward, done = env.next_step(action_dict)
            ppo_agent.rollout.rewards.append(reward)
            ppo_agent.rollout.is_terminals.append(done)
            ppo_agent.rollout.next_states.append(state)
            value_next_state = ppo_agent.policy_old.critic(state)
            ppo_agent.rollout.value_next_states.append(value_next_state)
            nb_dicovered_mols_step.append(len(env.new_mol_dict['mols']))
            r=r+reward
        R += r

    if (i+1)%10 ==0:
        print('training')
        print('R:{}'.format(R/10))
        records_rewards.append(R/10)
        print('n data:{}'.format(len(env.imitation_dataset)))
        # print('global', ppo_agent.rollout.actions_global_idx)
        # print('state',ppo_agent.rollout.states)
        # print('logp',ppo_agent.rollout.logprobs)
        # print('rewards',ppo_agent.rollout.rewards)
        # print('done', ppo_agent.rollout.is_terminals)
        # print('next states',ppo_agent.rollout.next_states)
        # print('v state',ppo_agent.rollout.value_states)
        # print('next VS',ppo_agent.rollout.value_next_states)
        # # self.advantages = []
        # # self.discounted_returns = []
        imitation_training(ppo_agent.policy_old.actor, epochs=5, dataset=env.imitation_dataset,optimizer=imitation_optimizer)
        ppo_agent.compute_advantages()
        ppo_agent.compute_returns()
        ppo_agent.train()




