
from torch.distributions import Categorical


import dgl

import torch.nn.functional as F
from dgl.nn.pytorch.glob import Set2Set

from torch.nn.utils.rnn import pad_sequence
import sys

sys.path.append('.')

from MARS.common.nn import GraphEncoder, MLP
from MARS.common.chem import mol_to_dgl

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from rdkit import Chem
from MARS.datasets.utils import load_vocab
from MARS.common.nn import masked_softmax


class Editor(ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.encoders = nn.ModuleDict()
        for name in ['act', 'del', 'add', 'arm']:
            encoder = GraphEncoder(
                config['n_atom_feat'], config['n_node_hidden'],
                config['n_bond_feat'], config['n_edge_hidden'], config['n_layers'])
            self.encoders[name] = encoder
        self.edge_mlp = MLP(config['n_bond_feat'], config['n_edge_hidden'])

    @abstractmethod
    def predict_act(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_del(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_add(self, g, h_node):
        raise NotImplementedError

    @abstractmethod
    def predict_arm(self, g, h_node):
        raise NotImplementedError

    def forward(self, g):
        '''
        @params:
            g: dgl.batch of molecular skeleton graphs
        @return:
            pred_act: action (del or add) prediction, torch.FloatTensor of shape (batch_size, 2)
            pred_del: bond breaking prediction,       torch.FloatTensor of shape (tot_n_edge, 1)
            pred_add: place to add arm,               torch.FloatTensor of shape (tot_n_node, 1)
            pred_arm: arm from vocab to add,          torch.FloatTensor of shape (tot_n_node, vocab_size)
        '''
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata['n_feat'].to(self.device)
            x_edge = g.edata['e_feat'].to(self.device)

        ### encode graph nodes
        encoded = {}
        for name, encoder in self.encoders.items():
            encoded[name] = encoder(g, x_node, x_edge)

        pred_act = self.predict_act(g, encoded['act'])
        pred_del = self.predict_del(g, encoded['del'])
        pred_add = self.predict_add(g, encoded['add'])
        pred_arm = self.predict_arm(g, encoded['arm'])
        return pred_act, pred_del, pred_add, pred_arm


class Actor(Editor):
    def __init__(self, config):
        super().__init__(config)
        self.set2set = Set2Set(config['n_node_hidden'], n_iters=6, n_layers=2)
        self.cls_act = MLP(config['n_node_hidden'] * 2, 2)
        self.cls_del = MLP(config['n_node_hidden'] * 2 + config['n_edge_hidden'], 1)
        self.cls_add = MLP(config['n_node_hidden'], 1)
        self.cls_arm = MLP(config['n_node_hidden'], config['vocab_size'])
        self.vocab_size = config['vocab_size']

    def predict_act(self, g, h_node):
        h = self.set2set(g, h_node)
        return self.cls_act(h)

    def predict_del(self, g, h_node):
        with torch.no_grad():
            x_edge = g.edata['e_feat'].to(self.device)

        h_u = h_node[g.edges()[0], :]  # (n_edges, n_node_hidden)
        h_v = h_node[g.edges()[1], :]  # (n_edges, n_node_hidden)
        h_edge = self.edge_mlp(x_edge)
        h_edge = torch.cat([h_u, h_edge, h_v], dim=1)  # (n_edges, n_node_hidden*2+n_edge_hidden)
        return self.cls_del(h_edge)

    def predict_add(self, g, h_node):
        return self.cls_add(h_node)

    def predict_arm(self, g, h_node):
        return self.cls_arm(h_node)


    def get_prob(self, graphs, mask = True):
        '''
        get probability of editing actions
        @params:
            graphs : molecular graphs, DGLGraphs batch
        @return:
            prob_act (list): (batch_size, 2)
            prob_del (list): (batch_size, n_edge)
            prob_add (list): (batch_size, n_node)
            prob_arm (list): (batch_size, n_node, vocab_size)
        '''
        with torch.no_grad():
            if mask :
                mask_edges = (graphs.edata['e_feat'].argmax(dim = 1) ==0).long().to(self.device)
                mask_nodes = (graphs.ndata['n_feat'][:,-1]>0).long().to(self.device)
            else :
                mask_edges = (graphs.edata['e_feat'].argmax(dim=1) >=0).long().to(self.device)
                mask_nodes = (graphs.ndata['n_feat'][:, -1] >=0).long().to(self.device)
        pred_act, pred_del, pred_add, pred_arm = self(graphs)

        # pred_act = F.softmax(pred_act, dim=1)  # (batch_size, 2)

        pred_del = pred_del  # (tot_n_edge, 1)
        pred_add = pred_add  # (tot_n_node, 1)
        pred_arm = F.softmax(pred_arm, dim =1)  # (tot_n_node, vocab_size)

        # prob_act = [p for p in pred_act]
        prob_act, prob_del, prob_add, prob_arm = [], [], [], []
        off_edge, off_node = 0, 0
        prob_global_act = []
        graphs = dgl.unbatch(graphs)
        for i,g in enumerate(graphs):
            n_edge = g.number_of_edges()
            n_node = g.number_of_nodes()
            p_del = pred_del[off_edge:off_edge + n_edge][:, 0].unsqueeze(0)  # (1, n_edges)
            mask_bond = mask_edges[off_edge:off_edge + n_edge].unsqueeze(0) # (1, n_edges)
            p_del = masked_softmax(p_del, mask_bond)  # (1, n_edges)
            p_add = pred_add[off_node:off_node + n_node][:, 0].unsqueeze(0)  # (1, n_node)
            mask_atoms = mask_nodes[off_node:off_node + n_node].unsqueeze(0) # (1, n_node)
            p_add = masked_softmax(p_add, mask_atoms)
            p_arm = pred_arm[off_node:off_node + n_node]  # (n_node, vocab_size)
            prob_del.append(p_del)
            prob_add.append(p_add)
            prob_arm.append(p_arm)
            off_edge += n_edge
            off_node += n_node
            mask_act = torch.LongTensor([mask_bond.sum()>0, mask_atoms.sum()>0]).to(self.device)
            assert mask_act.sum().item()>0
            p_act = masked_softmax(pred_act[i].unsqueeze(0), mask_act)

            ###TODO change debug
            # p_global_act = self.conditional_distribution2global_distribution(prob_act[i], p_del, p_add, p_arm)

            # prob_global_act.append(p_global_act)
            prob_act.append(p_act)

        return prob_act, prob_del, prob_add, prob_arm, prob_global_act



    def conditional_distribution2global_distribution(self, act_prob, idx_del_prob, idx_add_prob, idx_arm_prob):

        """
        :param act_prob:  proba del or add a fragment ([p_del,p_add])   (Batchsize = 1 , 2)
        :param idx_del_prob: distribution of del idx                    (n tot edges , 1)
        :param idx_add_prob: distribution of add idx                    (n tot nodes, 1)
        :param idx_arm_prob: distribution of arm idx according add idx  (n tot nodes. vocab size)
        :return: prob_action
        """
        ##### get shape informations
        with torch.no_grad():
            del_elt = idx_del_prob.shape[0]
            add_elt = idx_add_prob.shape[0]
            arm_elt = idx_arm_prob.shape[1]
            section = [del_elt, add_elt * arm_elt]
            section = torch.tensor(section).to(idx_arm_prob.device)


        act_prob = act_prob.squeeze()
        idx_del_prob = idx_del_prob.squeeze()
        idx_add_prob = idx_add_prob.squeeze()
        idx_arm_prob = idx_arm_prob.view(-1,)

        act_prob = torch.repeat_interleave(act_prob, section)
        idx_add_prob = torch.repeat_interleave(idx_add_prob, arm_elt)
        idx_add_arm_prob = idx_add_prob * idx_arm_prob
        prob_action = act_prob * torch.cat([idx_del_prob, idx_add_arm_prob])

        return prob_action


    def loss(self, batch, metrics=['loss']):
        g, edits = batch
        for key in edits.keys():
            edits[key] = edits[key].to(self.device)
        pred_act, pred_del, pred_add, pred_arm = self(g)
        dist_act = F.softmax(pred_act, dim=1) # (batch_size, 2)
        # dist_del = F.softmax(pred_del, dim=1)  # (tot_n_edge, 1)
        # dist_add = F.softmax(pred_add, dim=1)  # (tot_n_node, 1)
        # dist_arm = F.softmax(pred_arm, dim=1)  # (tot_n_node, vocab_size)
        loss_act = F.cross_entropy(pred_act, edits['act'])
        prob_act = dist_act.gather(dim=1,
            index=edits['act'].unsqueeze(dim=1)).mean()

        ### targets and masks
        n_node = g.number_of_nodes()
        n_edge = g.number_of_edges()
        off_node, off_edge = 0, 0
        prob_del, prob_add, prob_arm = [], [], []
        loss_add = 0
        loss_del = 0
        loss_arm = 0
        n_add = 0
        n_del = 0
        n_arm =0
        for i, g in enumerate(dgl.unbatch(g)):
            n_node = g.number_of_nodes()

            n_edge = g.number_of_edges()
            del_idx = edits['del'][i].item()
            add_idx = edits['add'][i].item()
            arm_idx = edits['arm'][i].item()
            if edits['act'][i].item() == 0:  # del
                pred_del_i = pred_del[off_edge : off_edge + n_edge,0].unsqueeze(0)
                loss_del += F.cross_entropy(pred_del_i, torch.LongTensor([del_idx]))
                n_del +=1

            else:  # add

                pred_add_i = pred_add[off_node : off_node + n_node,0].unsqueeze(0)
                loss_add += F.cross_entropy(pred_add_i, torch.LongTensor([add_idx]))

                pred_arm_i = pred_arm[off_node : off_node +n_node,:][add_idx].unsqueeze(0)
                loss_arm += F.cross_entropy(pred_arm_i, torch.LongTensor([arm_idx]))
                n_add+=1
                n_arm+=1


            off_node += n_node
            off_edge += n_edge

        loss = loss_act + loss_add/(n_add + 1e-6) + loss_arm/(n_arm + 1e-6) + loss_del/(n_del + 1e-6)
        local_vars = locals()
        metric_values = [local_vars[metric] for metric in metrics]
        return g.batch_size, metric_values


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']

        self.encoder = GraphEncoder(config['n_atom_feat'], config['n_node_hidden'],
                                    config['n_bond_feat'], config['n_edge_hidden'], config['n_layers'])
        self.set2set = Set2Set(config['n_node_hidden'], n_iters=6, n_layers=2)
        self.mlp_value = MLP(2* config['n_node_hidden'], 1)

    def forward(self, g):
        '''
        @params:
            g: dgl.batch of molecular skeleton graphs
        @return:
            value (batch_size, 1)
        '''
        with torch.no_grad():
            g = g.to(self.device)
            x_node = g.ndata['n_feat'].to(self.device)
            x_edge = g.edata['e_feat'].to(self.device)

        ### encode graph nodes
        h_nodes = self.encoder(g, x_node, x_edge)

        h_graph = self.set2set(g, h_nodes)
        return self.mlp_value(h_graph)



class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()

        self.actor = actor
        self.critic = critic
        self.vocab_size = actor.vocab_size



    def act(self, state):

        """

        :param state: dgl graph
        :return: action, action_logprob   ##### curt gradient line
        """
        act_prob, idx_del_prob, idx_add_prob, idx_arm_prob = self.actor.get_prob_one_graph(state)



        dist_act = Categorical(act_prob.squeeze())
        if idx_del_prob.sum().item() == 0 :   #### random take a element for del idx we don't use this index
            n = idx_del_prob.shape[0]
            idx_del_prob_detail = torch.FloatTensor([n*[1/n]])
        else :
            idx_del_prob_detail = idx_del_prob

        if idx_add_prob.sum().item() == 0 or idx_arm_prob.sum().item() ==0 :
            n = idx_add_prob.shape[0]
            idx_add_prob_detail = torch.FloatTensor([n*[1/n]])
            m = idx_arm_prob.shape[1]
            idx_arm_prob_detail = torch.FloatTensor(n*[m*[1/m]])

        else :
            idx_add_prob_detail = idx_add_prob
            idx_arm_prob_detail = idx_arm_prob



        dist_del = Categorical(idx_del_prob_detail.squeeze())
        dist_add = Categorical(idx_add_prob_detail.squeeze())


        act = dist_act.sample()
        act_logprob = dist_act.log_prob(act)

        del_idx = dist_del.sample()

        del_idx_logprob = dist_del.log_prob(del_idx)

        add_idx = dist_add.sample()


        add_idx_logprob = dist_add.log_prob(add_idx)
        dist_arm = Categorical(idx_arm_prob_detail[add_idx].squeeze())
        arm_idx = dist_arm.sample()
        arm_idx_logprob =dist_arm.log_prob(arm_idx)

        act = act.detach()
        del_idx = del_idx.detach()
        add_idx = add_idx.detach()
        arm_idx = arm_idx.detach()
        action_dict = {
            'act' : act,
            'del' : del_idx,
            'add' : add_idx,
            'arm' : arm_idx
        }

        act_logprob = act_logprob.detach()
        del_idx_logprob = del_idx_logprob.detach()
        add_idx_logprob = add_idx_logprob.detach()
        arm_idx_logprob = arm_idx_logprob.detach()

        logprob_dict ={
            'act' : act_logprob.item(),
            'del' : del_idx_logprob.item(),
            'add' : add_idx_logprob.item(),
            'arm' : arm_idx_logprob.item()
        }

        if (idx_add_prob.sum() + idx_del_prob.sum()).item() == 0:
            #### no possibility to add or del a fragment
            idx_del_prob = idx_del_prob_detail
            idx_add_prob = idx_add_prob_detail

        global_prob = self.conditional_distribution2global_distribution(act_prob, idx_del_prob, idx_add_prob, idx_arm_prob)


        assert abs(global_prob.sum().item() - 1)< 0.00001
        dist = Categorical(global_prob)
        global_action_idx = self.action_dict2global_action_idx(action_dict, state)
        logprob_global = dist.log_prob(global_action_idx)

        ##### untensorize
        action_dict = {key : action_dict[key].item() for key in action_dict}

        return global_action_idx.item(), logprob_global.item(), action_dict, logprob_dict

    def action_dict2global_action_idx(self, action, state):
        """

        :param action: action: dictionary of action
        :param state: dgl graph => determine range of del idx, add idx, arm idx
        :return: idx of action in global action [prob_del,prob_add]
        """
        act, del_idx, add_idx, arm_idx = action['act'], action['del'], action['add'], action['arm']
        with torch.no_grad():
            nb_edge = state.edges()[0].shape[0]         ######[0 : nb_edge -1] range of del_idx
        if act == 0:
            action_idx = del_idx
        elif act == 1:

            action_idx = nb_edge + add_idx * self.vocab_size + arm_idx

        else :
            NotImplementedError

        return action_idx

    def global_action_idx2action_dict(self, action_idx, nb_edges):

        """
        :param action_idx:
        :param nb_edges:
        :return: action dict act, add, arm, del
        """

        if action_idx < nb_edges :   ########### action => del a fragment for add and arm we can choose a random value like 0
            action = {
                'act' : 0,
                'del' : action_idx,
                'add' : 0,
                'arm' : 0
            }

        else :    ######### action => add a fragment we can choose a random value for del idx like 0

            add_idx = (action_idx - nb_edges) // self.vocab_size
            arm_idx = (action_idx - nb_edges) % self.vocab_size
            action = {
                'act' : 1,
                'del' : 0,
                'add' : add_idx,
                'arm' : arm_idx
            }

        return action


    def conditional_distribution2global_distribution(self, act_prob, idx_del_prob, idx_add_prob, idx_arm_prob):

        """
        :param act_prob:  proba del or add a fragment ([p_del,p_add])   (Batchsize =1, 2)
        :param idx_del_prob: distribution of del idx                    (n tot edges , 1)
        :param idx_add_prob: distribution of add idx                    (n tot nodes, 1)
        :param idx_arm_prob: distribution of arm idx according add idx  (n tot nodes. vocab size)
        :return: prob_action
        """
        ##### get shape informations
        with torch.no_grad():
            del_elt = idx_del_prob.shape[0]
            add_elt = idx_add_prob.shape[0]
            arm_elt = idx_arm_prob.shape[1]
            section = [del_elt, add_elt * arm_elt]
            section = torch.tensor(section).to(idx_arm_prob.device)

        act_prob = act_prob.squeeze()
        idx_del_prob = idx_del_prob.squeeze()
        idx_add_prob = idx_add_prob.squeeze()
        idx_arm_prob = idx_arm_prob.view(-1,)

        act_prob = torch.repeat_interleave(act_prob, section)
        idx_add_prob = torch.repeat_interleave(idx_add_prob, arm_elt)
        idx_add_arm_prob = idx_add_prob * idx_arm_prob
        prob_action = act_prob * torch.cat([idx_del_prob, idx_add_arm_prob])

        return prob_action




    def evaluate_one_graph(self, state, action):

        """
        :param state: dgl graph
        :param action: dictionary of actions
        :return: action_logprobs, state_values, dist_entropy
        """

        act, del_idx, add_idx, arm_idx = action['act'], action['del'], action['add'], action['arm']
        ####tensorize
        with torch.no_grad():
            device = state.device
            act = torch.tensor(act, dtype= torch.long).to(device)
            del_idx = torch.tensor(del_idx, dtype= torch.long).to(device)
            add_idx = torch.tensor(add_idx, dtype= torch.long).to(device)
            arm_idx = torch.tensor(arm_idx, dtype= torch.long).to(device)
            print('act', act)

        act_prob, idx_del_prob, idx_add_prob, idx_arm_prob = self.actor.get_prob_one_graph(state)

        dist_act = Categorical(act_prob.squeeze())
        dist_del = Categorical(idx_del_prob.squeeze())
        dist_add = Categorical(idx_add_prob.squeeze())
        dist_arm = Categorical(idx_arm_prob[add_idx].squeeze())

        act_logprob = dist_act.log_prob(act)
        del_idx_logprob = dist_del.log_prob(del_idx)

        add_idx_logprob = dist_add.log_prob(add_idx)

        arm_idx_logprob = dist_arm.log_prob(arm_idx)

        if act == 0 :
            action_logprobs = act_logprob + del_idx_logprob

        elif act == 1 :
            action_logprobs = act_logprob + add_idx_logprob + arm_idx_logprob

        else :
            NotImplementedError


        global_prob = self.conditional_distribution2global_distribution(act_prob, idx_del_prob, idx_add_prob, idx_arm_prob)
        global_dist = Categorical(global_prob)
        dist_entropy = global_dist.entropy()

        global_action_idx = self.action_dict2global_action_idx(action,state)
        with torch.no_grad() :
            global_action_idx = torch.tensor(global_action_idx)
        global_action_logprob = global_dist.log_prob(global_action_idx)

        state_values = self.critic(state)


        return action_logprobs, state_values, dist_entropy





    def evaluate (self, states, global_act_idx_batch):

        """
        :param states: batch dgl graph
        :param global_act_idx_batch: tensor action idx batch (batch, ) range(0, max(nb_edges) + max(nb_nodes) * vocab_size -1)
        :return: action_logprobs, state_values, dist_entropy
        """



        _, _, _,_, global_act_prob = self.actor.get_prob(states)

        global_act_prob = pad_sequence(global_act_prob, batch_first= True)
        global_dist = Categorical(global_act_prob)
        dist_entropy = global_dist.entropy()

        global_action_logprob = global_dist.log_prob(global_act_idx_batch)

        state_values = self.critic(states)


        return global_action_logprob, state_values, dist_entropy






if __name__ == '__main__':
    from MARS.common.nn import masked_softmax
    pred = torch.randn(size= (10,7))
    mask = torch.randint(0,2, size = (10,7))

    config = {
        'device': 'cpu',
        'n_atom_feat': 16,
        'n_node_hidden': 64,
        'n_bond_feat': 4,
        'n_edge_hidden': 128,
        'n_layers': 6,
        'vocab_size': 10,
        'batch_size': 128
    }
    actor = Actor(config)
    with open('mol_test.txt', 'r') as f:
        mols = f.readlines()

    mols = [Chem.MolFromSmiles(smi.split(',')[0]) for smi in mols if Chem.MolFromSmiles(smi.split(',')[0])]

    graphs = [mol_to_dgl(mol) for mol in mols]
    graphs = dgl.batch(graphs)
    prob_act_mask, prob_del_mask, prob_add_mask, prob_arm_mask, prob_global_act = actor.get_prob(graphs)
    prob_act, prob_del, prob_add, prob_arm, prob_global_act = actor.get_prob(graphs, mask=False)
    for i in range(len(prob_act)):
        print('act mask',prob_act_mask[i])
        print('act mask', prob_act[i])
        print('add mask',prob_add_mask[i])
        print('add',prob_add[i])
        print('del mask', prob_del_mask[i])
        print('del ', prob_del[i])

    from environment import Environement
    from MARS.common.utils import sample_idx
    env = Environement()
    env.current_graph = dgl.unbatch(graphs)[0]
    env.current_mol = mols[0]
    state = env.current_graph
    for i in range(100):
        prob_act_mask, prob_del_mask, prob_add_mask, prob_arm_mask, prob_global_act = actor.get_prob(dgl.batch([state]))
        print(prob_act_mask[0].tolist())
        act = sample_idx(prob_act_mask[0].tolist()[0])
        del_idx = sample_idx(prob_del_mask[0].tolist()[0])
        add_idx = sample_idx(prob_add_mask[0].tolist()[0])
        arm_idx = sample_idx(prob_arm_mask[0].tolist()[0])
        action = {
            'act' :act,
            'add' : add_idx,
            'arm' : arm_idx,
            'del' : del_idx
        }
        print(action)
        state, _ = env.next_step(action)



