import copy

import torch
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from MARS.common.chem import break_bond_mol, add_arm, check_validity, mol_to_dgl, draw_mol, find_common_scaffold
from MARS.datasets.utils import load_vocab
from MARS.common.utils import sample_idx
import numpy as np
import math
import random

from MARS.estimator.scorer.scorer import get_scores, get_score
from rdkit import Chem

from MARS.datasets.datasets import ImitationDataset
from torch.utils import data

class Environement:

    def __init__(self):
        self.vocab = load_vocab('MARS/data', 'chembl', 1000)
        with open('MARS/data/actives_gsk3b,jnk3.txt','r') as f:
            smiles = f.readlines()
        smiles = [line.split()[0] for line in smiles]
        smiles = [line.split(',')[0] for line in smiles]
        ref_smiles = smiles[1:]
        ref_mols = [Chem.MolFromSmiles(smi) for smi in ref_smiles if Chem.MolFromSmiles(smi)]

        ref_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in ref_mols]
        self.ref_mol_dict = {
            'fps' : ref_fps,
            'mols' : ref_mols,
            'smiles' : ref_smiles,
           }

        new_smiles = []
        new_fps = []
        new_mols = []
        self.new_mol_dict = {
            'fps': new_fps,
            'mols': new_mols,
            'smiles': new_smiles

        }

        self.nb_episode = 0
        self.threshold_nov = 0.4
        ##TODO change value of threshold
        self.threshold_div = 0.3
        self.thresholf_mydiv = 0.5
        self.max_step = 10
        self.step = 0
        self.new = True
        self.current_mol = None
        self.current_graph = None
        self.indice_init_mol = None           ####### indice in the set (subtract len(ref_mols) if init mol from the new_mol_set)
        self.init_mol_from_ref = True

        self.size_ref_dataset = len(self.ref_mol_dict['mols'])
        self.nb_call_reward = 0


        with open('scaffold.txt','r') as f:
            smiles = f.readlines()
        smiles = [line.split()[0] for line in smiles]
        scaffold = [Chem.MolFromSmiles(smi) for smi in smiles if Chem.MolFromSmiles(smi)]
        self.scaffold_mol_dict = {
            'mols' : scaffold,
            'smiles' : smiles,
            'UCB' : list(np.zeros(len(scaffold))),
            'R' : list(np.zeros(len(scaffold))),
            'N' : list(np.zeros(len(scaffold)))
        }

        self.previous_score = None

        self.imitation_dataset = ImitationDataset(graphs=[], edits= {
            'act' :[],
            'del' : [],
            'add' : [],
            'arm' : [],
            'global_act' : []
        })

        self.MaxImittationSetSize = 50000
        self.index_scaffold = None


    def reset(self):
        self.new = True
        self.step = 0
        self.current_mol = None
        self.current_graph = None
        self.index_scaffold = None
        if self.new :
            self.current_mol, self.current_graph = self.init_mol()
            self.new = False

        self.nb_episode +=1



    def init_mol(self):
        """
        return init molecule
        """
        R = np.array(self.scaffold_mol_dict['R'])
        N = np.array(self.scaffold_mol_dict['N'])

        ##TODO change the way to init the molecule
        prob =  1 + (1+R)/(N+1e-6)
        idx_init_mol = sample_idx(prob)
        init_mol = self.scaffold_mol_dict['mols'][idx_init_mol]
        init_graph = mol_to_dgl(init_mol)
        self.index_scaffold = idx_init_mol
        self.previous_score,_,_ = self.compute_reward(init_mol, True)
        return init_mol, init_graph


    def edit(self, graph, mol, action, del_idx, add_idx, arm_idx):
        """

        @param graph:  dgl graph
        @param mol:  molecule
        @param action: action 0 or 1
        @param del_idx:  idx in graph
        @param arms_idx: arm idx
        @param add_idx:  add idx


        @return new_mol, not_changed
        """

        if action == 0:  ####del

            u = graph.all_edges()[0][del_idx].item()
            v = graph.all_edges()[1][del_idx].item()
            try:
                new_mol = break_bond_mol(mol, u, v)
                if new_mol.GetNumBonds() <= 0:
                    raise ValueError

            except ValueError:
                new_mol = None

            if check_validity(new_mol):
                return new_mol, False

        elif action ==1 : ###add arm
            new_arm = self.vocab.arms[arm_idx]
            try :
                new_mol = add_arm(mol, u=add_idx, arm = new_arm.mol, v = new_arm.v)
            except :
                new_mol = None

            if check_validity(new_mol) and new_mol.GetNumAtoms() <= 50:  # limit size
                return new_mol, False
        else :
            NotImplementedError


        return mol, True

    def next_step(self, action):
        """

        :param action: dict 'act', 'del', 'add', 'arm'
        :return: state (new_mol), reward, done(if no add to new_mols)
        """

        self.step += 1
        act = action['act']
        del_idx = action['del']
        add_idx = action['add']
        arm_idx = action['arm']
        global_action = action['global_act']

        ###make a copy of graph and molecule
        mol = copy.deepcopy(self.current_mol)
        graph = copy.deepcopy(self.current_graph)

        ###edition of molecule
        new_mol, not_changed = self.edit(graph, mol, action=act, del_idx=del_idx, add_idx=add_idx, arm_idx=arm_idx)
        if not_changed == False:
            self.current_mol = new_mol
            self.current_graph = mol_to_dgl(new_mol)


        ###compute reward
        reward, CP, CND = self.compute_reward(new_mol, not_changed)



        ##### save molecule according its chemical property
        if CP and CND:
            self.new_mol_dict['mols'].append(new_mol)
            self.new_mol_dict['fps'].append(AllChem.GetMorganFingerprintAsBitVect(new_mol, 3, 2048))
            self.update_scaffold(new_mol)
            self.scaffold_mol_dict['R'][self.index_scaffold] += 1




        # ###improvement imitation learning
        if self.previous_score < reward:
            self.update_imitation_dataset(action, graph)

        self.previous_score = reward
        if self.step >= self.max_step:
            return self.current_graph, reward, True
        else:
            return self.current_graph, reward, False


    def update_imitation_dataset(self, action, graph):
        edits = copy.deepcopy(action)
        for k in edits:
            edits[k] = [edits[k]]
        dataset = ImitationDataset(graphs=[graph], edits=edits)
        self.imitation_dataset.merge_(dataset)
        n_sample = len(dataset)
        if n_sample > self.MaxImittationSetSize:
            indices = [i for i in range(n_sample)]
            random.shuffle(indices)
            indices = indices[:50000]
            self.imitation_dataset = data.Subset(self.imitation_dataset, indices)
            self.imitation_dataset = ImitationDataset.reconstruct(self.imitation_dataset)

    def update_scaffold(self, mol):
        for m in self.ref_mol_dict['mols'] + self.new_mol_dict['mols']:
            scaffold = find_common_scaffold(mol, m)
            if scaffold is not None:
                self.scaffold_mol_dict['R'].append(0)
                self.scaffold_mol_dict['N'].append(0)
                self.scaffold_mol_dict['mols'].append(scaffold)

    def compute_reward(self, new_mol, not_change):
        """

        :param new_mol: new_state mol
        :param not_change: if mol change or not (if don't change => invalid action)
        :return: reward
        """
        gsk3b_score, jnk3_score, qed_score, sa_score = self.compute_prop_score(new_mol)
        sim_with_ref = self.compute_sim_with_data(new_mol, self.ref_mol_dict['fps'])
        if len(self.new_mol_dict['fps']):
            sim_with_proposal = self.compute_sim_with_data(new_mol, self.new_mol_dict['fps'])
        else :
            sim_with_proposal = [0]
        C_div = int(np.array(sim_with_proposal).mean()<self.threshold_div)
        C_novelty = int(max(sim_with_ref)<self.threshold_nov)
        CND = C_div*C_novelty
        CP = int(gsk3b_score>=0.5) * int(jnk3_score>=0.5) * int(qed_score>=0.6) * int(sa_score>=0.67)

        prop_reward = gsk3b_score* jnk3_score * qed_score* sa_score /4
        nov_reward = 1 - max(sim_with_ref) if max(sim_with_ref) >= self.threshold_nov + 0.1 else 1 - (self.threshold_nov) + 0.1
        # div_reward = 1 - max(sim_with_proposal) if max(sim_with_proposal) >= self.threshold_nov + 0.1 else 1 - (self.threshold_nov) + 0.1
        div_reward = 1 - max(sim_with_proposal)
        reward = prop_reward * nov_reward * div_reward
        return reward, CP, CND

    def compute_prop_score(self, mol):
        gsk3b_score = get_scores('gsk3b', [mol])[0]
        jnk3_score = get_scores('jnk3', [mol])[0]
        sa_score = get_score('sa', mol)
        qed_score = get_score('qed', mol)

        return gsk3b_score, jnk3_score, qed_score, sa_score

    def compute_sim_with_data(self, mol, data):
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
        sim = DataStructs.BulkTanimotoSimilarity(fps, data)
        return sim






if __name__ == '__main__':

    mol = Chem.MolFromSmiles('NC(=O)c1ccccc1Nc1ccnc(Oc2ccccc2)c1')
    graph = mol_to_dgl(mol)


    env = Environement()

    env.current_graph = graph
    env.current_mol = mol
    for i in range(50):
        print(i)
        add_avalaible_atom = torch.nonzero(env.current_graph.ndata['n_feat'][:, -1], as_tuple=True)[0]
        del_avalaible_bond = torch.nonzero(env.current_graph.edata['e_feat'].argmax(dim=-1) == 0, as_tuple=True)[0]

        action_prob = [int(del_avalaible_bond.shape[0] > 0), int(add_avalaible_atom.shape[0] > 0)]
        if sum(action_prob)==0 :
            break
        action = sample_idx(action_prob)
        if action ==0 :

            del_idx = del_avalaible_bond.tolist()[sample_idx(len(del_avalaible_bond.tolist())*[1])]
            arm_idx = None
            add_idx = None
            print('check bond type')
            # for index in del_avalaible_bond :
            #     u = env.current_graph.all_edges()[0][index].item()
            #     v = env.current_graph.all_edges()[1][index].item()
            #     print(env.current_mol.GetBondBetweenAtoms(u,v).GetBondType())
        elif action==1 :
            arm_idx = sample_idx([i for i in range(1000)])
            add_idx = add_avalaible_atom.tolist()[sample_idx(len(add_avalaible_atom.tolist())*[1])]
            del_idx = None

        action_dict = {
            'act' : action,
            'del' : del_idx,
            'arm' : arm_idx,
            'add' : add_idx
        }
        _, not_changed = env.next_step(action_dict)

        print('not changed ', not_changed)
        if not_changed :
            print(action_dict)
            print('nb atoms', len(env.current_mol.GetAtoms()))
            # if action == 0 :
            #     u = env.current_graph.all_edges()[0][del_idx].item()
            #     v = env.current_graph.all_edges()[1][del_idx].item()
            #     type_bond = str(env.current_mol.GetBondBetweenAtoms(u,v).GetBondType())
            #     path = 'mols/'+type_bond +'_mol_del_{}_{}.jpg'.format(u,v)
            #     draw_mol(env.current_mol,path)
            #     print(path)
            #     print(action_prob)
            #     print(del_idx)
            #     print(del_avalaible_bond)
            #     for index in del_avalaible_bond:
            #         u = env.current_graph.all_edges()[0][index].item()
            #         v = env.current_graph.all_edges()[1][index].item()
            #         print(env.current_mol.GetBondBetweenAtoms(u, v).GetBondType())

            if action == 1:
                path = 'mols/mol_add_idx_{}_arm_idx_{}.jpg'.format(add_idx, arm_idx)
                # draw_mol(env.current_mol,path)
                print(path)
                print(env.current_graph.in_degrees())
                print(torch.nonzero(env.current_graph.out_degrees()==1, as_tuple =True))
                leaf = torch.nonzero(env.current_graph.out_degrees()==1, as_tuple =True)[0].unsqueeze(0).repeat(10,1)
                print(leaf)
                gr = env.current_graph.all_edges()[0][:10].unsqueeze(1).repeat(1,torch.nonzero(env.current_graph.out_degrees()==1, as_tuple =True)[0].shape[0])
                print(gr)
                print((gr==leaf).sum(dim=-1))
    # env.edit(graph,mol,action= , del_idx=, add_idx=, arm_idx=)


    # if __name__ == '__main__':
    #     env = Environment_improve()
    #     action = {
    #         'act' : 0,
    #         'del_idx' : 5,
    #         'add_idx' : 3,
    #         'arm_idx' : 6
    #     }
    #     mol, _ = env.init_mol()
    #
    #     print(env.compute_prop_score(mol))
    #     print()
    #     print(env.scaffold_mol_dict['smiles'][env.index_scaffold])
    #     # print(env.indice_init_mol, env.init_mol_from_ref)
#
#     # with open('score.txt','w') as f:
#     #     for i,smi in enumerate(env.ref_mol_dict['smiles']):
#     #         text = smi
#     #         mol = env.ref_mol_dict['mols'][i]
#     #         gsk3 = get_scores('gsk3b',[mol])
#     #         jnk3 = get_scores('jnk3', [mol])
#     #         text += ','+str(gsk3)+','+str(jnk3)
#     #         f.write(text+'\n')
#     #
#     # with open('MARS/data/actives_jnk3.txt', 'r') as f :
#     #     data = f.readlines()
#     # data = [smi[:-1] for smi in data[1:]]
#     # score = []
#     # smiles = []
#     # mols = []
#     # for d in data :
#     #     smi, score_ = d.split(',')
#     #     smiles.append(smi)
#     #     score.append(score_)
#     #     mols.append(Chem.MolFromSmiles(smi))
#     # scores = np.array(get_scores('jnk3',mols))>=0.5
#     # print(scores.mean())
#
#     done = False
#     for t in range(100):
#
#         done = False
#         env.reset()
#         c = 0
#         while done ==False  and c<10:
#             c+=1
#             graph = env.current_graph
#             nb_edges = graph.all_edges()[0].shape[0]
#             nb_nodes = graph.number_of_nodes()
#
#             act = sample_idx([0.5,0.5])
#             del_idx = sample_idx(nb_edges*[1/nb_edges])
#             p_add = np.array(nb_nodes*[1/nb_nodes])*(graph.ndata['n_feat'][:,-1]>0).numpy()
#             add_idx = sample_idx(p_add.tolist())
#             arm_idx = sample_idx(10*[1/10])
#             action = {
#                 'act' : 1,
#                 'del' : del_idx,
#                 'add' : add_idx,
#                 'arm' : arm_idx
#             }
#             state, reward, done = env.next_step(action)
#             print('imitation size ', len(env.imitation_dataset))
#
#     # ucb_ref = np.array(env.ref_mol_dict['UCB'])
#     # print(ucb_ref[ucb_ref!=0])
#     # print(np.array(env.ref_mol_dict['N'])[np.array(env.ref_mol_dict['N'])!=0],'N')
#     # print(np.array(env.ref_mol_dict['R'])[np.array(env.ref_mol_dict['N'])!=0],'R')
#     # print(np.array(env.ref_mol_dict['N']).sum()+np.array(env.new_mol_dict['N']).sum())
#     # print(np.array(env.ref_mol_dict['R']).sum() + np.array(env.new_mol_dict['R']).sum())
#     # print(len(env.new_mol_dict['mols']))
#     # print(env.new_mol_dict['smiles'])
#     # print(env.new_mol_dict['UCB'])
#     # print(env.new_mol_dict)
#
#     new_mol = Chem.MolFromSmiles('O=[SH](=O)c1ccccc1Nc1nc(Nc2ccccc2)ncc1Cl')
#     gsk3b_score = get_scores('gsk3b', [new_mol])[0]
#     jnk3_score = get_scores('jnk3', [new_mol])[0]
#     sa_score = get_score('sa', new_mol)
#     qed_score = get_score('qed', new_mol)
#     print(gsk3b_score)
#     print(jnk3_score)
#     print(sa_score)
#     print(qed_score)
#     print(min(gsk3b_score,0.6))





