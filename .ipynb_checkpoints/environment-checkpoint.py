import copy

from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from MARS.common.chem import break_bond, Skeleton, combine, check_validity, mol_to_dgl
from MARS.datasets.utils import load_vocab
from MARS.common.utils import sample_idx
import numpy as np
import math
import random

from MARS.estimator.scorer.scorer import get_scores, get_score
from rdkit import Chem
from utils import draw_mol, find_common_scaffold
from MARS.datasets import ImitationDataset
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
            'UCB' : list(np.zeros(len(ref_mols))),
            'R' : list(np.zeros(len(ref_mols))),
            'N' : list(np.zeros(len(ref_mols)))
        }

        new_smiles = []
        new_fps = []
        new_mols = []
        new_UCB = []
        self.new_mol_dict = {
            'fps': new_fps,
            'mols': new_mols,
            'smiles': new_smiles
        }
        self.nb_episode = 0
        self.threshold_nov = 0.4
        self.threshold_div = 0.6
        self.max_step = 10
        self.step = 0
        self.new = True
        self.current_mol = None
        self.current_graph = None
        self.indice_init_mol = None           ####### indice in the set (subtract len(ref_mols) if init mol from the new_mol_set)
        self.init_mol_from_ref = True

        self.size_ref_dataset = len(self.ref_mol_dict['UCB'])
        self.nb_call_reward = 0







    def next_step(self, action):
        """

        :param action: dict 'act', 'del', 'add', 'arm'
        :return: state (new_mol), reward, done(if no add to new_mols)
        """
        return NotImplementedError

    def compute_reward(self, new_mol, not_change) :
        """
        
        :param new_mol: new_state mol
        :param not_change: if mol change or not (if don't change => invalid action)
        :return: reward 
        """
        return NotImplementedError

    def compute_prop_score(self, mol):
        gsk3b_score = get_scores('gsk3b', [mol])[0]
        jnk3_score = get_scores('jnk3', [mol])[0]
        sa_score = get_score('sa', mol)
        qed_score = get_score('qed', mol)

        return gsk3b_score, jnk3_score,  qed_score, sa_score

    def compute_sim_with_data(self, mol, data):
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
        sim = max(DataStructs.BulkTanimotoSimilarity(fps, data))
        return  sim



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
                new_mol = break_bond(mol, u, v)
                if new_mol.GetNumBonds() <= 0:
                    raise ValueError

            except ValueError:
                new_mol = None

            if check_validity(new_mol):
                return new_mol, False

        elif action ==1 : ###add arm
            new_arm = self.vocab.arms[arm_idx]
            skeleton = Skeleton(mol, u=add_idx, bond_type=new_arm.bond_type)
            # print(add_idx)
            # print(Chem.MolToSmiles(mol))
            # draw_mol(mol,'mol.jpg')
            # print(new_arm.v)
            # print(Chem.MolToSmiles(new_arm.mol))
            # draw_mol(new_arm.mol,'arm.jpg')
            new_mol = combine(skeleton, new_arm)


            if check_validity(new_mol) and new_mol.GetNumAtoms() <= 50:  # limit size
                return new_mol, False
        else :
            NotImplementedError


        return mol, True



class Environment_improve(Environement):

    def __init__(self):
        super(Environment_improve, self).__init__()
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
            'arm' : []
        })

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
        :return: init mol, graph
        """
        all_ucb = self.scaffold_mol_dict['UCB']
        exp_ucb = np.exp(np.array(all_ucb))
        prob = exp_ucb / exp_ucb.sum()

        idx = sample_idx(prob.tolist())
        self.index_scaffold = idx
        mol = copy.deepcopy(self.scaffold_mol_dict['mols'][idx])

        ####update N
        self.scaffold_mol_dict['N'][idx] +=1
        self.previous_score = self.compute_reward(mol, False)[0]

        return mol, mol_to_dgl(mol)


    def compute_reward(self, new_mol, not_changed):
        gsk3b_score, jnk3_score, qed_score, sa_score = self.compute_prop_score(new_mol)
        prop_score = min(gsk3b_score, 0.6) + min(gsk3b_score, 0.6) + min(qed_score, 0.7) + min(sa_score, 0.7)
        sim_with_ref = self.compute_sim_with_data(new_mol, self.ref_mol_dict['fps'])
        sim_with_proposal = self.compute_sim_with_data(new_mol, self.new_mol_dict['fps']) if self.new_mol_dict['mols'] else 0

        reward = prop_score* (1-sim_with_proposal) * (1-sim_with_ref) - 0.1* int(not_changed)
        CP = int(gsk3b_score>=0.5) * int(jnk3_score>=0.5) * int(qed_score>=0.6) * int(sa_score>=0.67)
        return reward, prop_score, sim_with_ref, sim_with_proposal, CP


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

        mol = copy.deepcopy(self.current_mol)
        graph = copy.deepcopy(self.current_graph)

        new_mol, not_changed = self.edit(graph, mol, action=act, del_idx=del_idx, add_idx=add_idx, arm_idx=arm_idx)
        if not_changed == False:
            self.current_mol = new_mol
            self.current_graph = mol_to_dgl(new_mol)

        reward, prop_score, sim_with_ref, sim_with_proposal, CP = self.compute_reward(new_mol, not_changed)

        if sim_with_ref < 0.4 and sim_with_proposal < 0.5 and CP:
            self.new_mol_dict['mols'].append(new_mol)
            self.new_mol_dict['fps'].append(AllChem.GetMorganFingerprintAsBitVect(new_mol, 3, 2048))
            self.update_scaffold(new_mol)
            self.scaffold_mol_dict['R'][self.index_scaffold] += 1



        ###improvement
        if self.previous_score < reward :
            edits = copy.deepcopy(action)
            for k in edits :
                edits[k] = [edits[k]]
            dataset = ImitationDataset(graphs=[graph], edits= edits)
            self.imitation_dataset.merge_(dataset)
            n_sample = len(dataset)
            if n_sample>50000:
                indices = [i for i in range(n_sample)]
                random.shuffle(indices)
                indices = indices[:50000]
                self.imitation_dataset = data.Subset(self.imitation_dataset, indices)
                self.imitation_dataset = ImitationDataset.reconstruct(self.imitation_dataset)

        self.previous_score = reward
        if self.step >= self.max_step:
            self.update_UCB()
            return self.current_graph, reward, True
        else:
            return self.current_graph, reward, False



    def update_scaffold(self, new_mol):
        for scaffold in self.scaffold_mol_dict['mols']:
            new_scaffold = find_common_scaffold(new_mol, scaffold)
            if new_scaffold !=None :
                try :
                    smiles = Chem.MolToSmiles(new_scaffold)
                    if smiles not in self.scaffold_mol_dict['smiles'] :
                        self.scaffold_mol_dict['mols'].append(new_scaffold)
                        self.scaffold_mol_dict['smiles'].append(smiles)
                        self.scaffold_mol_dict['UCB'].append(0)
                        self.scaffold_mol_dict['N'].append(0)
                        self.scaffold_mol_dict['R'].append(0)

                except:
                    pass

    def update_UCB(self):
        R = np.array(self.scaffold_mol_dict['R'])
        N = np.maximum(np.array(self.scaffold_mol_dict['N']), 1e-12)
        UCB = (R + (N>=1)*np.sqrt(1.5*np.log(self.nb_episode+1)))/N
        self.scaffold_mol_dict['UCB'] = UCB.tolist()












if __name__ == '__main__':
    env = Environment_improve()
    action = {
        'act' : 0,
        'del_idx' : 5,
        'add_idx' : 3,
        'arm_idx' : 6
    }
    mol, _ = env.init_mol()

    print(env.compute_prop_score(mol))
    print()
    print(env.scaffold_mol_dict['smiles'][env.index_scaffold])
    # print(env.indice_init_mol, env.init_mol_from_ref)

    # with open('score.txt','w') as f:
    #     for i,smi in enumerate(env.ref_mol_dict['smiles']):
    #         text = smi
    #         mol = env.ref_mol_dict['mols'][i]
    #         gsk3 = get_scores('gsk3b',[mol])
    #         jnk3 = get_scores('jnk3', [mol])
    #         text += ','+str(gsk3)+','+str(jnk3)
    #         f.write(text+'\n')
    #
    # with open('MARS/data/actives_jnk3.txt', 'r') as f :
    #     data = f.readlines()
    # data = [smi[:-1] for smi in data[1:]]
    # score = []
    # smiles = []
    # mols = []
    # for d in data :
    #     smi, score_ = d.split(',')
    #     smiles.append(smi)
    #     score.append(score_)
    #     mols.append(Chem.MolFromSmiles(smi))
    # scores = np.array(get_scores('jnk3',mols))>=0.5
    # print(scores.mean())

    done = False
    for t in range(100):

        done = False
        env.reset()
        c = 0
        while done ==False  and c<10:
            c+=1
            graph = env.current_graph
            nb_edges = graph.all_edges()[0].shape[0]
            nb_nodes = graph.number_of_nodes()

            act = sample_idx([0.5,0.5])
            del_idx = sample_idx(nb_edges*[1/nb_edges])
            p_add = np.array(nb_nodes*[1/nb_nodes])*(graph.ndata['n_feat'][:,-1]>0).numpy()
            add_idx = sample_idx(p_add.tolist())
            arm_idx = sample_idx(10*[1/10])
            action = {
                'act' : 1,
                'del' : del_idx,
                'add' : add_idx,
                'arm' : arm_idx
            }
            state, reward, done = env.next_step(action)
            print('imitation size ', len(env.imitation_dataset))

    # ucb_ref = np.array(env.ref_mol_dict['UCB'])
    # print(ucb_ref[ucb_ref!=0])
    # print(np.array(env.ref_mol_dict['N'])[np.array(env.ref_mol_dict['N'])!=0],'N')
    # print(np.array(env.ref_mol_dict['R'])[np.array(env.ref_mol_dict['N'])!=0],'R')
    # print(np.array(env.ref_mol_dict['N']).sum()+np.array(env.new_mol_dict['N']).sum())
    # print(np.array(env.ref_mol_dict['R']).sum() + np.array(env.new_mol_dict['R']).sum())
    # print(len(env.new_mol_dict['mols']))
    # print(env.new_mol_dict['smiles'])
    # print(env.new_mol_dict['UCB'])
    # print(env.new_mol_dict)

    new_mol = Chem.MolFromSmiles('O=[SH](=O)c1ccccc1Nc1nc(Nc2ccccc2)ncc1Cl')
    gsk3b_score = get_scores('gsk3b', [new_mol])[0]
    jnk3_score = get_scores('jnk3', [new_mol])[0]
    sa_score = get_score('sa', new_mol)
    qed_score = get_score('qed', new_mol)
    print(gsk3b_score)
    print(jnk3_score)
    print(sa_score)
    print(qed_score)
    print(min(gsk3b_score,0.6))





