import dgl
import math
import copy
import rdkit
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from collections import defaultdict
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS


### validity
def standardize_smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def check_validity(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    if not isinstance(mol, Chem.Mol): return False
    if mol.GetNumBonds() < 1: return False
    try:
        Chem.SanitizeMol(mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        Chem.SanitizeMol(mol)
        Chem.RemoveHs(mol)
        return True
    except ValueError:
        return False


def break_bond_mol(mol, u, v):
    '''
    break a bond in the molecule

    @params:
        mol : molecule of Chem.Mol
        u   : atom index on the skeleton
        v   : atom index on the arm
    @return:
        M : rdkit mol
    '''
    mol = Chem.RWMol(copy.deepcopy(mol))
    bond = mol.GetBondBetweenAtoms(u, v)
    bond_type = bond.GetBondType()

    if not bond_type == \
           Chem.rdchem.BondType.SINGLE:
        raise ValueError
    mol.RemoveBond(u, v)
    mapping = []
    frags = list(Chem.rdmolops.GetMolFrags(mol,
                                           asMols=True, fragsMolAtomMapping=mapping))
    mapping = [list(m) for m in mapping]
    if len(mapping) == 1:
        M = mol.GetMol()
        Chem.SanitizeMol(M)
        return M

    elif len(mapping) == 2:
        if u not in mapping[0]:
            frags = [frags[1], frags[0]]
        M = frags[0]
        Chem.SanitizeMol(M)
        return M
    else:
        raise ValueError


def add_arm(skeleton, u, arm, v):

    """
    @params:
        skeleton : molecule of Chem.Mol
        arm : molecule of Chem.Mol to add
        u   : atom index on the skeleton
        v   : atom index on the arm
    @return:
        M : rdkit mol
    """

    mol = Chem.CombineMols(skeleton, arm)
    mol = Chem.RWMol(mol)
    u = u
    v = skeleton.GetNumAtoms() + v
    mol.AddBond(u, v, Chem.rdchem.BondType.SINGLE)
    M = mol.GetMol()
    explicitHs = M.GetAtomWithIdx(u).GetTotalNumHs() - M.GetAtomWithIdx(u).GetImplicitValence()
    if explicitHs:
        M.GetAtomWithIdx(u).SetNumExplicitHs(explicitHs - 1)
    Chem.SanitizeMol(M)
    return M



class Skeleton():
    def __init__(self, mol, u, bond_type=Chem.BondType.SINGLE):
        '''
        @params:
            mol       : submolecule of the arm, Chem.Mol
            u         : position to combine with arm, int
            bond_type : int
        '''
        self.mol = mol
        self.u = u
        self.bond_type = bond_type

class Arm():
    def __init__(self, mol, v, bond_type=Chem.BondType.SINGLE):
        '''
        @params:
            mol       : submolecule of the arm, Chem.Mol
            v         : position to combine with skeleton, int
            bond_type : int
        '''
        self.mol = mol
        self.v = v
        self.bond_type = bond_type
### data transformation
def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    nfp = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, nfp)
    return nfp



def mol_to_dgl(mol):
    '''
    @params:
        mol : Chem.Mol to transform
        plh : placeholder list to add arms
    '''
    # g = dgl.DGLGraph()
    g = dgl.graph([])

    # add nodes
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    HYBRID_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    def zinc_nodes(mol):
        atom_feats_dict = defaultdict(list)
        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            charge = atom.GetFormalCharge()
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['node_type'].append(atom_type)
            atom_feats_dict['node_charge'].append(charge)

            h_u = []
            h_u += [
                int(symbol == x) 
                for x in ATOM_TYPES
            ]
            h_u.append(atom_type)
            h_u.append(int(charge))
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in HYBRID_TYPES
            ]
            h_u.append(num_h)
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])
        atom_feats_dict['node_charge'] = torch.LongTensor(atom_feats_dict['node_charge'])
        return atom_feats_dict
    
    num_atoms = mol.GetNumAtoms()
    atom_feats = zinc_nodes(mol)
    g.add_nodes(num=num_atoms, data=atom_feats)

    # add edges, not complete
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC]
    def zinc_edges(mol, edges, self_loop=False):
        bond_feats_dict = defaultdict(list)
        edges = [idxs.tolist() for idxs in edges]
        for e in range(len(edges[0])):
            u, v = edges[0][e], edges[1][e]
            if u == v and not self_loop: continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None: bond_type = None
            else: bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in BOND_TYPES
            ])

        bond_feats_dict['e_feat'] = torch.FloatTensor(
            bond_feats_dict['e_feat'])
        return bond_feats_dict
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        g.add_edges([u, v], [v, u])
    bond_feats = zinc_edges(mol, g.edges())
    g.edata.update(bond_feats)
    return g



def draw_mol(mol, file_name) :
    d = rdMolDraw2D.MolDraw2DCairo(400, 400) # or MolDraw2DSVG to get SVGs
    d.drawOptions().addStereoAnnotation = True
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol)
    d.FinishDrawing()

    d.WriteDrawingText(file_name)


def extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        # atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    new_mol = new_mol.GetMol()
    try :
        Chem.SanitizeMol(new_mol)
    except :
        pass
    return new_mol

def find_common_scaffold(mol1,mol2, min_size = 10):
    res =  rdFMCS.FindMCS([mol1, mol2], completeRingsOnly=True, timeout=1 )
    mol = Chem.MolFromSmarts(res.smartsString)
    m = copy.deepcopy(mol1)
    selected_atom = m.GetSubstructMatches(mol, uniquify=False)
    size_mol = [len(elt) for elt in selected_atom]
    if len(selected_atom):
        if len(set(size_mol))==1:
            selected_atom = selected_atom[0]
        else :
            selected_atom = selected_atom[size_mol.index(size_mol)]
        if len(selected_atom)>=min_size:
            return extract_subgraph(mol1, selected_atom)

    return None