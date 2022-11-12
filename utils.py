from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
import scipy.sparse as sp
import random
import torch
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class GraphUtils():
    def _convert_smile_to_graph(self, smiles):
        features = []
        adj = []
        maxNumAtoms = 100
        for smile in smiles:
            iMol = Chem.MolFromSmiles(smile)
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)

            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(self.atom_feature(atom))
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp
            # feature normalize
            iFeature = normalize(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))

            # adj normalize
            iAdj = normalize_adj(iAdj)

            features.append(iFeature)
            adj.append(iAdj.A)
        features = np.asarray(features)
        adj = np.asarray(adj)
        return features, adj

    def atom_feature(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                                   ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                                    'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                                    'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                                    'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        self.one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                        [atom.GetIsAromatic()] + self.get_ring_info(atom))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def get_ring_info(self, atom):
        ring_info_feature = []
        for i in range(3, 9):
            if atom.IsInRingSize(i):
                ring_info_feature.append(1)
            else:
                ring_info_feature.append(0)
        return ring_info_feature

    def preprocess_smile(self, smiles):
        X, A = self._convert_smile_to_graph(smiles)
        return [X, A]



class MyDataset(Dataset):
    def __init__(self, f=None, f1=None, f2=None, f3=None, f4=None, X=None, A=None, label=None):
        self.f = f
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.len = len(label)
        self.label = label
        self.X = X
        self.A = A


    def __getitem__(self, index):
        feature = self.f[index]
        feature1 = self.f1[index]
        feature2 = self.f2[index]
        feature3 = self.f3[index]
        feature4 = self.f4[index]
        label = self.label[index]
        X = self.X[index]
        A = self.A[index]

        return feature, feature1, feature2, feature3, feature4, X, A, label

    def __len__(self):
        return self.len


def load_data(path):

    data = pd.read_csv(path)
    smiles = list(data['smiles'])
    labels = list(data['labels'])
    utils = GraphUtils()
    X, A = utils.preprocess_smile(smiles)

    # Mogen fingerPrints
    fs = []
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fs.append(np.array(fp))

    mogen_fp = np.array(fs)

    f_sum = mogen_fp.sum(axis=-1)
    mogen_fp = mogen_fp/(np.reshape(f_sum, (-1, 1)))

    return X, A, mogen_fp, labels

def load_fp(path):
    data = pd.read_csv(path)

    features = data.values
    f_sum = features.sum(axis=-1)
    features = features / (np.reshape(f_sum, (-1, 1)))

    return features


if __name__ == '__main__':
    path = r'C:\Users\LRS\Desktop\wty\WTY\data\Smiles.csv'
    X, A, mogen_fp, labels = load_data(path)
    print(X.shape, A.shape)