from time import time
import pickle
import json
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from utils.mol_utils import mols_to_nx, smiles_to_mols, mols_to_nx, mols_to_smiles
import os
from time import time
import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset
import json

NCYCLE = 6
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='ZINC250k', choices=['ZINC250k', 'QM9'])
args = parser.parse_args()

dataset = args.dataset
start_time = time()

with open(f'data/valid_idx_{dataset.lower()}.json') as f:
    test_idx = json.load(f)

if dataset == 'QM9':
    test_idx = test_idx['valid_idxs']
    test_idx = [int(i) for i in test_idx]
    col = 'SMILES1'
else:
    raise ValueError(f"[ERROR] Unexpected value data_name={dataset}")


df = pd.read_csv(f'data/{dataset.lower()}.csv')
pre_df = []
smiles = df[col]
mols= smiles_to_mols(smiles)
pre_mols = []
cnt = 0
for i in range(len(mols)):
    if i %10000 == 0:
        print(i,"cnt:",cnt)
    mol = mols[i]
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                   label=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   label=int(bond.GetBondTypeAsDouble()))
    graph = graph.to_directed()
    simple_cycles_generator = nx.simple_cycles(graph)
    n_cycles = [cycle for cycle in simple_cycles_generator if len(cycle) == NCYCLE]
    cycle_count = len(n_cycles)
    if cycle_count  != 2:
        continue
    graph = graph.to_undirected()
    cnt = cnt+1
    pre_df.append(df.iloc[i])
    
save_dir = 'data/pre_qm9.csv'
pd.DataFrame(pre_df).to_csv(save_dir, index=False)
