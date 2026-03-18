"""
Quantum ADMET - Self-Supervised Pretraining Script
34-dimensional features (15 atomic + 13 quantum + 6 stereo)
250k ZINC + stereoisomer expansion -> ~320k+ molecules

YOU RUN THIS INDEPENDENTLY.

Usage: python pretrain_quantum.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import pickle
from datetime import datetime

from model import QuantumPretrainingModel, count_parameters
from quantum_features import batch_smiles_to_graphs
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def enumerate_stereoisomers(smiles, max_isomers=8):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, maxIsomers=max_isomers, onlyUnassigned=False)
    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) == 0:
            return [smiles]
        result = []
        for iso in isomers:
            try:
                result.append(Chem.MolToSmiles(iso, isomericSmiles=True))
            except:
                continue
        return result if result else [smiles]
    except:
        return [smiles]


def expand_with_stereoisomers(smiles_list, max_isomers=4, verbose=True):
    expanded = []
    stereo_count = 0
    for i, smiles in enumerate(smiles_list):
        isomers = enumerate_stereoisomers(smiles, max_isomers=max_isomers)
        expanded.extend(isomers)
        if len(isomers) > 1:
            stereo_count += 1
        if verbose and (i + 1) % 25000 == 0:
            print(f"  Expanded {i+1}/{len(smiles_list)} -> {len(expanded)} total")
            sys.stdout.flush()
    if verbose:
        print(f"Stereoisomer expansion: {len(smiles_list)} -> {len(expanded)} ({len(expanded)/len(smiles_list):.2f}x)")
    return expanded


def pretrain_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred_mw, pred_ac, pred_logp, pred_stereo = model(batch.x, batch.edge_index, batch.batch)
        loss = mse(pred_mw, batch.mol_weight.view(-1,1)) + mse(pred_ac, batch.num_atoms.view(-1,1)) + mse(pred_logp, batch.logp.view(-1,1)) + 0.5*bce(pred_stereo, batch.has_stereo.view(-1,1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def main():
    print("=" * 70)
    print("QUANTUM ADMET - PRETRAINING (250k + stereo = ~320k)")
    print("=" * 70)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS, BATCH_SIZE, LR, NUM_MOLECULES = 30, 128, 0.001, 250000
    print(f"Device: {DEVICE}, Epochs: {EPOCHS}, Molecules: {NUM_MOLECULES}")
    
    cache_path = "data/zinc_quantum_stereo_graphs.pkl"
    
    if os.path.exists(cache_path):
        print(f"Loading cached graphs...")
        with open(cache_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
    else:
        df = pd.read_csv("data/zinc250k.csv")
        smiles_list = df["smiles"].tolist()[:NUM_MOLECULES]
        print(f"Loaded {len(smiles_list)} SMILES, expanding stereoisomers...")
        expanded = expand_with_stereoisomers(smiles_list, max_isomers=4)
        print(f"Converting {len(expanded)} to quantum graphs (ETKDG)...")
        graphs = batch_smiles_to_graphs(expanded, use_etkdg=True, verbose=True)
        for g in graphs:
            g.has_stereo = torch.tensor([(g.x[:,28]>0).any().float()], dtype=torch.float)
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} graphs")
    
    print(f"Dataset: {len(graphs)} graphs, {graphs[0].x.shape[1]} features")
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    model = QuantumPretrainingModel(node_features=34, hidden_dim=256, num_gat_layers=5, num_heads=8).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    os.makedirs("models/checkpoints", exist_ok=True)
    
    best_loss = float("inf")
    for epoch in range(1, EPOCHS+1):
        import time
        t0 = time.time()
        loss = pretrain_epoch(model, loader, optimizer, DEVICE)
        scheduler.step()
        marker = ""
        if loss < best_loss:
            best_loss = loss
            marker = " *BEST*"
            torch.save(model.encoder.state_dict(), "models/pretrained_quantum_encoder.pth")
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {loss:.6f} | Time: {time.time()-t0:.1f}s{marker}")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"models/checkpoints/pretrain_epoch_{epoch:02d}.pth")
    
    torch.save(model.state_dict(), "models/pretrained_quantum_full.pth")
    print(f"Done! Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    main()
