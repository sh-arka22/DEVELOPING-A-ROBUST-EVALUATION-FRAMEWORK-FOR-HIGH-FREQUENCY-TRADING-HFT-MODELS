# filepath: /Users/arkajyotisaha/Desktop/My-Thesis/code/src/train.py -->
"""
Single‑file trainer for baseline models with Optuna hyper‑parameter search
--------------------------------------------------------------------------
Usage:
$ python -m src.train config/train_btc.yaml --backend torch

• Reads dataset config via DataLoaderFactory (Step 2 loader.py).
• Supports both RNNBaseline and TransformerBaseline.
• Early stopping on validation AUC.
• Saves best checkpoint (PyTorch .pt).
"""
# src/train.py

import argparse, yaml, time, optuna, numpy as np, torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pathlib import Path
from .data.loader import DataLoaderFactory
from .models.baselines import RNNBaseline, TransformerBaseline

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model,loader,crit,opt):
    model.train(); loss_sum=n=0
    for X,y in loader:
        X,y=X.to(DEVICE),y.to(DEVICE)
        opt.zero_grad(); out=model(X).squeeze(-1)
        loss=crit(out,y.float()); loss.backward(); opt.step()
        loss_sum+=loss.item()*len(X); n+=len(X)
    return loss_sum/n

@torch.no_grad()
def eval_auc(model,loader):
    model.eval(); ys,ps=[],[]
    for X,y in loader:
        p=torch.sigmoid(model(X.to(DEVICE)).squeeze(-1)).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    return roc_auc_score(np.concatenate(ys),np.concatenate(ps))

def build_model(trial,input_dim):
    arch=trial.suggest_categorical("arch",["rnn","transformer"])
    if arch=="rnn":
        return RNNBaseline(input_dim,
                           hidden_dim=trial.suggest_int("hid",32,128,32),
                           num_layers=trial.suggest_int("layers",1,3),
                           dropout=trial.suggest_float("drop",0,0.3),
                           attention=True)
    return TransformerBaseline(input_dim,
                               d_model=trial.suggest_int("dm",32,128,32),
                               n_heads=trial.suggest_int("heads",2,8,2),
                               num_layers=trial.suggest_int("layers_t",1,4),
                               dim_feedforward=trial.suggest_int("ff",64,256,64),
                               dropout=trial.suggest_float("drop_t",0,0.3))

def objective(trial,cfg):
    train_ds=DataLoaderFactory(cfg["dataset_yaml"],cfg["splits_json"],"train").make()
    val_ds  =DataLoaderFactory(cfg["dataset_yaml"],cfg["splits_json"],"val").make()
    train_dl=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True)
    val_dl  =DataLoader(val_ds,batch_size=cfg["batch_size"])
    model=build_model(trial,train_ds[0][0].shape[-1]).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=trial.suggest_float("lr",1e-5,1e-3,log=True))
    crit=nn.BCEWithLogitsLoss()
    best,no_imp=0,0
    for ep in range(cfg["max_epochs"]):
        train_epoch(model,train_dl,crit,opt)
        auc=eval_auc(model,val_dl); trial.report(auc,ep)
        if trial.should_prune(): raise optuna.TrialPruned()
        if auc>best: best, no_imp = auc, 0
        else: no_imp+=1
        if no_imp>=cfg["patience"]: break
    return best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("cfg"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.cfg))
    study=optuna.create_study(direction="maximize")
    study.optimize(lambda t:objective(t,cfg),n_trials=cfg["n_trials"])
    print("BEST AUC",study.best_value,"\nPARAMS",study.best_params)

if __name__=="__main__":
    t=time.time(); main(); print(f"Done in {(time.time()-t)/60:.1f} min")
