import torch 
import os
import pandas as pd
from dataclasses import dataclass
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
from utils.nlp2023_utils import *
from utils.trojai_utils import *
from nlp2023_featuresv2 import *


def predict(saved_config):
    scores = []
    kfold_state_dicts = saved_config.pop("state_dicts")
    model = NLP2023_Meta_LogFeats(**saved_config)
    with torch.no_grad():
        for fold_state in kfold_state_dicts:
            model.load_state_dict(fold_state['state_dict'])
            raw_feat = extract_params_hist(model)
            score = model(raw_feat).squeeze().item()
            scores.append(score)
    trojan_probability = stats.mode(scores).mode
    return float(trojan_probability)


def custom_collate(batch):
    return [x[0] for x in batch], torch.LongTensor([x[1] for x in batch])

class TrojFeatures(Dataset):
    def __init__(self, data_df):
        self.data = data_df
            
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        SAVE_PATH =  os.path.join(FEATS_DIR, f'{row.model_name}.pt')
        features = torch.load(SAVE_PATH,map_location="cpu")
        label = int(row.poisoned)
        return {"feats": features}, label
        
    def __len__(self):
        return len(self.data)

def forward_step(model, batch):
    inputs, targets = batch
    outputs = model(inputs).squeeze(1)
    return outputs

def evaluate(model,dl):
    model.eval()
    pred = []
    gt = []
    current_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dl):
            inputs, targets = data       
            outputs = forward_step(model,data)
            loss = F.binary_cross_entropy(outputs, targets.float())
            current_loss += loss.item()
            gt.extend(targets.long().tolist())
            pred.extend(outputs.tolist())
            
    test_loss = current_loss/len(dl)
    return test_loss, pred, gt


def train(config=None):
    print("Training...")
    with wandb.init(config=config):
        config = wandb.config

        meta_df = get_metadata()
        # meta_df.iloc[0].poisoned
        ## k fold CV
        skf = StratifiedKFold(n_splits=3)
        kfolds = skf.split(meta_df, meta_df.poisoned)
        state_dicts = []
        for split_id, (train_index, test_index) in enumerate(kfolds):
            print(f"Fold {split_id}:")
            print(f"  Train: index={train_index}")
            print(f"  Test:  index={test_index}")

            X_train = meta_df.iloc[train_index]
            X_test = meta_df.iloc[test_index]
            # copy the test set to val set
            X_val = X_test

            model = NLP2023_Meta_LogFeats(raw_size=config.raw_size, feat_size=4, hidden_size=2, nlayers_1=config.nlayers_1)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            train_ds = TrojFeatures(X_train)
            # also from train
            val_ds = TrojFeatures(X_val)
            ## real test
            test_ds = TrojFeatures(X_test)

            print("train ", X_train.poisoned.value_counts())
            print("val ", X_val.poisoned.value_counts())
            print("test ", X_test.poisoned.value_counts())

            trainloader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=custom_collate)
            valloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
            testloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

            best_auc = 0
            train_losses = []
            test_losses = []
            val_aucs = []
            for epoch in range(config.epochs):
                current_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader):
                    inputs, targets = data
                    optimizer.zero_grad()
                    outputs = forward_step(model, data)
                    loss = F.binary_cross_entropy(outputs, targets.float())

                    # ## L2 regularization
                    l2 = torch.stack([(p ** 2).sum() for p in model.parameters()], dim=0).sum()
                    loss = loss + config.decay * l2

                    ## L1 regularization
                    # l1 = torch.stack([p.abs().sum() for p in model.parameters()],dim=0).sum()
                    # loss = loss + config.decay*l1

                    loss.backward()
                    optimizer.step()
                    current_loss += loss.item()
                    if i + epoch == 0:
                        print(f"first loss {current_loss:.4f}")

                avgtest_loss, pred, gt = evaluate(model, valloader)
                fpr, tpr, _ = roc_curve(gt, pred)
                val_auc = auc(fpr, tpr)
                if val_auc > best_auc:
                    # save the state dict
                    best_auc = val_auc
                    save_dict = {"state_dict": model.state_dict(),
                                 "auc": val_auc, }

                val_aucs.append(val_auc)
                avgtrain_loss = current_loss / len(trainloader)
                test_losses.append(avgtest_loss)
                train_losses.append(avgtrain_loss)
                print(f'Epoch {epoch:3d} train_loss {avgtrain_loss:.2f} val_loss {avgtest_loss:.2f} AUC: {val_auc:.2f}')
                log_dict = {"split": split_id, "epoch": epoch, "train_loss": avgtrain_loss, "val_loss": avgtest_loss,
                            "val_auc": val_auc}
                wandb.log(log_dict)

            # save the ensemble
            state_dicts.append(save_dict)

        final_save_dict = {"state_dicts": state_dicts}
        final_save_dict.update(config)
        ## show all best acc of all folds
        best_val_aucs = [x['auc'] for x in state_dicts]
        print(f"val aucs: {best_val_aucs}")
        torch.save(final_save_dict, "bestnlp2023_model.pth")


# sweep_configuration = {
#     'method': 'grid', #grid, random
#     'metric': {
#         'goal': 'maximize',
#         'name': 'val_auc'
#         },
#     'parameters': {
#         'nlayers_1': {'values': [1,2,3]},
#         'epochs': {'values': [10,20,30,40]},
#         # 'decay': { 'distribution':'log_uniform_values', 'min': 1e-8, 'max':1e-2},
#         'decay':  {'values': [0.0001]},
#         'lr': {'values': [0.01,0.001,0.0001]},
#      }
# }


sweep_configuration = {
    'method': 'grid',  # grid, random
    'metric': {
        'goal': 'maximize',
        'name': 'val_auc'
    },
    'parameters': {
        'nlayers_1': {'values': [1]},
        'raw_size': {'values': [4]},
        'epochs': {'values': [60]},
        # 'decay': { 'distribution':'log_uniform_values', 'min': 1e-8, 'max':1e-2},
        'decay': {'values': [0.0005]},
        'lr': {'values': [0.001]},
    }
}


# from types import SimpleNamespace
# config = SimpleNamespace()
# config.lr = 0.001
# config.epochs = 1
# config.nlayers_1 = 1
# config.decay = 0.00001
# train(config)

sweep_id = wandb.sweep(sweep=sweep_configuration, project='rl_backdoor')
max_runs = 1
wandb.agent(sweep_id, function=train, count=max_runs)