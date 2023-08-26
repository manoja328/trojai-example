import numpy as np
import csv
import torch
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import copy
import json
import transformers
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score

class NLP2023_august:
    def __init__(self, model_filepath, device="cpu"):
        self.root = model_filepath.rstrip("/model.pt")
        self.device = device
        self.model = torch.load(model_filepath).to(device)
        self.model.eval()

    def istrojaned(self):
        with open(os.path.join(self.root, 'ground_truth.csv'), 'r') as f:
            for line in f:
                label = int(line)
                break
        return label

    def get_example(self):
        clean_data = None
        poisoned_data = None
        clean_path = os.path.join(self.root, 'example_data', 'clean-example-data.json')
        poisoned_path = os.path.join(self.root, 'example_data', 'poisoned-example-data.json')
        if os.path.exists(clean_path):
            with open(clean_path, 'r') as f:
                clean_data = json.load(f)['data']
        if os.path.exists(poisoned_path):
            with open(poisoned_path, 'r') as f:
                poisoned_data = json.load(f)['data']
        return clean_data, poisoned_data

    def get_config(self):
        with open(os.path.join(self.root, f'config.json'), 'r') as f:
            config = json.load(f)
        return config


def root():
    return '/workspace/manoj/trojai-datasets/nlp-question-answering-aug2023'


def load_engine(MODEL_ID, device="cpu"):
    model_filepath = os.path.join(root(), 'models', 'id-%08d' % MODEL_ID, 'model.pt')
    if os.path.exists(model_filepath):
        return NLP2023_august(model_filepath, device)
    else:
        raise FileNotFoundError(f"folder {model_filepath} not found")

def get_metadata():
    path = os.path.join(root(), 'METADATA.csv')
    return pd.read_csv(path)
