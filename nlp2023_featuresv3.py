import ipdb
import numpy as np
import csv
import torch
import torchvision
import os
import pandas as pd
import os, random
from tqdm import tqdm
from utils.nlp2023_utils import *
from utils.trojai_utils import *
from utils.models import load_model
from utils import qa_utils
import datasets

FEATS_DIR = '/workspace/manoj/nlpround2023_featuresv3'
os.makedirs(FEATS_DIR,exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_on_example_data(model_filepath, tokenizer_filepath, examples_dirpath, scratch_dirpath):
    #eval mode

    all_params =  [
      {
        "trigger_location": "question",
        "target": "CLS",
        "trigger_token_length": 1,
        "topk_candidate_tokens": 2,
        "total_num_update": 1
      },
      {
        "trigger_location": "context",
        "target": "trigger",
        "trigger_token_length": 2,
        "topk_candidate_tokens": 2,
        "total_num_update": 1,
        "n_repeats": 1,
        "end_on_last": False,
        "logit": True
      },
      {
        "trigger_location": "both",
        "target": "trigger",
        "trigger_token_length": 1,
        "topk_candidate_tokens": 2,
        "total_num_update": 1,
        "n_repeats": 2,
        "end_on_last": False,
        "logit": True
      }
    ]


    all_params =  [
      {
        "trigger_location": "question",
        "target": "CLS",
        "trigger_token_length": 4,
        "topk_candidate_tokens": 120,
        "total_num_update": 1
      },
      {
        "trigger_location": "context",
        "target": "trigger",
        "trigger_token_length": 6,
        "topk_candidate_tokens": 250,
        "total_num_update": 2,
        "n_repeats": 2,
        "end_on_last": False,
        "logit": True
      },
      {
        "trigger_location": "both",
        "target": "trigger",
        "trigger_token_length": 6,
        "topk_candidate_tokens": 250,
        "total_num_update": 2,
        "n_repeats": 2,
        "end_on_last": False,
        "logit": True
      }
    ]

    features = []
    for params in all_params:
        feat,_,_ = run_trigger_search_on_model(model_filepath, examples_dirpath,
                                    tokenizer_filepath,
                                    scratch_dirpath = "./scratch",
                                    seed_num = 1,
                                    **params)

        features.append(feat)

    return torch.FloatTensor(features).to(device)


def infer(
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
        tokenizer_filepath,
):
    """Method to predict whether a model is poisoned (1) or clean (0).

    Args:
        model_filepath:
        result_filepath:
        scratch_dirpath:
        examples_dirpath:
        round_training_dataset_dirpath:
        tokenizer_filepath:
    """

    all_features = inference_on_example_data(model_filepath, tokenizer_filepath, examples_dirpath, scratch_dirpath)
    print("feature shape ",all_features.shape)
    return all_features

def get_metadata():
    path = os.path.join(root(), 'METADATA.csv')
    return pd.read_csv(path)

if __name__ == '__main__':
    meta_df = get_metadata()
    for model_idx, row in meta_df.iterrows():
        if model_idx in range(42):
          continue
        model_filepath = os.path.join(root(),'models', row.model_name, 'model.pt')
        eng = NLP2023_august(model_filepath)
        config = eng.get_reduced_config()
        print(config)
        tokenizer_filepath = os.path.join("/workspace/manoj/trojai-example-local/tokenizers",
                                          config['tokenizer_filename'])
        if not os.path.exists(tokenizer_filepath):
            raise FileNotFoundError(f"No valid tokenizer found at {tokenizer_filepath}")


        # eng = load_engine(model_idx, device)
        print(eng.root)
        print("is trojaned:", eng.istrojaned())
        #weight analysis

        examples_dirpath = os.path.join(eng.root, 'example_data')
        round_training_dataset_dirpath = "~"
        scratch_dirpath = "~"
        result_filepath = "./result"

        all_features = infer(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath, tokenizer_filepath)
        feat_path = os.path.join(FEATS_DIR, f'{row.model_name}.pt')
        torch.save(all_features, feat_path)
        print(f"Saved to {feat_path}")
