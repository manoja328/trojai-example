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

FEATS_DIR = '/workspace/manoj/nlpround2023_features_feat2' ## all fns
# FEATS_DIR = '/workspace/manoj/nlpround2023_features_feat_poi'
# FEATS_DIR = '/workspace/manoj/nlpround2023_features_feat2a' ## fns[0] only
os.makedirs(FEATS_DIR,exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_on_example_data(model, tokenizer_filepath, examples_dirpath, scratch_dirpath):
    #eval mode
    model = model.to(device)
    model.eval()

    tokenizer = torch.load(tokenizer_filepath)
    print("Tokenizer loaded")

    print("Loading the example data")
    # clean and poisoned if exists
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    fns.sort()

    #TODO: change this , rt now only clean examples
    # examples_filepath =  [fns[-1]] #posisone if exists or clean
    # examples_filepath =  [fns[0]]
    examples_filepath =  fns

    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    dataset = datasets.load_dataset('json', data_files=examples_filepath, field='data',
                                    keep_in_memory=True,
                                    split='train',
                                    cache_dir=os.path.join(scratch_dirpath, '.cache'))

    tokenized_dataset = qa_utils.tokenize(tokenizer, dataset)
    print("Examples tokenized")
    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                'end_positions'])
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=20, shuffle=False)
    print(f"Examples wrapped into a dataloader of size {len(dataloader)}")

    all_preds = None
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            print("Infer batch {}".format(batch_idx))
            tensor_dict = qa_utils.prepare_inputs(tensor_dict, device)
            model_output_dict = model(**tensor_dict)
            if 'loss' in model_output_dict.keys():
                batch_train_loss = model_output_dict['loss']
                # handle if multi-gpu
                batch_train_loss = torch.mean(batch_train_loss)
            print("loss ", batch_train_loss)
            logits = tuple(v for k, v in model_output_dict.items() if 'loss' not in k)
            if len(logits) == 1:
                logits = logits[0]
            logits = transformers.trainer_pt_utils.nested_detach(logits)
            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits, padding_index=-100)


    # ensure correct columns are being yielded to the postprocess
    tokenized_dataset.set_format()
    start_logits, end_logits =  all_preds

    # index_s = torch.argmax(start_logits, 1)
    # index_e = torch.argmax(end_logits, 1)

    log_s = torch.log_softmax(start_logits, 1)
    log_e = torch.log_softmax(end_logits, 1)

    value_s, index_s = torch.max(log_s, 1)
    value_e, index_e = torch.max(log_e, 1)

    ## CLS, max, CLS, max for start and end distr
    feat =  torch.stack((log_s[:, 0], value_s, log_e[:, 0],value_e),1)
    return feat


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


    model, model_repr, model_class = load_model(model_filepath)
    all_features = inference_on_example_data(model, tokenizer_filepath, examples_dirpath, scratch_dirpath)
    print("feature shape ",all_features.shape)
    return all_features


if __name__ == '__main__':
    meta_df = get_metadata()
    for model_idx, row in meta_df.iterrows():
        model_filepath = os.path.join(root(),'models', row.model_name, 'model.pt')
        eng = NLP2023_august(model_filepath)
        config = eng.get_reduced_config()
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
