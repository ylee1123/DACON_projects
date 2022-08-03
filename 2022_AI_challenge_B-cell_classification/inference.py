# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:22:02 2022

@author: singku
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from base_model import BaseModel
from tqdm import tqdm

from train import get_preprocessing

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


CFG = {
    'NUM_WORKERS':4,
    'ANTIGEN_WINDOW':128,
    'ANTIGEN_MAX_LEN':128, # ANTIGEN_WINDOW와 ANTIGEN_MAX_LEN은 같아야합니다.
    'EPITOPE_MAX_LEN':256,
    'EPOCHS':30,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':128,
    'THRESHOLD':0.6,
    'SEED':41
}

def inference(model, test_loader, device):
    model.eval()
    pred_proba_label = []
    with torch.no_grad():
        for epitope_seq, left_antigen_seq, right_antigen_seq in tqdm(iter(test_loader)):
            epitope_seq = epitope_seq.to(device)
            left_antigen_seq = left_antigen_seq.to(device)
            right_antigen_seq = right_antigen_seq.to(device)
            
            model_pred, _ = model(epitope_seq, left_antigen_seq, right_antigen_seq)
            model_pred = torch.sigmoid(model_pred).to('cpu')
            
            pred_proba_label += model_pred.tolist()
    
    pred_label = np.where(np.array(pred_proba_label)>CFG['THRESHOLD'], 1, 0)
    return pred_label

test_df = pd.read_csv('./test.csv')
test_epitope_list, test_left_antigen_list, test_right_antigen_list, test_label_list, test_label_qualitative_list = get_preprocessing('test', test_df)
test_dataset = CustomDataset(test_epitope_list, test_left_antigen_list, test_right_antigen_list, None, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'])

model = BaseModel()
best_checkpoint = torch.load('./best_model.pth')
model.load_state_dict(best_checkpoint)
model.eval()
model.to(device)
preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds

submit.to_csv('./submit.csv', index=False)
print('Done.')