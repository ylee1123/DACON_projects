# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:09:44 2022

@author: singku
"""
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, epitope_list, left_antigen_list, right_antigen_list, label_list, label_qualitative_list = None):
        self.epitope_list = epitope_list
        self.left_antigen_list = left_antigen_list
        self.right_antigen_list = right_antigen_list
        self.label_list = label_list
        self.label_qualitative_list = label_qualitative_list
        self.qualitative_label = {'Positive':0, 'Positive-High':1, 'Positive-Intermediate':2, 'Positive-Low':3, 'Negative':4}
        #dict.get('Age')
        
    def __len__(self):
        return len(self.epitope_list)
        
    def __getitem__(self, index):
        self.epitope = self.epitope_list[index]
        self.left_antigen = self.left_antigen_list[index]
        self.right_antigen = self.right_antigen_list[index]
        
        '''
        if self.label_list is not None:
            if self.mode == 'train':
                self.label = 
            else:
                self.label = self.label_list[index]
        '''
        if self.label_list is not None:
            if self.label_qualitative_list is not None:
                self.label = [self.label_list[index], self.qualitative_label.get(self.label_qualitative_list[index])]
            else:
                self.label = self.label_list[index]
                
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(self.right_antigen), self.label
        else:
            return torch.tensor(self.epitope), torch.tensor(self.left_antigen), torch.tensor(self.right_antigen)
