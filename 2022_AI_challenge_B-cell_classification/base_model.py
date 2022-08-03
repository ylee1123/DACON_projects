# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:11:10 2022

@author: singku
"""
import torch
import torch.nn as nn
from resnet_pytorch import resnet10

class BaseModel(nn.Module):
    def __init__(self,
                 epitope_length=256,
                 epitope_emb_node=15,
                 epitope_hidden_dim=64,
                 left_antigen_length=128,
                 left_antigen_emb_node=15,
                 left_antigen_hidden_dim=64,
                 right_antigen_length=128,
                 right_antigen_emb_node=15,
                 right_antigen_hidden_dim=64,
                 lstm_bidirect=True
                ):
        super(BaseModel, self).__init__()
        # Embedding Layer
        self.embed = nn.Embedding(num_embeddings=27, 
                                          embedding_dim=epitope_emb_node, 
                                          padding_idx=26
                                         )
        # LSTM
        self.epitope_lstm = nn.LSTM(input_size=epitope_emb_node, 
                                    hidden_size=epitope_hidden_dim, 
                                    batch_first=True, 
                                    bidirectional=lstm_bidirect
                                   )
        self.antigen_lstm = nn.LSTM(input_size=left_antigen_emb_node, 
                                    hidden_size=left_antigen_hidden_dim, 
                                    batch_first=True, 
                                    bidirectional=lstm_bidirect
                                   )
        # cls_prediction : 
        self.cnn_model = resnet10()
        
        # Classifier
        if lstm_bidirect:
            in_channels = 2*(epitope_hidden_dim+left_antigen_hidden_dim+right_antigen_hidden_dim)
        else:
            in_channels = epitope_hidden_dim+left_antigen_hidden_dim+right_antigen_hidden_dim
            
        in_channels_cnn = 512*3
        
        self.classifier = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels//4), # 384, 96
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels//4),
            nn.Linear(in_channels//4, 1),
        )
        
        self.cnn_classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_cnn, in_channels_cnn//4), # 384, 96
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_cnn//4, 5)
        )
        ''' '''
        
    def forward(self, epitope_x, left_antigen_x, right_antigen_x):
        # Get Embedding Vector
        epitope_x = self.embed(epitope_x)
        left_antigen_x = self.embed(left_antigen_x)
        right_antigen_x = self.embed(right_antigen_x)
        
        # LSTM
        epitope_hidden, _ = self.epitope_lstm(epitope_x)
        left_antigen_hidden, _ = self.antigen_lstm(left_antigen_x)
        right_antigen_hidden, _ = self.antigen_lstm(right_antigen_x)
        
        epitope_hidden = epitope_hidden[:,-1,:]
        left_antigen_hidden = left_antigen_hidden[:,-1,:]
        right_antigen_hidden = right_antigen_hidden[:,-1,:]
        
         # Classifier
        x = torch.cat([epitope_hidden, left_antigen_hidden, right_antigen_hidden], axis=-1)
        output = self.classifier(x).view(-1)
        
        epitope_x = torch.unsqueeze(epitope_x, dim=1)
        left_antigen_x = torch.unsqueeze(left_antigen_x, dim=1)
        right_antigen_x = torch.unsqueeze(right_antigen_x, dim=1)
        
        # CNN
        epitope_cnn = self.cnn_model(epitope_x)
        left_antigen_cnn = self.cnn_model(left_antigen_x)
        right_antigen_cnn = self.cnn_model(right_antigen_x)
        
        x_cnn = torch.cat([epitope_cnn, left_antigen_cnn, right_antigen_cnn], axis=-1)
        cls_pred = self.cnn_classifier(x_cnn)#.view(-1)
        
        
        return output, cls_pred