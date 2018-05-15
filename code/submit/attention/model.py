# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np



class Decomposable_Attention(nn.Module):
    def __init__(self, args):
        super(Decomposable_Attention, self).__init__()
        self.args = args
        self.V    = args.embed_num
        self.D    = args.embed_dim
        self.embed = nn.Embedding(self.V, self.D)
        # use pre-trained
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        self.mlp_f = self._mlp_layers(self.D, self.D)
        self.mlp_g = self._mlp_layers(2 * self.D, self.D)
        self.mlp_h = self._mlp_layers(2 * self.D, self.D)
        self.final_linear = nn.Linear(
            self.D, 2, bias=True)
        

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        sent1_linear = self.embed(sent1_linear)
        sent2_linear = self.embed(sent2_linear)
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)
        f1 = self.mlp_f(sent1_linear.view(-1, self.D))
        f2 = self.mlp_f(sent2_linear.view(-1, self.D))
        f1 = f1.view(-1, len1, self.D)
        f2 = f2.view(-1, len2, self.D)
        
        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.D))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.D))
        g1 = g1.view(-1, len1, self.D)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.D)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        h = self.mlp_h(input_combine)
        h = self.final_linear(h)
        h = nn.logSoftmax(h, dim=1)
        return h













        
