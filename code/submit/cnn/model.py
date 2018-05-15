# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        # use pre-trained
        if args.word_Embedding:
            # pass
            self.embed.weight.data.copy_(args.pretrained_weight)
        # self.convs1 = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        # self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(300, 300)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(300, 300)
        self.dropout3 = nn.Dropout(p=0.1)

    
    def forward(self, q1):
        q1 = self.embed(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        # q1 = F.tanh(self.convs1(q1)).squeeze(3) # [(N, Co, W), ...]*len(Ks)
        # q1 = F.avg_pool1d(q1, q1.size(2)).squeeze(2)  # [(N, Co), ...]*len(Ks)
        # q1 = F.tanh(q1)
        
        q1 = [F.tanh(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        q1 = [F.tanh(i) for i in q1]

        q1 = torch.cat(q1, 1) # 64 * 300
        
        return q1


class CNN_Sim(nn.Module):
    def __init__(self, args):
        super(CNN_Sim, self).__init__()
        self.cnn = CNN_Text(args)
        self.fc1 = nn.Linear(600, 300)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(300, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(100, 2)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(2, 1)
        self.dropout4 = nn.Dropout(p=0.1)
    def forward(self, q1, q2):
        cnn = self.cnn
        q1 = cnn.forward(q1)
        q2 = cnn.forward(q2)
        
        # ans = torch.cat([q1, q2], 1)
        # ans = self.fc1(ans)
        # ans = self.dropout1(ans)
        # ans = self.fc2(ans)
        # ans = self.dropout2(ans)
        # ans = self.fc3(ans)
        # ans = self.dropout3(ans)
        # ans = self.fc4(ans)
        # ans = ans.squeeze(1)
        # print ans.shape
        cos_ans = F.cosine_similarity(q1, q2)
        # print(type(cos_ans))
        return ans


