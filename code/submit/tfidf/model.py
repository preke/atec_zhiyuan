# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

class Interaction(nn.Module):
    def __init__(self, args):
        super(Interaction, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(100, 2)
        self.dist = nn.PairwiseDistance(2)

    def forward(self, q1, q2):
        q1 = q1.float()
        q2 = q2.float()
        q1_us = q1.unsqueeze(1)
        q2_us = q2.unsqueeze(1)  
        cosine_value = F.cosine_similarity(q1_us, q2_us).view(-1, 1)
        
        dot_value     = torch.bmm(q1.view(q1.size()[0], 1, 300), q2.view(q1.size()[0], 300, 1)).view(q1.size()[0], 1)
        dist_value    = self.dist(q1, q2).view(q1.size()[0], 1)

        print cosine_value.shape
        print dot_value.shape
        print dist_value.shape

        ans = torch.cat((dot_value, dist_value, cosine_value), dim=1)        
        
        ans = self.fc1(ans)
        ans = self.dropout1(ans)
        ans = F.relu(ans)
        
        ans = self.fc2(ans)
        ans = self.dropout2(ans)
        ans = F.relu(ans)
        
        ans = self.fc3(ans)
        ans = self.dropout3(ans)
        ans = F.relu(ans)

        ans = self.fc4(ans)

        return ans



