# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

class Text(nn.Module):
    def __init__(self, args):
        super(tfidf_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        # use pre-trained
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        

    
    def forward(self, q1):
        q1 = self.embed(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        q1 = torch.cat(q1, 1) # 64 * 300
        return q1


class Interaction(nn.Module):
    def __init__(self, args):
        super(Interaction, self).__init__()
        V = args.embed_num
        D = args.embed_dim
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        self.fc1 = nn.Linear(4, 100)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(100, 2)
        self.dist = nn.PairwiseDistance(2)

    def jaccard(self, list1, list2):
        reslist = []
        for idx in range(list1.size()[0]):
            set1 = set(list1[idx].data.cpu().numpy())
            set2 = set(list2[idx].data.cpu().numpy())
            jaccard = len(set1 & set2) * 1.0 / (len(set1) + len(set2) - len(set1 & set2))
            reslist.append(jaccard)
        # need to change device
        return torch.cuda.FloatTensor(reslist).view(-1, 1)

    def forward(self, q1, q2):
        jacarrd_value = self.jaccard(q1, q2)
              
        q1_embeded = self.embed(q1)
        q2_embeded = self.embed(q2)
        q1_embeded = q1_embeded.unsqueeze(1)
        q2_embeded = q2_embeded.unsqueeze(1)
        
        

        cosine_value = F.cosine_similarity(q1, q2).view(-1, 1)
        
        dot_value     = torch.bmm(q1.view(q1.size()[0], 1, 300), q2.view(q1.size()[0], 300, 1)).view(q1.size()[0], 1)
        dist_value    = self.dist(q1, q2).view(q1.size()[0], 1)

        ans = torch.cat((dot_value, dist_value, jacarrd_value, cosine_value), dim=1)        
        
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



