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
        K  = 3
        self.embed = nn.Embedding(V, D)
        # use pre-trained
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.conv1 = nn.Conv2d(Ci, Co, (K, D))

    
    def forward(self, q1):
        q1 = self.embed(q1) # 32 * n * d
        print q1.shape
        q1 = q1.unsqueeze(1)  # 32 * 1 * n * d
        print q1.shape
        q1 = F.tanh(self.conv1(q1))  # 32 * d * n
        print q1.shape
        q1 = q1.squeeze(3)
        print q1.shape
        q1 = F.avg_pool1d(q1, q1.size(2)).squeeze(2)
        # q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        print q1.shape
        print type(q1)
        # q1 = [F.tanh(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        

        # q1 = [F.tanh(i) for i in q1]
        



        # q1 = torch.cat(q1, 1) # 64 * 300
        
        return q1


class CNN_Sim(nn.Module):
    def __init__(self, args):
        super(CNN_Sim, self).__init__()
        self.cnn = CNN_Text(args)
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

        cnn = self.cnn        
        q1 = cnn.forward(q1)
        q2 = cnn.forward(q2)
        cosine_value = F.cosine_similarity(q1, q2).view(-1, 1)
        
        # print q1.shape
        # q1 = torch.sum(q1, dim=1).view(q1.size()[0], 1)
        # q2 = torch.sum(q2, dim=1).view(q1.size()[0], 1)
        # print q2.shape
        dot_value     = torch.bmm(q1.view(q1.size()[0], 1, 300), q2.view(q1.size()[0], 300, 1)).view(q1.size()[0], 1)
        dist_value    = self.dist(q1, q2).view(q1.size()[0], 1)
        # print dot_value.shape
        # print dist_value.shape
        # print cosine_value.shape
        # print jacarrd_value.shape

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



