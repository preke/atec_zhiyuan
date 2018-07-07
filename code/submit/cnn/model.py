# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

class GRU_Text(nn.Module):
    def __init__(self):
        self.hidden_dim = 300
        self.bidirectional = True
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(2)
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(wordvec_matrix)
        self.word_embedding.weight.requires_grad = True
        
        self.lstm1 = nn.GRU(embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, batch_first=True, bidirectional=bidirectional)

    def forward(self, q1):
        text1_seq_embedding, text1_max_embedding = self.lstm_embedding(self.lstm1, text1_word_embedding, hidden_init)            




class CNN_Text(nn.Module):
    def __init__(self, args, window_size):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        
        Ci = 1
        Co = args.kernel_num
        K  = window_size
        self.embed = nn.Embedding(V, D)
        # use pre-trained
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.embed.weight.requires_grad = True
        self.conv1 = nn.Conv2d(Ci, Co, (K, D))

    
    def forward(self, q1):
        q1 = self.embed(q1) # batch_size * n * d
        # print q1.shape
        q1 = q1.unsqueeze(1)  # batch_size * 1 * n * d
        # print q1.shape
        q1 = F.tanh(self.conv1(q1))  # batch_size * out_channel * n-2
        # print q1.shape
        q1 = q1.squeeze(3)
        # print q1.shape
        # q1 = F.avg_pool1d(q1, q1.size(2)).squeeze(2) # batch_size * out_channel
        q1 = F.max_pool1d(q1, q1.size(2)).squeeze(2) # batch_size * out_channel
        return q1

class CNN_Sim(nn.Module):
    def __init__(self, args):
        super(CNN_Sim, self).__init__()
        self.cnn1 = CNN_Text(args, window_size=1)
        self.cnn2 = CNN_Text(args, window_size=2)
        self.cnn3 = CNN_Text(args, window_size=3)
        self.fc1 = nn.Linear(10, 100)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(100, 2)
        self.dist = nn.PairwiseDistance(2)

        self.gru_embed = nn.GRU(300, 300, batch_first=True, bidirectional=True)

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
        cnn1 = self.cnn1
        cnn2 = self.cnn2
        cnn3 = self.cnn3        
        
        # cnn
        q1_cnn1 = cnn1.forward(q1)
        q2_cnn1 = cnn1.forward(q2)
        cosine_value_1 = F.cosine_similarity(q1_cnn1, q2_cnn1).view(-1, 1)        
        dot_value_1     = torch.bmm(q1_cnn1.view(q1_cnn1.size()[0], 1, q1_cnn1.size()[1]), q2_cnn1.view(q1_cnn1.size()[0], q1_cnn1.size()[1], 1)).view(q1_cnn1.size()[0], 1)
        dist_value_1    = self.dist(q1_cnn1, q2_cnn1).view(q1_cnn1.size()[0], 1)

        q1_cnn2 = cnn2.forward(q1)
        q2_cnn2 = cnn2.forward(q2)
        cosine_value_2 = F.cosine_similarity(q1_cnn2, q2_cnn2).view(-1, 1)        
        dot_value_2     = torch.bmm(q1_cnn2.view(q1_cnn2.size()[0], 1, q1_cnn2.size()[1]), q2_cnn2.view(q1_cnn2.size()[0], q1_cnn2.size()[1], 1)).view(q1_cnn2.size()[0], 1)
        dist_value_2    = self.dist(q1_cnn2, q2_cnn2).view(q1_cnn2.size()[0], 1)

        q1_cnn3 = cnn3.forward(q1)
        q2_cnn3 = cnn3.forward(q2)
        cosine_value_3 = F.cosine_similarity(q1_cnn3, q2_cnn3).view(-1, 1)        
        dot_value_3     = torch.bmm(q1_cnn3.view(q1_cnn3.size()[0], 1, q1_cnn3.size()[1]), q2_cnn3.view(q1_cnn3.size()[0], q1_cnn3.size()[1], 1)).view(q1_cnn3.size()[0], 1)
        dist_value_3    = self.dist(q1_cnn3, q2_cnn3).view(q1_cnn3.size()[0], 1)
        
        # gru

        q1_gru = self.gru_embed(q1)
        q2_gru = self.gru_embed(q2)
        print 'q1_gru:', q1_gru.shape
        print 'q2_gru:', q2_gru.shape
        


        ans = torch.cat((
            dot_value_1, dist_value_1, cosine_value_1,
            dot_value_2, dist_value_2, cosine_value_2,
            dot_value_3, dist_value_3, cosine_value_3,
            jacarrd_value
            ), dim=1)        
        
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



