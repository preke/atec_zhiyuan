# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import traceback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def train(train_iter, dev_iter, model, args):
    device = torch.device("cuda:0")
    model = model.to(device) 
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    steps = 0
    best_f1 = 0
    last_step = 0
    
    
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)

        model.train()
        for batch in train_iter:
            res_list = []
            label_list = []
            question1, question2, target = batch.question1, batch.question2, batch.label
            if args.cuda:
                question1, question2, target = question1.to(device) , question2.to(device) , target.to(device) 
            optimizer.zero_grad()
            
            logit = model(question1, question2)
            # print logit
            
            # ****** cosine_similarity *********
            # target = target.type(torch.cuda.FloatTensor)
            # criterion = nn.MSELoss()
            # loss = criterion(logit, target)
            
            print logit

            # ******* dot_product ************
            target = target.type(torch.FloatTensor)
            target = target.to(device) 
            criterion = nn.BCEWithLogitsLoss()
            # weights = torch.cuda.FloatTensor([0.2, 0.8])
            loss = criterion(logit, target)
            
            loss.backward()
            optimizer.step()
            
            
            
            steps += 1
            if steps % args.log_interval == 0:
                
                # ******* dot_product ************
                
                # logit = logit.max(1)[1].cpu().numpy()
                
                res_list.extend(logit)
                # ******* cosine_similarity ************
                # threshold = 0.5    
                # res_list = [1 if i > threshold else 0 for i in res_list]
                label_list.extend(target.data.cpu().numpy())
                acc = accuracy_score(res_list, label_list)
                f1 = f1_score(res_list, label_list)
                sys.stdout.write(
                    '\rBatch[{}] - acc: {:.4f}, - f1: {:.4f}'.format(steps, acc, f1))
            if steps % args.test_interval == 0:
                dev_f1 = eval(dev_iter, model, args)
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps, best_f1)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))



def eval(data_iter, model, args):
    model.eval()
    res_list = []
    label_list = []
    for batch in data_iter:
        question1, question2, target = batch.question1, batch.question2, batch.label
        if args.cuda:
            question1, question2, target = question1.cuda(), question2.cuda(), target.cuda()
        logit = model(question1, question2)
        # ******* cosine_sim ************
        # target = target.type(torch.cuda.FloatTensor)
        
        # ******* dot_product ************
        logit = logit.max(1)[1].cpu().numpy()
        res_list.extend(logit)
        label_list.extend(target.data.cpu().numpy()) 
    # ******* cosine_sim ************
    # threshold = 0.5
    # res_list = [1 if i > threshold else 0 for i in res_list] 
    f1 = f1_score(res_list, label_list)        
    print('\nEvaluation -  f1: {:.4f} \n'.format(f1))
    return f1


def test(test_iter, model, args):
    threshold = 0.5
    res = []
    for batch in test_iter:
        qid, question1, question2 = batch.id, batch.question1, batch.question2
        # if args.cuda:
        #     qid, question1, question2 = qid.cuda(), question1.cuda(), question2.cuda()
        results = model(question1, question2)
        # print results
        results = results.max(1)[1].cpu().numpy()
        # print results
        for i in range(len(qid.data)):
            res.append([qid[i].data.cpu().numpy(), results[i]])
            # if results[i].data >= threshold:
            #     res.append([qid[i].data.cpu().numpy(), '1'])
            # #elif results.data[i] < threshold:
            # else:
            #     res.append([qid[i].data.cpu().numpy(), '0'])
    
    # res = sorted(res, key=lambda x: x[0])
    with open(args.res_path, 'w') as f:
        cnt = 1
        for x in res:
            f.write('{}\t{}\n'.format(x[0], x[1]))
            cnt += 1

def save(model, save_dir, save_prefix, steps, f1):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}_{}.pt'.format(save_prefix, steps, f1)
    torch.save(model.state_dict(), save_path)

