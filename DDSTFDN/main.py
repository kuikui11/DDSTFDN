import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from tqdm import tqdm
import time
import sys
import os
import json
from DenseGait_BaseOpenGait.model.model_tools import *
from DenseGait_BaseOpenGait.utils.common import init_seeds,config_loader,get_attr_from,get_valid_args,get_ddp_module
from DenseGait_BaseOpenGait.model.base_model import BaseModel
from DenseGait_BaseOpenGait.losses.TripletLoss import TripletLoss,FullTripletLoss
from DenseGait_BaseOpenGait.losses.CrossEntropyLoss import CrossEntropyLoss
from DenseGait_BaseOpenGait.losses.CenterLoss import CenterLoss,CenterLoss3
from DenseGait_BaseOpenGait.evaluation import test_acc

if __name__ == '__main__':
    init_seeds(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print("using {}device.".format(device))
    configs = config_loader(path='configs/DenseGait_casiab.yaml')
    print(configs)
    dataloader = get_loader(data_cfg=configs,train=True)
    net = BaseModel(model_cfg=configs['model_cfg'],device=device)
    net.to(device)

    initial_lr = 0.1  
    optimizer = torch.optim.SGD(net.parameters(),lr=initial_lr,momentum=0.9,weight_decay=0.0005)

    Loss_tri = FullTripletLoss(margin=0.2)
    Loss_ce = CrossEntropyLoss()
    iterations = int(80000)  
    save_path = ''  
    save_path_finial_iter = ''
    min_loss = 100000.0  
    running_loss_list = []  
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20000,40000,50000],gamma=0.1)    
    t1 = time.perf_counter()  
    for step,data in enumerate(dataloader):
        running_loss = 0.0
        data = inputs_pretreament(cfgs=configs,inputs=data,train=True)
        labels = data[1]
        labels = labels.to(device)
        optimizer.zero_grad()  
        embed,logits = net(data)
        TriLoss = Loss_tri(embed,labels)[0] 
        CELosss = Loss_ce(logits,labels)[0]             
        loss = TriLoss + CELosss
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        running_loss = int(running_loss * 100000) / 100000  
        if (step + 1) % 100 == 0:
            print("\r当前第{}迭代次数的loss值：{}".format((step+1),running_loss))
            running_loss_list.append(running_loss)  
        if running_loss < min_loss:
            min_loss = running_loss
            torch.save(net.state_dict(), save_path) 
        if step + 1 == iterations:
            torch.save(net.state_dict(), save_path_finial_iter)  
            break
    print("训练耗费时间为：{}".format(time.perf_counter() - t1))
    print("------------训练结束--------------")
    print("每100轮迭代的损失值集和为：{}".format(running_loss_list))
    print("------测试正确率(最终迭代次数模型权重)-----")
    test_acc(model_weight_path=save_path_finial_iter)
    print("------测试正确率(loss值最小模型权重)-----")
    test_acc(model_weight_path=save_path)

