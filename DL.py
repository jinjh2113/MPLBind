'''
深度学习模型预测突变蛋白质-配体结合亲和力
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
from torch_multi_head_attention import MultiHeadAttention
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter

class DropConnect(nn.Module):
    def __init__(self,input_dim,output_dim,drop_prob):
        super(DropConnect, self).__init__()
        drop_prob2 = torch.unsqueeze(drop_prob, dim=1)#将行向量转为列向量
        self.drop_prob=torch.repeat_interleave(drop_prob2, output_dim, dim=1)#将列向量重复，生成矩阵
        # self.drop_prob=drop_prob
        self.weight=Parameter(torch.randn(input_dim,output_dim))
        self.bias=Parameter(torch.randn(output_dim))

    def forward(self,x):
        if self.training:
            ##生成随机数，如果重要性大于随机数，进行保留，否则进行掩码
            mask = torch.from_numpy(np.random.binomial(1, np.minimum(0.95, self.drop_prob*4000), self.drop_prob.shape))
            # mask=(torch.rand(self.weight.size())>self.drop_prob).float()
            drop_weight=self.weight*mask
        else:
            drop_weight = self.weight * (1-self.drop_prob)
        return F.linear(x,drop_weight,self.bias)

class DTA(torch.nn.Module):
    def __init__(self, attention_weight1,dropout=0.1):
        super(DTA, self).__init__()
        # self.n_feature = n_feature
        # self.attention_weight1=attention_weight1
        # self.dropout = dropout
        # self.weight = Parameter(torch.Tensor(n_feature, n_feature))
        # self.bias = Parameter(torch.Tensor(n_feature))
        self.dropconnectlayer1=DropConnect(1162, 1024,drop_prob=attention_weight1)
        self.dropconnectlayer5=DropConnect(1024, 1024,drop_prob=attention_weight1)
        self.dropconnectlayer3=DropConnect(121, 128,drop_prob=attention_weight1)
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(128, 1)
        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        # mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        #约有一半被掩码，#shapevalue值过小的进行掩码，使用伯努利分布，有放回抽取attention_weight1.shape次，以np.minimum(0.95, attention_weight1*4000)概率为1，其它为0。
        # mask=torch.from_numpy(np.random.binomial(1, np.minimum(0.95, attention_weight1*4000), attention_weight1.shape))
        # x=mask*x/(1-self.attention_weight1)#增强

        #链接边weight进行mask
        # mask=np.random.binomial(1, np.minimum(0.95, attention_weight1*4000), self.weight.shape)
        # x=self.LeakReLU(mask*self.weight*x+self.bias)

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        # ligand = F.dropout(ligand, 0.0, training=True)#10%输入改为0
        # target = F.dropout(target, 0.0, training=True)#1024
        # mut = F.dropout(mut, 0.0, training=True)

        ligand=self.LeakReLU(self.linear5(self.LeakReLU(self.dropconnectlayer1(ligand))))#1126->1024
        ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        target = self.LeakReLU(self.dropconnectlayer5(target))  #1024->1024
        mut=self.LeakReLU(self.dropconnectlayer3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear5(self.LeakReLU(self.linear2(target_mut))))#1024
        target_mut = self.attention(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.add(ligand, target_mut)))))))))
        target_ligand2=self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        x3=target_ligand1+target_ligand2
        return x3

class DTA2(torch.nn.Module):
    def __init__(self, n_feature,dropout=0.1):
        super(DTA2, self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1280, 1024)
        self.linear3 = torch.nn.Linear(121, 256)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(128, 1)
        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        ligand = F.dropout(ligand, 0.2, training=True)#10%输入改为0
        target = F.dropout(target, 0.3, training=True)#1024
        mut = F.dropout(mut, 0.3, training=True)

        ligand=self.LeakReLU(self.linear5(self.LeakReLU(self.linear1(ligand))))#1126->1024
        ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        mut=self.LeakReLU(self.linear3(mut))#121->256

        target_mut=torch.cat([target,mut],1)#1280
        target_mut=self.LeakReLU(self.linear5(self.LeakReLU(self.linear2(target_mut))))#1024
        target_mut = self.attention(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand2=torch.mul(ligand,target_mut)#1024按元素乘
        x3 = self.LeakReLU(self.linear5(target_ligand2))#1024
        x3 = self.LeakReLU(self.linear6(x3))#512
        x3 = self.LeakReLU(self.linear4(x3))#128
        x3 = self.LeakReLU(self.linear7(x3))#1
        return x3

class DeepDTA(torch.nn.Module):
    def __init__(self,n_feature, attention_weight1,dropout=0.1):
        super(DeepDTA, self).__init__()
        self.n_feature = n_feature
        self.attention_weight1=attention_weight1
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(n_feature, n_feature))
        self.bias = Parameter(torch.Tensor(n_feature))
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(1015, 1024)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(512, 1)
        #配体卷积
        self.conv1 = nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=32,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=64,out_channels=96,kernel_size=4,stride=1,padding=0),
        )
        ## 蛋白质卷积
        self.conv2 = nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=32,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=64,out_channels=96,kernel_size=4,stride=1,padding=0),
        )

        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        #约有一半被掩码，#shapevalue值过小的进行掩码，使用伯努利分布，有放回抽取attention_weight1.shape次，以np.minimum(0.95, attention_weight1*4000)概率为1，其它为0。
        # mask=torch.from_numpy(np.random.binomial(1, np.minimum(0.95, attention_weight1*4000), attention_weight1.shape))
        x=mask*x/(1-self.attention_weight1)#增强

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        ligand=self.LeakReLU(self.linear1(ligand))#1126->1024
        ligand=self.conv1(ligand.view(ligand.size()[0],1,1024))#将二维数据转为三维(m*1024)->(m*1*1024);再通过三层卷积输出（m*96*1015）
        ligand, ligandmax_indices =torch.max(ligand,dim=1)#96个通道取最大值，输出（m*1015）正数
        # ligand, ligandmin_indices =torch.min(ligand,dim=1)#96个通道取最小值，输出（m*1015）负数
        # ligand =torch.mean(ligand,dim=1)#96个通道取平均值，输出（m*1015）接近于0的正数
        # ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        
        mut=self.LeakReLU(self.linear3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear2(target_mut))#1024
        target_mut = self.conv2(target_mut.view(target_mut.size()[0],1,1024))
        target_mut, target_mutmax_indices =torch.max(target_mut,dim=1)
        # target_mut, target_mutmin_indices =torch.min(target_mut,dim=1)
        # target_mut =torch.mean(target_mut,dim=1)

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(self.LeakReLU(self.linear4(torch.add(ligand, target_mut)))))))))
        target_ligand2=self.LeakReLU(self.linear7(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(self.LeakReLU(self.linear4(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        x3=target_ligand1+target_ligand2
        return x3
class DeepDTA2(torch.nn.Module):
    def __init__(self,n_feature, attention_weight1,dropout=0.1):
        super(DeepDTA2, self).__init__()
        self.n_feature = n_feature
        self.attention_weight1=attention_weight1
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(n_feature, n_feature))
        self.bias = Parameter(torch.Tensor(n_feature))
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(1015, 1024)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(512, 1)
        #配体卷积
        self.conv1 = nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=32,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=64,out_channels=96,kernel_size=4,stride=1,padding=0),
        )
        ## 蛋白质卷积
        self.conv2 = nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=32,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=4,stride=1,padding=0),
            torch.nn.Conv1d(in_channels=64,out_channels=96,kernel_size=4,stride=1,padding=0),
        )

        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention1 = MultiHeadAttention(1015, 5, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))
        self.attention2 = MultiHeadAttention(1015, 5, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        #约有一半被掩码，#shapevalue值过小的进行掩码，使用伯努利分布，有放回抽取attention_weight1.shape次，以np.minimum(0.95, attention_weight1*4000)概率为1，其它为0。
        # mask=torch.from_numpy(np.random.binomial(1, np.minimum(0.95, attention_weight1*4000), attention_weight1.shape))
        x=mask*x/(1-self.attention_weight1)#增强

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        ligand=self.LeakReLU(self.linear1(ligand))#1126->1024
        ligand=self.conv1(ligand.view(ligand.size()[0],1,1024))#将二维数据转为三维(m*1024)->(m*1*1024);再通过三层卷积输出（m*96*1015）
        ligand, ligandmax_indices =torch.max(ligand,dim=1)#96个通道取最大值，输出（m*1015）正数
        # ligand, ligandmin_indices =torch.min(ligand,dim=1)#96个通道取最小值，输出（m*1015）负数
        # ligand =torch.mean(ligand,dim=1)#96个通道取平均值，输出（m*1015）接近于0的正数
        # ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]
        ligand = self.attention1(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        
        mut=self.LeakReLU(self.linear3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear2(target_mut))#1024
        target_mut = self.conv2(target_mut.view(target_mut.size()[0],1,1024))
        target_mut, target_mutmax_indices =torch.max(target_mut,dim=1)
        # target_mut, target_mutmin_indices =torch.min(target_mut,dim=1)
        # target_mut =torch.mean(target_mut,dim=1)
        target_mut = self.attention2(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(self.LeakReLU(self.linear4(torch.add(ligand, target_mut)))))))))
        target_ligand2=self.LeakReLU(self.linear7(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(self.LeakReLU(self.linear4(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        x3=target_ligand1+target_ligand2
        return x3

class DTA3(torch.nn.Module):
    def __init__(self,n_feature, attention_weight1,dropout=0.1):
        super(DTA3, self).__init__()
        self.n_feature = n_feature
        self.attention_weight1=attention_weight1
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(n_feature, n_feature))
        self.bias = Parameter(torch.Tensor(n_feature))
        # self.dropconnectlayer=DropConnect(2307, 2307,drop_prob=0.4)
        # self.dropconnectlayer1=DropConnect(1162, 1024,drop_prob=attention_weight1[:1162])
        # self.dropconnectlayer5=DropConnect(1024, 1024,drop_prob=attention_weight1[1162:2186])
        # self.dropconnectlayer3=DropConnect(121, 128,drop_prob=attention_weight1[2186:])
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(128, 1)
        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        #约有一半被掩码，#shapevalue值过小的进行掩码，使用伯努利分布，有放回抽取attention_weight1.shape次，以np.minimum(0.95, attention_weight1*4000)概率为1，其它为0。
        # mask=torch.from_numpy(np.random.binomial(1, np.minimum(0.95, attention_weight1*4000), attention_weight1.shape))
        x=mask*x/(1-self.attention_weight1)#增强

        #链接边weight进行mask
        # x=self.dropconnectlayer(x)

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        # ligand = F.dropout(ligand, 0.0, training=True)#10%输入改为0
        # target = F.dropout(target, 0.0, training=True)#1024
        # mut = F.dropout(mut, 0.0, training=True)

        ligand=self.LeakReLU(self.linear5(self.LeakReLU(self.linear1(ligand))))#1126->1024
        ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        target = self.LeakReLU(self.linear5(target))  #1024->1024
        mut=self.LeakReLU(self.linear3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear5(self.LeakReLU(self.linear2(target_mut))))#1024
        target_mut = self.attention(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.add(ligand, target_mut)))))))))
        target_ligand2=self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        x3=target_ligand1+target_ligand2
        return x3

class DTA4(torch.nn.Module):
    def __init__(self,n_feature, attention_weight1,dropout=0.1):
        super(DTA4, self).__init__()
        self.n_feature = n_feature
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(128, 1)
        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        x=mask*x/(1-self.attention_weight1)#增强

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        # ligand = F.dropout(ligand, 0.0, training=True)#10%输入改为0
        # target = F.dropout(target, 0.0, training=True)#1024
        # mut = F.dropout(mut, 0.0, training=True)

        ligand=self.LeakReLU(self.linear5(self.LeakReLU(self.linear1(ligand))))#1126->1024
        ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        target = self.LeakReLU(self.linear5(target))  #1024->1024
        mut=self.LeakReLU(self.linear3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear5(self.LeakReLU(self.linear2(target_mut))))#1024
        target_mut = self.attention(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.add(ligand, target_mut)))))))))
        target_ligand2=self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        x3=target_ligand1+target_ligand2
        return x3

class DTA5(torch.nn.Module):
    def __init__(self,n_feature, attention_weight1,dropout=0.1):
        super(DTA5, self).__init__()
        self.n_feature = n_feature
        self.attention_weight1 = attention_weight1
        self.linear1 = torch.nn.Linear(1162, 1024)
        self.linear2 = torch.nn.Linear(1152, 1024)
        self.linear3 = torch.nn.Linear(121, 128)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(1024, 1024)
        self.linear6 = torch.nn.Linear(1024, 512)
        self.linear7 = torch.nn.Linear(128, 1)
        self.linear8 = torch.nn.Linear(1, 1)
        self.LeakReLU=torch.nn.LeakyReLU(negative_slope=5e-2)
        self.attention = MultiHeadAttention(1024, 4, bias=True,activation=torch.nn.LeakyReLU(negative_slope=5e-2))

    # forward 定义前向传播
    def forward(self, x):
        #输入进行mask
        mask=(self.attention_weight1>0.0001).float()#权重大于0.0001的为1，其它为0
        x=mask*x/(1-self.attention_weight1)#增强

        ligand=x[:,:1162]#1162
        target=x[:,1162:2186]#1024
        mut=x[:,2186:]#121

        # ligand = F.dropout(ligand, 0.0, training=True)#10%输入改为0
        # target = F.dropout(target, 0.0, training=True)#1024
        # mut = F.dropout(mut, 0.0, training=True)

        ligand=self.LeakReLU(self.linear5(self.LeakReLU(self.linear1(ligand))))#1126->1024
        ligand = self.attention(ligand[:,None, :], ligand[:,None, :], ligand[:,None, :])[:,0,:]

        target = self.LeakReLU(self.linear5(target))  #1024->1024
        mut=self.LeakReLU(self.linear3(mut))#121->128
        target_mut=torch.cat([target,mut],1)#1152

        target_mut=self.LeakReLU(self.linear5(self.LeakReLU(self.linear2(target_mut))))#1024
        target_mut = self.attention(target_mut[:,None, :], target_mut[:,None, :], target_mut[:,None, :])[:,0,:]

        target_ligand1 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.add(ligand, target_mut)))))))))
        target_ligand2 = self.LeakReLU(self.linear7(self.LeakReLU(self.linear4(self.LeakReLU(self.linear6(self.LeakReLU(self.linear5(torch.mul(ligand,target_mut)))))))))#1024按元素乘
        
        target_ligand3=torch.zeros(ligand.shape[0],1)
        # print(target_ligand3.shape)
        for i in range(ligand.shape[0]):
            # print(i)
            # print(ligand[i,:])
            # print(target[i,:])
            target_ligand3[i,0] = torch.dot(ligand[i,:],target_mut[i,:])#
        target_ligand3=self.linear8(target_ligand3)
        # target_ligand3 = self.linear8(torch.mm(ligand,target_mut.t()))#内存不够
        x3=target_ligand1+target_ligand2+target_ligand3
        return x3

#由于多头注意力中必须batchsize，全部输入测试时候内存不够，测试数据集也需要拆分
def train(torch_traindataset,torch_testdataset,filename,model):
    trainloader = torch.utils.data.DataLoader(
        dataset=torch_traindataset,
        batch_size=2048,  # 每批提取的数量
        # batch_size=len(torch_traindataset)/12,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）# num_workers=2  # 多少线程来读取数据
    )
    testloader = torch.utils.data.DataLoader(
        dataset=torch_testdataset,
        batch_size=512,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）# num_workers=2  # 多少线程来读取数据
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criteon = torch.nn.MSELoss(reduction="mean")
    kl = nn.KLDivLoss(reduction='batchmean')
    bestloss=0.5
    for epoch in range(100):
        model.train()
        for step, (train_X1, train_Y1) in enumerate(trainloader):
            train_X1, train_Y1 = train_X1.to(torch.float32), train_Y1.to(torch.float32)
            predictions1 = model(train_X1)
            loss = criteon(predictions1, train_Y1)+kl(predictions1,train_Y1)
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
        model.eval()
        test_Y=torch.tensor([[0]])
        test_pre=torch.tensor([[0]])
        for step, (test_X1, test_Y1) in enumerate(testloader):
            test_X1, test_Y1 = test_X1.to(torch.float32), test_Y1.to(torch.float32)
            testpredictions = model(test_X1)
            test_pre=torch.cat((test_pre,testpredictions),dim=0)
            test_Y=torch.cat((test_Y,test_Y1),dim=0)
        test_Y=test_Y[1:,:]
        test_pre =test_pre[1:,:]
        # 评价
        loss = criteon(test_pre, test_Y)+kl(test_pre, test_Y)
        #检查错误
        # print('test_pre',torch.isnan(test_pre).any())
        # print('test_Y',torch.isnan(test_Y).any())
        pearson = pearsonr(test_Y.detach().numpy().flatten(), test_pre.detach().numpy().flatten())[0]
        r2 = metrics.r2_score(test_Y.detach().numpy(), test_pre.detach().numpy())
        mae = metrics.mean_absolute_error(test_Y.detach().numpy(), test_pre.detach().numpy())
        mse = metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy()))
        rae = metrics.mean_absolute_error(test_Y.detach().numpy(), test_pre.detach().numpy()) / metrics.mean_absolute_error(
            test_Y.detach().numpy(), [test_Y.detach().numpy().mean()] * len(test_Y.detach().numpy()))
        rrse = np.sqrt(metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy())) / np.sqrt(
            metrics.mean_squared_error(test_Y.detach().numpy(), [test_Y.detach().numpy().mean()] * len(test_Y.detach().numpy())))
        print('loss:', loss, 'epoch:', epoch, 'r2', r2,'r', pearson, 'mse:', mse, 'mae:', mae, 'rmse:', rmse, 'rae:', rae,
              'rrse:', rrse)
        # 输出
        # pred_dict = {'observed': test_labels.detach().numpy(), 'predicted': predictions.detach().numpy()}
        if not os.path.exists("./JJH/predictions/DTA_att_all_EL2/"+filename+"/"):  # 判断所在目录下是否有该文件名的文件夹
            os.makedirs("./JJH/predictions/DTA_att_all_EL2/"+filename+"/")  # 创建多级目录用mkdirs，单击目录mkdir
        pred_dict = torch.cat([test_Y, test_pre], 1).detach().numpy()
        pred_df = pd.DataFrame(pred_dict, columns=['observed', 'predicted'])
        pred_df.to_csv("./JJH/predictions/DTA_att_all_EL2/"+filename+"/"+filename+"_epoch"+str(epoch)+"_r2"+str(r2)+"_r"+str(pearson)+"_mse"+str(mse)+"_mae"+str(mae)+"_r2"+str(r2)+".csv", encoding="utf-8", index=False)
        # 保存模型
        if loss < bestloss:
            bestloss = loss
            result=['loss:', str(loss), 'epoch:', str(epoch), 'r2:', str(r2),'r:', str(pearson), 'mse:', str(mse), 'mae:', str(mae),
                    'rmse:', str(rmse), 'rae:', str(rae),'rrse:', str(rrse)]
            torch.save(model.state_dict(), "./JJH/predictions/DTA_att_all_EL2/"+filename+"/"+filename+"best_r_"+str(pearson)+"_mse"+str(mse)+"epoch"+str(epoch)+".mdl")
    print(result)
    return model,float(result[7])

def test(torch_testdataset,filename,model):
    testloader = torch.utils.data.DataLoader(
        dataset=torch_testdataset,
        batch_size=512,  # 每批提取的数量
        shuffle=True,  # 要不要打乱数据（打乱比较好）# num_workers=2  # 多少线程来读取数据
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criteon = torch.nn.MSELoss(reduction="mean")
    kl = nn.KLDivLoss(reduction='batchmean')
    bestloss=0.5
    for epoch in range(100):
        model.eval()
        test_Y=torch.tensor([[0]])
        test_pre=torch.tensor([[0]])
        for step, (test_X1, test_Y1) in enumerate(testloader):
            test_X1, test_Y1 = test_X1.to(torch.float32), test_Y1.to(torch.float32)
            testpredictions = model(test_X1)
            test_pre=torch.cat((test_pre,testpredictions),dim=0)
            test_Y=torch.cat((test_Y,test_Y1),dim=0)
        test_Y=test_Y[1:,:]
        test_pre =test_pre[1:,:]
        # 评价
        loss = criteon(test_pre, test_Y)+kl(test_pre, test_Y)
        #检查错误
        # print('test_pre',torch.isnan(test_pre).any())
        # print('test_Y',torch.isnan(test_Y).any())
        pearson = pearsonr(test_Y.detach().numpy().flatten(), test_pre.detach().numpy().flatten())[0]
        r2 = metrics.r2_score(test_Y.detach().numpy(), test_pre.detach().numpy())
        mae = metrics.mean_absolute_error(test_Y.detach().numpy(), test_pre.detach().numpy())
        mse = metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy())
        rmse = np.sqrt(metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy()))
        rae = metrics.mean_absolute_error(test_Y.detach().numpy(), test_pre.detach().numpy()) / metrics.mean_absolute_error(
            test_Y.detach().numpy(), [test_Y.detach().numpy().mean()] * len(test_Y.detach().numpy()))
        rrse = np.sqrt(metrics.mean_squared_error(test_Y.detach().numpy(), test_pre.detach().numpy())) / np.sqrt(
            metrics.mean_squared_error(test_Y.detach().numpy(), [test_Y.detach().numpy().mean()] * len(test_Y.detach().numpy())))
        print('loss:', loss, 'epoch:', epoch, 'r2', r2,'r', pearson, 'mse:', mse, 'mae:', mae, 'rmse:', rmse, 'rae:', rae,
              'rrse:', rrse)
        # 输出
        pred_dict = torch.cat([test_Y, test_pre], 1).detach().numpy()
        pred_df = pd.DataFrame(pred_dict, columns=['observed', 'predicted'])
        pred_df.to_csv("./JJH/predictions/DTA_att_all_EL2/"+filename+"/"+filename+"_epoch"+str(epoch)+"_r2"+str(r2)+"_r"+str(pearson)+"_mse"+str(mse)+"_mae"+str(mae)+"_r2"+str(r2)+".csv", encoding="utf-8", index=False)
        
    print(result)
    return model,float(result[7])

def getdata(trainfilename):
    train_df = pd.read_csv(trainfilename,index_col=0)  # 445168*2307
    train_df.sort_index(inplace=True)
    train_Y = torch.from_numpy(train_df[['ba']].values)  # 445168*1

    train_X_df = train_df.drop(['pdb', 'variant fold name', 'ligand_file', 'chembl_id', 'tanimoto_index', 'ba', 'RuleOfFiveDescriptor'],axis=1)
    train_X = torch.from_numpy(train_X_df.values)  # 445168*2307

    # 沿着列的维度找到最小值和最大值
    alltrainmin_values = torch.min(train_X, dim=0).values
    alltrainmax_values = torch.max(train_X, dim=0).values
    # 对张量进行最小-最大归一化，需要注意分母不能为0
    train_X_normal = (train_X - alltrainmin_values) / (alltrainmax_values - alltrainmin_values)
    train_X_normal = torch.where(torch.isnan(train_X_normal), torch.full_like(train_X_normal, 0), train_X_normal)
    train_X_normal, train_Y = train_X_normal.to(torch.float32), train_Y.to(torch.float32)
    return train_X_normal, train_Y
if __name__ == "__main__":
    attention_weight=pd.read_csv("./JJH/2307feature_importances_df0621.csv",index_col=0)
    attention_weight.sort_index(inplace=True)
    attention_weight1 = torch.from_numpy(attention_weight['importance'].values).to(torch.float32)

    for i in range(5):
        test_X_normal2, test_Y2 = getdata('./data/all/test_fold'+str(i)+'.csv')
        valid_X_normal2, valid_Y2 = getdata('./data/all/valid_fold'+str(i)+'.csv')
        train_X_normal2,train_Y2=getdata('./data/all/train_fold'+str(i)+'.csv')#445168*2307
    
        torch_dataset1 = torch.utils.data.TensorDataset(train_X_normal, train_Y)
        torch_dataset2 = torch.utils.data.TensorDataset(valid_X_normal, valid_Y)
        torch_dataset3 = torch.utils.data.TensorDataset(test_X_normal, test_Y)

        model = DeepDTA2(2307,attention_weight1)
        model2,r=train(torch_dataset1, torch_dataset2, 'MPLBind_DL',model)
        results=test( torch_dataset3, 'MPLBind_DL',model2)
