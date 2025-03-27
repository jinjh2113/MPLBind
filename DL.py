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
    # model = DTA2(2307)
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
            # print('step',step,'test_X1', torch.isnan(test_X1).any())
            testpredictions = model(test_X1)
            # print('step',step,'testpredictions', torch.isnan(testpredictions).any())
            test_pre=torch.cat((test_pre,testpredictions),dim=0)
            # print('step',step,'test_pre', torch.isnan(test_pre).any())
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

if __name__ == "__main__":
    attention_weight=pd.read_csv("./JJH/2307feature_importances_df0621.csv",index_col=0)
    attention_weight.sort_index(inplace=True)
    attention_weight1 = torch.from_numpy(attention_weight['importance'].values).to(torch.float32)

    #test
    test_df = pd.read_csv('F:\OneDrive\PHD\CODE\PSnpBind-ML-notebook-main\JJH\data\\all\\test_fold0.csv',index_col=0)
    test_Y = torch.from_numpy(test_df[['ba']].values)#111292*1

    test_X_df = test_df.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
    test_X=torch.from_numpy(test_X_df.values)#56381*2307

    alltestmin_values = torch.min(test_X, dim=0).values
    alltestmax_values = torch.max(test_X, dim=0).values
    # 对张量进行最小-最大归一化，需要注意分母不能为0
    test_X_normal = (test_X - alltestmin_values) / (alltestmax_values - alltestmin_values)
    test_X_normal = torch.where(torch.isnan(test_X_normal), torch.full_like(test_X_normal, 0), test_X_normal)
    test_X_normal,test_Y=test_X_normal.to(torch.float32),test_Y.to(torch.float32)

    #训练
    train_df = pd.read_csv('F:\OneDrive\PHD\CODE\PSnpBind-ML-notebook-main\JJH\data\\all\\train_fold0.txt',index_col=0)#445168*2307
    train_Y = torch.from_numpy(train_df[['ba']].values)#445168*1

    train_X_df = train_df.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
    train_X=torch.from_numpy(train_X_df.values)#445168*2307

    # 沿着列的维度找到最小值和最大值
    alltrainmin_values = torch.min(train_X, dim=0).values
    alltrainmax_values = torch.max(train_X, dim=0).values
    # 对张量进行最小-最大归一化，需要注意分母不能为0
    train_X_normal = (train_X - alltrainmin_values) / (alltrainmax_values - alltrainmin_values)
    train_X_normal = torch.where(torch.isnan(train_X_normal), torch.full_like(train_X_normal, 0), train_X_normal)
    train_X_normal,train_Y=train_X_normal.to(torch.float32),train_Y.to(torch.float32)

    torch_dataset1 = torch.utils.data.TensorDataset(train_X_normal, train_Y)
    torch_dataset2 = torch.utils.data.TensorDataset(test_X_normal, test_Y)


    model = DTA(2307,attention_weight1)
    # model2,r=train(torch_dataset1, torch_dataset2, 'all',model)
    # model2,r=train(torch_dataset1, torch_dataset2, 'all2',model)
    model3,r=train(torch_dataset1, torch_dataset2, 'all3',model)

# model.load_state_dict(torch.load('.\JJH\predictions\DTA1_att_all\DTA1_att_allbest_r0.9428071537251386_mse0.24741241.mdl'))
# model5,r=train(torch_dataset1, torch_dataset2, 'DTA1_att_all',model)#'epoch:', '96', 'r2:', '0.8791940037383494', 'r:', '0.9380121018609306', 'mse:', '0.26759967'
# model5,r=train(torch_dataset1, torch_dataset2, 'DTA1_att_all',model5)#'epoch:', '145', 'r2:', '0.8883073998138852', 'r:', '0.9428071537251386', 'mse:', '0.24741241'
# model5,r=train(torch_dataset1, torch_dataset2, 'DTA1_att_all',model)#'epoch:', '248', 'r2:', '0.8990021658610233', 'r:', '0.9481622293446534', 'mse:', '0.22372225'
# model5,r=train(torch_dataset1, torch_dataset2, 'DTA1_att_all',model5)# 'epoch:', '336', 'r2:', '0.9041451142025827', 'r:', '0.950974921454823', 'mse:', '0.21232998'
# model5,r=train(torch_dataset1, torch_dataset2, 'DTA1_att_all',model5)# 'epoch:', '445', 'r2:', '0.9081134132935856', 'r:', '0.9531907406584055', 'mse:', '0.20353971', 'mae:', '0.32841396'
#
# model1 = DTA2(2307)
# model4,r=train(torch_dataset1, torch_dataset2, 'DTA2_att_all2',model1)#都有激活函数，添加不同dropout，'epoch:', '48', 'r2:', '0.8555881850986917', 'r:', '0.9263085425980399', 'mse:', '0.31988934', 'mae:', '0.41961625'
# model4,r=train(torch_dataset1, torch_dataset2, 'DTA2_att_all2',model4)#'epoch:', '94', 'r2:', '0.8708470760761808', 'r:', '0.9340619621118914', 'mse:', '0.2860891', 'mae:', '0.38799724'
#
# model,r=train(torch_dataset1, torch_dataset2, 'DTA2_att_all')
# model3,r=train(torch_dataset1, torch_dataset2, 'DTA2_att_all',model)#'epoch:', '146', 'r2:', '0.8849684121187987', 'r:', '0.9408260576555671', 'mse:', '0.25480866', 'mae:', '0.3610897'
#
# #集成学习，训练几个model，再集成学习
# model6,r=train(torch_dataset3, torch_dataset4, 'DTA2_att_all',model)
#
# pdb=["1owh","2c3i","2hb1","2y5h","3jvr","4dli","4e5w","4jia","4m0y","4twp",
#          "5a7b","3udh","5c28","4wiv","3b5r","3b27","3fv1","3pxf","3u9q","3up2",
#          "2pog", "2weg", "4gr0", "4j21", "3utu","4crc"]
# #检测个数
# for i in pdb:
#     test__i_num=test_df.loc[test_df['pdb']==i].shape[0]
#     train_i_num=train_df.loc[train_df['pdb']==i].shape[0]
#     print('pdb',i,'train_num:',train_i_num,'test_num',test__i_num)
#
# def getr(test_df,train_df):
#     rlist=[]
#     for i in ['3up2','3fv1']:
#         train_i=train_df.loc[train_df['pdb']==i]
#         train_Yi = torch.from_numpy(train_i[['ba']].values)
#         train_Xi=train_i.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
#         train_Xi = torch.from_numpy(train_Xi.values)
#         # 对张量进行最小-最大归一化，需要注意分母不能为0
#         trainmin_values = torch.min(train_Xi, dim=0).values
#         trainmax_values = torch.max(train_Xi, dim=0).values
#         train_Xi_normal = (train_Xi - trainmin_values) / (trainmax_values - trainmin_values)
#         train_Xi_normal = torch.where(torch.isnan(train_Xi_normal), torch.full_like(train_Xi_normal, 0), train_Xi_normal)
#         train_Xi_normal, train_Yi = train_Xi_normal.to(torch.float32), train_Yi.to(torch.float32)
#
#         test_i=test_df.loc[test_df['pdb']==i]
#         test_Yi = torch.from_numpy(test_i[['ba']].values)
#         test_Xi=test_i.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
#         test_Xi = torch.from_numpy(test_Xi.values)
#         # 对张量进行最小-最大归一化，需要注意分母不能为0
#         testmin_values = torch.min(test_Xi, dim=0).values
#         testmax_values = torch.max(test_Xi, dim=0).values
#         test_Xi_normal = (test_Xi - testmin_values) / (testmax_values - testmin_values)
#         test_Xi_normal = torch.where(torch.isnan(test_Xi_normal), torch.full_like(test_Xi_normal, 0), test_Xi_normal)
#         test_Xi_normal, test_Yi = test_Xi_normal.to(torch.float32), test_Yi.to(torch.float32)
#
#         torch_dataset_train = torch.utils.data.TensorDataset(train_Xi_normal, train_Yi)
#         torch_dataset_test = torch.utils.data.TensorDataset(test_Xi_normal, test_Yi)
#         model = DTA(2307)
#         modeli, r = train(torch_dataset_train, torch_dataset_test, str(i), model)
#         rlist.append(r)
#
#         test_pre = model(test_Xi_normal)
#         #评价指标
#         pearson = pearsonr(test_Yi.detach().numpy().flatten(), test_pre.detach().numpy().flatten())[0]
#         # pdbrvalue.append(pearson)
#         r2 = metrics.r2_score(test_Yi.detach().numpy(), test_pre.detach().numpy())
#         mae = metrics.mean_absolute_error(test_Yi.detach().numpy(), test_pre.detach().numpy())
#         mse = metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy())
#         rmse = np.sqrt(metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy()))
#         rae = metrics.mean_absolute_error(test_Yi.detach().numpy(), test_pre.detach().numpy()) / metrics.mean_absolute_error(
#             test_Yi.detach().numpy(), [test_Yi.detach().numpy().mean()] * len(test_Yi.detach().numpy()))
#         rrse = np.sqrt(metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy())) / np.sqrt(
#             metrics.mean_squared_error(test_Yi.detach().numpy(),
#                                        [test_Yi.detach().numpy().mean()] * len(test_Yi.detach().numpy())))
#         print("==================================pdb ressult==================================================")
#         print('pdb:',i,  'r2', r2, 'r', pearson, 'mse:', mse, 'mae:', mae, 'rmse:', rmse, 'rae:', rae,
#               'rrse:', rrse)
#     return (r)
# rlist = getr(test_df, train_df)
#         #绘图预测和真实值的对比散点图
#         # plt.figure(figsize=(6, 6))
#         # plt.scatter(test_Yi.detach().numpy().flatten(), test_pre.detach().numpy().flatten())
#         # plt.title(str(i))
#         # plt.xlabel('Experimental Binding Affinity (kcal/mol$^-1$)')
#         # plt.ylabel('Predicted Binding Affinity (kcal/mol$^-1$)')
#         # plt.savefig('.\JJH\\figure\\predicted_binding_r_train_'+str(i)+'.tiff',dpi=300)
#         # plt.show()
#         # plt.close()
#
#         # 绘制亲和力分布分布图
#         # bincount = int((max(dockingdf.loc[:, 'ba'].tolist()) - min(dockingdf.loc[:, 'ba'].tolist())) * 2)
#         plt.hist(train_Yi.detach().numpy().flatten(), edgecolor='black', color='grey')
#         plt.xlabel('Binding Affinity (kcal/mol$^-1$)')
#         plt.ylabel('Frequency')
#         plt.title("Histogram of binding affinity values")
#         # plt.savefig('.\JJH\\figure\\bindingfrequencyall.tiff', dpi=300)
#         plt.show()
#         plt.close()
#
# #结果不好的pdb的实验
# i='4e5w'
# train_i=train_df.loc[train_df['pdb']==i]
# train_Yi = torch.from_numpy(train_i[['ba']].values)
# train_Xi=train_i.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
# train_Xi = torch.from_numpy(train_Xi.values)
# # 对张量进行最小-最大归一化，需要注意分母不能为0
# # 沿着列的维度找到最小值和最大值
# min_values = torch.min(train_Xi, dim=0).values
# max_values = torch.max(train_Xi, dim=0).values
# train_Xi_normal = (train_Xi - min_values) / (max_values - min_values)
# # train_Xi_normal = (train_Xi - trainmin_values) / (trainmax_values - trainmin_values)
# train_Xi_normal = torch.where(torch.isnan(train_Xi_normal), torch.full_like(train_Xi_normal, 0), train_Xi_normal)
# train_Xi_normal, train_Yi = train_Xi_normal.to(torch.float32), train_Yi.to(torch.float32)
#
# test_i=test_df.loc[test_df['pdb']==i]
# test_Yi = torch.from_numpy(test_i[['ba']].values)
# test_Xi=test_i.drop(['pdb', 'variant fold name','ligand_file', 'chembl_id', 'tanimoto_index', 'ba','RuleOfFiveDescriptor'], axis=1)
# test_Xi = torch.from_numpy(test_Xi.values)
# testmin_values = torch.min(test_Xi, dim=0).values
# testmax_values = torch.max(test_Xi, dim=0).values
# # 对张量进行最小-最大归一化，需要注意分母不能为0
# test_Xi_normal = (test_Xi - testmin_values) / (testmax_values - testmin_values)
# test_Xi_normal = torch.where(torch.isnan(test_Xi_normal), torch.full_like(test_Xi_normal, 0), test_Xi_normal)
# test_Xi_normal, test_Yi = test_Xi_normal.to(torch.float32), test_Yi.to(torch.float32)
#
# torch_dataset_train = torch.utils.data.TensorDataset(train_Xi_normal, train_Yi)
# torch_dataset_test = torch.utils.data.TensorDataset(test_Xi_normal, test_Yi)#'epoch:', '99', 'r2:', '0.025613112958933315', 'r:', '0.49191204120872634', 'mse:', '0.3619041'
#
# torch_dataset_train2 = torch.utils.data.TensorDataset(train_Xi, train_Yi)#
# torch_dataset_test2 = torch.utils.data.TensorDataset(test_Xi, test_Yi)
# model = DTA(2307)
# model2, r = train(torch_dataset_train, torch_dataset_test, str(i), model)#batchsize=100,epoch: 99 r2 0.020069249015356005 r 0.5048059753302647 mse: 0.3639632 mae: 0.48572978
# model3, r = train(torch_dataset_train, torch_dataset_test, str(i), model2)#'epoch:', '298', 'r2:', '0.4114451128976844', 'r:', '0.6452083132641111', 'mse:', '0.21859944', 'mae:', '0.35592622'
# model4, r = train(torch_dataset_train, torch_dataset_test, str(i), model3)#'epoch:', '575', 'r2:', '0.5707600363279413', 'r:', '0.7556209707691005', 'mse:', '0.15942714', 'mae:', '0.29861078'
# model5, r = train(torch_dataset_train, torch_dataset_test, str(i), model4)#'epoch:', '798', 'r2:', '0.6019389935931829', 'r:', '0.7758860095544196', 'mse:', '0.14784673', 'mae:', '0.28547075'
# model6, r = train(torch_dataset_train, torch_dataset_test, str(i), model5)# 'epoch:', '990', 'r2:', '0.6116999119678792', 'r:', '0.7822556388362067', 'mse:', '0.14422134', 'mae:', '0.28109318'
# model7 = DTA(2307)
# model7.load_state_dict(torch.load('.\JJH\predictions\DDTA_att_all_EL2\\4e5w\\4e5wbest_r0.7822556388362067_mse0.14422134.mdl'))
# model8, r = train(torch_dataset_train, torch_dataset_test, str(i), model7)
#
#
# model3, r = train(torch_dataset_train2, torch_dataset_test2, str(i), model)
#
# alltest_Y=torch.tensor([[0]])
# alltest_pre=torch.tensor([[0]])
# for i in pdb:
#     # train_i = train_df.loc[train_df['pdb'] == i]
#     # train_Yi = torch.from_numpy(train_i[['ba']].values)
#     # train_Xi = train_i.drop(
#     #     ['pdb', 'variant fold name', 'ligand_file', 'chembl_id', 'tanimoto_index', 'ba', 'RuleOfFiveDescriptor'],
#     #     axis=1)
#     # train_Xi = torch.from_numpy(train_Xi.values)
#     # # 对张量进行最小-最大归一化，需要注意分母不能为0
#     # # trainmin_values = torch.min(train_Xi, dim=0).values
#     # # trainmax_values = torch.max(train_Xi, dim=0).values
#     # train_Xi_normal = (train_Xi - alltrainmin_values) / (alltrainmax_values - alltrainmin_values)
#     # train_Xi_normal = torch.where(torch.isnan(train_Xi_normal), torch.full_like(train_Xi_normal, 0), train_Xi_normal)
#     # train_Xi_normal, train_Yi = train_Xi_normal.to(torch.float32), train_Yi.to(torch.float32)
#
#     test_i = test_df.loc[test_df['pdb'] == i]
#     test_Yi = torch.from_numpy(test_i[['ba']].values)
#     test_Xi = test_i.drop(
#         ['pdb', 'variant fold name', 'ligand_file', 'chembl_id', 'tanimoto_index', 'ba', 'RuleOfFiveDescriptor'],
#         axis=1)
#     test_Xi = torch.from_numpy(test_Xi.values)
#     # 对张量进行最小-最大归一化，需要注意分母不能为0
#     # testmin_values = torch.min(test_Xi, dim=0).values
#     # testmax_values = torch.max(test_Xi, dim=0).values
#     test_Xi_normal = (test_Xi - alltestmin_values) / (alltestmax_values - alltestmin_values)
#     test_Xi_normal = torch.where(torch.isnan(test_Xi_normal), torch.full_like(test_Xi_normal, 0), test_Xi_normal)
#     test_Xi_normal, test_Yi = test_Xi_normal.to(torch.float32), test_Yi.to(torch.float32)
#
#     model = DTA(2307)
#     # if i == '4e5w':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL2\\4e5w\\4e5wbest_r_0.7841476529546667_mse0.1435155epoch999.mdl'))
#     # elif i == '3up2':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL2\\3up2\\3up2best_r_0.8542353974334358_mse0.15890177epoch987.mdl'))
#     # elif i == '3fv1':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL2\\3fv1\\3fv1best_r_0.8542476072373946_mse0.1559665epoch972.mdl'))
#     # elif i == '3b5r':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL2\\3b5r\\3b5rbest_r_0.8497951746846897_mse0.26174113epoch989.mdl'))
#     # elif i == '5a7b':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\5a7b\\5a7bbest_r_0.8376907440030024_mse0.106933594epoch972.mdl'))
#     if i == '4e5w':
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\4e5w\\4e5wbest_r0.7844631032818306_mse0.14314522.mdl'))
#     elif i == '3up2':
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\3up2\\3up2best_r_0.81145532010277_mse0.2019219epoch994.mdl'))
#     elif i == '3fv1':
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\3fv1\\3fv1best_r_0.8275989799865692_mse0.1823363epoch934.mdl'))
#     elif i == '3b5r':
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\3b5r\\3b5rbest_r_0.848661490281699_mse0.2634992epoch994.mdl'))
#     elif i == '5a7b':
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA_att_all_EL\\5a7b\\5a7bbest_r_0.8376907440030024_mse0.106933594epoch972.mdl'))
#     # if i == '4twp':
#     #     model.load_state_dict(torch.load('.\JJH\predictions\DDTA_att_all_EL2\\4twp\\4twp.mdl'))
#     else:
#         model.load_state_dict(torch.load('.\JJH\predictions\DTA1_att_all\\DTA1_att_allbest_r0.9531907406584055_mse0.20353971.mdl'))
#
#     test_pre = model(test_Xi_normal)
#     alltest_pre = torch.cat((alltest_pre, test_pre), dim=0)
#     alltest_Y = torch.cat((alltest_Y, test_Yi), dim=0)
#     # 评价指标
#     pearson = pearsonr(test_Yi.detach().numpy().flatten(), test_pre.detach().numpy().flatten())[0]
#     # pdbrvalue.append(pearson)
#     r2 = metrics.r2_score(test_Yi.detach().numpy(), test_pre.detach().numpy())
#     mae = metrics.mean_absolute_error(test_Yi.detach().numpy(), test_pre.detach().numpy())
#     mse = metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy())
#     rmse = np.sqrt(metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy()))
#     rae = metrics.mean_absolute_error(test_Yi.detach().numpy(),
#                                       test_pre.detach().numpy()) / metrics.mean_absolute_error(
#         test_Yi.detach().numpy(), [test_Yi.detach().numpy().mean()] * len(test_Yi.detach().numpy()))
#     rrse = np.sqrt(metrics.mean_squared_error(test_Yi.detach().numpy(), test_pre.detach().numpy())) / np.sqrt(
#         metrics.mean_squared_error(test_Yi.detach().numpy(),
#                                    [test_Yi.detach().numpy().mean()] * len(test_Yi.detach().numpy())))
#     print("==================================pdb ressult==================================================")
#     print('pdb:', i, 'r2', r2, 'r', pearson, 'mse:', mse, 'mae:', mae, 'rmse:', rmse, 'rae:', rae,
#           'rrse:', rrse)
# # 全部评价指标
# alltest_Y=alltest_Y[1:,:]
# alltest_pre =alltest_pre[1:,:]
# pearson = pearsonr(alltest_Y.detach().numpy().flatten(), alltest_pre.detach().numpy().flatten())[0]
# r2 = metrics.r2_score(alltest_Y.detach().numpy(), alltest_pre.detach().numpy())
# mae = metrics.mean_absolute_error(alltest_Y.detach().numpy(), alltest_pre.detach().numpy())
# mse = metrics.mean_squared_error(alltest_Y.detach().numpy(), alltest_pre.detach().numpy())
# rmse = np.sqrt(metrics.mean_squared_error(alltest_Y.detach().numpy(), alltest_pre.detach().numpy()))
# rae = metrics.mean_absolute_error(alltest_Y.detach().numpy(),
#                                   alltest_pre.detach().numpy()) / metrics.mean_absolute_error(
#     alltest_Y.detach().numpy(), [alltest_Y.detach().numpy().mean()] * len(alltest_Y.detach().numpy()))
# rrse = np.sqrt(metrics.mean_squared_error(alltest_Y.detach().numpy(), alltest_pre.detach().numpy())) / np.sqrt(
#     metrics.mean_squared_error(alltest_Y.detach().numpy(),
#                                [alltest_Y.detach().numpy().mean()] * len(alltest_Y.detach().numpy())))
# print("==================================all pdb ressult==================================================")
# print('r2', r2, 'r', pearson, 'mse:', mse, 'mae:', mae, 'rmse:', rmse, 'rae:', rae,'rrse:', rrse)