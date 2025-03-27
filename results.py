'''
结果绘图可视化
'''
import numpy as np
import pandas as pd
import scipy
import pickle
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from Bio import pairwise2


#分别绘制26个pdb的结合亲和力分布图
def drawbindingfrequency(dockingdf):
    for i in pdb:
        #设置bin的个数
        bincount = int((max(dockingdf.loc[dockingdf['pdb'] == i, 'ba'].tolist()) - min(
            dockingdf.loc[dockingdf['pdb'] == i, 'ba'].tolist())) * 2)
        plt.hist(dockingdf.loc[dockingdf['pdb'] == i, 'ba'].tolist(), bincount, edgecolor='black', color='grey')
        plt.xlabel('Binding Affinity (kcal/mol$^-1$)')
        plt.xlim(-16, -4)
        plt.ylabel('Frequency')
        plt.ylim(0, 20000)
        plt.title("Histogram of " + i + " binding affinity values")
        plt.savefig('./figure/bindingfrequency/bindingfrequency' + i + '.tiff', dpi=300)
        plt.show()
        plt.close()

#一起绘制26个pdb的结合亲和力的小提琴图
def drawbindingfrequencyall(dockingstatisticsdf):
    rankmean = np.argsort(dockingstatisticsdf.mean_ba.tolist())  # 按照由低到高的排序方式找index
    rankvar = np.argsort(dockingstatisticsdf.var_ba.tolist())
    pdbmeansort = np.array(pdb)[rankmean]  # 重新按照均值排序pdb
    pdbvarsort = np.array(pdb)[rankvar]

    sns.set(style="ticks")
    # 绘制小提琴图
    plt.figure(figsize=(24, 6))
    sns.violinplot(x="pdb", y="ba", order=pdbmeansort, data=dockingdf)
    # 添加坐标轴标题
    plt.ylabel('Binding Affinity (kcal/mol$^-1$)')
    plt.xlabel('PDB')
    plt.savefig('./figure/bindingfrequency/26pdbbindingviolinplot_meansort.tiff', dpi=300)
    # 显示图表
    plt.show()
    plt.close()

    plt.figure(figsize=(24, 6))
    sns.violinplot(x="pdb", y="ba", order=pdbvarsort, data=dockingdf)
    plt.ylabel('Binding Affinity (kcal/mol$^-1$)')
    plt.xlabel('PDB')
    plt.savefig('./figure/bindingfrequency/26pdbbindingviolinplot_varsort.tiff', dpi=300)
    plt.show()
    plt.close()

#绘制热图
def drawheatmap(data,name):
    # 绘制热图
    plt.figure(figsize=(80, 80))
    ax = sns.heatmap(data, cmap="YlGn",
                     square=True,  # 正方形格仔
                     cbar=True,  # 是否去除 color bar
                     xticklabels=False, yticklabels=False)  # 去除纵、横轴 label
    # 调整color bar的字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    fig = ax.get_figure()
    plt.savefig('./figure/protein/'+name+'.tiff', dpi=300)
    plt.show()
    plt.close()

#计算并绘制蛋白质的序列大模型特征相似性热图
def getproteinProtBertsim():
    # 蛋白质ProtBert特征，731个野生和突变蛋白质的1024维度特征
    protein_ProtBertfeature = pd.read_csv('./data/PsnpBind/process/731protein_ProtBert1024.csv', index_col=0)

    # 计算大模型学习的蛋白质特征相似性
    proteinsimarray = cosine_similarity(protein_ProtBertfeature)
    proteinsimdf = pd.DataFrame(proteinsimarray, columns=protein_ProtBertfeature.index.tolist(),
                                index=protein_ProtBertfeature.index.tolist())
    proteinsimdf.to_csv('./data/PsnpBind/process/protein_ProtBertfeature_similaritydf.csv')

    # 绘制热图
    drawheatmap(proteinsimarray, 'protein_ProtBertfeature_cosinesim_heatmap')


#计算并绘制蛋白质的序列相似性热图
def getproteinsequencesim():
    protein_sequence = pd.read_csv('./data/PsnpBind/process/731protein_sequence.csv', index_col=0)
    proteinseqsimarray = np.zeros((731, 731))
    for i in protein_sequence.index:
        for j in range(i, 731):
            alignments = pairwise2.align.globalxx(protein_sequence.loc[i, 'sequence'], protein_sequence.loc[j, 'sequence'])
            score = alignments[0].score
            start = alignments[0].start
            end = alignments[0].end
            proteinseqsimarray[i, j] = score / (end - start)
            proteinseqsimarray[j, i] = score / (end - start)
    proteinseqsimdf = pd.DataFrame(proteinseqsimarray, columns=protein_sequence['fold name'].tolist(),
                                   index=protein_sequence['fold name'].tolist())
    proteinseqsimdf.to_csv('./data/PsnpBind/process/protein_sequence_similaritydf.csv')

    drawheatmap(proteinseqsimarray, 'protein_sequence_cosinesim_heatmap')




if __name__ == "__main__":
    pdb=['1owh','2c3i','2hb1','2pog','2weg','2y5h','3b27','3b5r','3fv1','3jvr','3pxf','3u9q','3udh','3up2','3utu',
         '4crc','4dli','4e5w','4gr0','4j21','4jia','4m0y','4twp','4wiv','5a7b','5c28']
    print("------绘制结合亲和力分布图")
    dockingdf=pd.read_csv('./data/PsnpBind/process/dockingdf631653.csv',index_col=0)
    drawbindingfrequency(dockingdf)

    # 整理26个蛋白质的结合亲和力的分布情况，均值，方差,并绘制小提琴图
    dockingstatisticsdf=pd.read_csv("./data/PsnpBind/process/dockingdata_26pdb_statistics.csv")
    drawbindingfrequencyall(dockingstatisticsdf)

    chemicalfeaturesdf = pd.read_csv("./data/PsnpBind/process/chemicalfeaturesdf36373_1167.csv", index_col=0)
