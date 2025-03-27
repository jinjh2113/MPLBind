from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import multiprocessing
import pickle
import scipy
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

'''
PSnpBind所有数据集整理数据
'''
#整理638984条结合亲和力数据,选取分子对接的结合亲和力最大值作为ba值，保留ba在-16到-4之间的值，共631653条，并保留
dockingdf=pd.read_csv('./data/PsnpBind/data/docking-results-all.tsv',sep='\t',header=None)#638984
dockingdf.columns=['pdb','protein','chemical','dockingvalue']
#拆分亲和力
dockingdf['dockingvalue'] = dockingdf['dockingvalue'].replace(['Conformer 1: ','Conformer 2: ','Conformer 3: '], '', regex=True)
dockingdf[['ba1','ba2','ba3']]=dockingdf.dockingvalue.str.split(pat=';',expand=True)
convert_dict = {'ba1':float,'ba2':float,'ba3':float}
dockingdf=dockingdf.astype(convert_dict)
#选取亲和力最大的值
dockingdf['ba']=dockingdf[['ba1','ba2','ba3']].apply(lambda  x:x.min(),axis=1)
#有一些ba大于0的为误差值，删除；保留ba在-16到-4之间的值
dockingdf=dockingdf.loc[(dockingdf['ba']<-4)&(dockingdf['ba']>-16)]#631653
dockingdf.to_csv('./data/PsnpBind/process/dockingdf631653.csv')


#整理26个蛋白质的结合亲和力的分布情况，均值，方差
pdb=['1owh','2c3i','2hb1','2pog','2weg','2y5h','3b27','3b5r','3fv1','3jvr','3pxf','3u9q','3udh','3up2','3utu',
         '4crc','4dli','4e5w','4gr0','4j21','4jia','4m0y','4twp','4wiv','5a7b','5c28']
dockingstatistics=dockingdf.groupby('pdb')
meandata=[]
vardata=[]
for i in pdb:
    meandata.append(dockingstatistics.get_group(i).ba.mean())
    vardata.append(dockingstatistics.get_group(i).ba.var())
dockingstatisticsdf=pd.DataFrame(data={'pdb':pdb,'mean_ba':meandata,'var_ba':vardata})
dockingstatisticsdf.to_csv("./data/PsnpBind/process/dockingdata_26pdb_statistics.csv")



#蛋白质PsnpBind特征，26个野生型蛋白质的62维特征
WTprotein_emb60df=pd.read_csv('./data/PsnpBind/data/protein_descriptors.tsv',sep='\t')
#蛋白质ProtBert特征，731个野生和突变蛋白质的1024维度特征
protein_ProtBertfeature=pd.read_csv('data/PsnpBind/process/731protein_ProtBert1024.csv', index_col=0)

uniprottopdb={'P00749': '1owh', 'P11309': '2c3i', 'P18031': '2hb1', 'P03372': '2pog', 'P00918': '2weg', 'P00742': '2y5h', 'P07900': '3b27', 'P10275': '3b5r', 'P39086': '3fv1', 'O14757': '3jvr', 'P24941': '3pxf', 'P37231': '3u9q', 'P56817': '3udh', 'O14965': '3up2', 'P00734': '3utu', 'P03951': '4crc', 'Q16539': '4dli', 'P23458': '4e5w', 'P39900': '4gr0', 'Q9H2K2': '4j21', 'O60674': '4jia', 'Q08881': '4m0y', 'P00519': '4twp', 'O60885': '4wiv', 'P04637': '5a7b', 'Q9Y233': '5c28'}

#计算突变对亲和力的影响


#野生型蛋白质
WTproteinfoldname=['1owh_protein_Repair_WT', '2c3i_protein_Repair_WT', '2hb1_protein_Repair_WT', '2pog_protein_Repair_WT', '2weg_protein_Repair_WT', '2y5h_protein_Repair_WT', '3b27_protein_Repair_WT', '3b5r_protein_Repair_WT', '3fv1_protein_Repair_WT', '3jvr_protein_Repair_WT', '3pxf_protein_Repair_WT', '3u9q_protein_Repair_WT', '3udh_protein_Repair_WT', '3up2_protein_Repair_WT', '3utu_protein_Repair_WT', '4crc_protein_Repair_WT', '4dli_protein_Repair_WT', '4e5w_protein_Repair_WT', '4gr0_protein_Repair_WT', '4j21_protein_Repair_WT', '4jia_protein_Repair_WT', '4m0y_protein_Repair_WT', '4twp_protein_Repair_WT', '4wiv_protein_Repair_WT', '5a7b_protein_Repair_WT', '5c28_protein_Repair_WT']
pdbtoWTfoldnamedic={'1owh': '1owh_protein_Repair_WT', '2c3i': '2c3i_protein_Repair_WT', '2hb1': '2hb1_protein_Repair_WT', '2pog': '2pog_protein_Repair_WT', '2weg': '2weg_protein_Repair_WT', '2y5h': '2y5h_protein_Repair_WT', '3b27': '3b27_protein_Repair_WT', '3b5r': '3b5r_protein_Repair_WT', '3fv1': '3fv1_protein_Repair_WT', '3jvr': '3jvr_protein_Repair_WT', '3pxf': '3pxf_protein_Repair_WT', '3u9q': '3u9q_protein_Repair_WT', '3udh': '3udh_protein_Repair_WT', '3up2': '3up2_protein_Repair_WT', '3utu': '3utu_protein_Repair_WT', '4crc': '4crc_protein_Repair_WT', '4dli': '4dli_protein_Repair_WT', '4e5w': '4e5w_protein_Repair_WT', '4gr0': '4gr0_protein_Repair_WT', '4j21': '4j21_protein_Repair_WT', '4jia': '4jia_protein_Repair_WT', '4m0y': '4m0y_protein_Repair_WT', '4twp': '4twp_protein_Repair_WT', '4wiv': '4wiv_protein_Repair_WT', '5a7b': '5a7b_protein_Repair_WT', '5c28': '5c28_protein_Repair_WT'}

#配体特征
chemblfeatures = pd.read_csv("./data/PsnpBind/data/1owh/chembl_ligands_features_1owh.tsv", sep='\t')
chemicalfeaturesdf=pd.DataFrame(columns=chemblfeatures.columns.tolist())
for i in pdb:
    chemblfeatures = pd.read_csv('./data/PsnpBind/data/'+i+'/chembl_ligands_features_'+i+'.tsv', sep='\t')
    chemicalfeaturesdf=pd.concat([chemicalfeaturesdf,chemblfeatures],axis=0,join='inner',sort=False)
chemicalfeaturesdf.to_csv("./data/PsnpBind/process/chemical_featuresdf36373_1167.csv")#36373个ligand_file name,35813chembl_id
chemicalfeaturesdf.dropna(axis=0,how='any',inplace=True)#27608 rows x 1167 columns
chemicalfeaturesdf.to_csv("./data/PsnpBind/process/chemical_featuresdfdropna27608_1167.csv")

#突变特征
variant1owh=pd.read_csv("./data/PsnpBind/data/1owh/pdbbind_pocket_variants_features_1owh.tsv",sep='\t')#74*156
variantfeaturedf=pd.DataFrame(columns=variant1owh.columns.tolist())
for i in pdb:
    variantfeature=pd.read_csv('./data/PsnpBind/data/'+i+'/pdbbind_pocket_variants_features_'+i+'.tsv',sep='\t')
    variantfeaturedf=pd.concat([variantfeaturedf,variantfeature],axis=0,join='inner',sort=False)
variantfeaturedf.to_csv("./data/PsnpBind/process/variant_featuredf1410_156.csv")#705个mut和705WT
variantfeaturedf=variantfeaturedf.iloc[:,35:]
variantfeaturedf=variantfeaturedf.loc[~variantfeaturedf['variant fold name'].isin(WTproteinfoldname)]#705*122
variantfeaturedf.index=range(len(variantfeaturedf))#705*122
#WT的特征均设为0
arr = np.zeros((26,122))
variantWTfeature=pd.DataFrame(arr,columns=variantfeaturedf.columns.tolist())
variantWTfeature['variant fold name']=WTproteinfoldname
variantfeaturedf=pd.concat([variantfeaturedf,variantWTfeature],axis=0,join='inner',sort=False)#731*122
variantfeaturedf.to_csv('./data/PsnpBind/process/variant_featuredf731_122.csv')
variantfeaturedf=pd.read_csv('./data/PsnpBind/process/variant_featuredf731_122.csv',index_col=0)

#生成集合蛋白质特征，配体特征，突变特征的拼接数据
datadf=pd.merge(dockingdf,chemicalfeaturesdf,how='inner',on=['ligand_file','pdb'])#556460*1169
protein_ProtBertfeature['variant fold name']=protein_ProtBertfeature.index.tolist()
datadf2=pd.merge(datadf,protein_ProtBertfeature,how='inner',on=['variant fold name'])#556460*2193
datadf3=pd.merge(datadf2,variantfeaturedf,how='inner',on=['variant fold name'])#556460*2314
datadf3.to_csv('./data/PsnpBind/process/datadf_556460_2314.csv')




#拆分数据
cv_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv_split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=datadf3, y=datadf3['variant fold name'])):
    print(f'Fold {fold_i} generated!')
    train_df = datadf3.iloc[train_index]
    for fold_j, (train_index1, valid_index) in enumerate(cv_split2.split(X=train_df, y=train_df['variant fold name'])):
        train_df1 = train_df.iloc[train_index1]
        valid_df = train_df.iloc[valid_index]
        train_df1.to_csv(f'./data/PsnpBind/process/all/train_fold{fold_i}.csv')
        valid_df.to_csv(f'./data/PsnpBind/process/all/valid_fold{fold_i}.csv')#111292
    test_df = datadf3.iloc[test_index]
    test_df.to_csv(f'./data/PsnpBind/process/all/test_fold{fold_i}.csv')#111292

datadf=pd.read_csv('./data/PsnpBind/process/datadf_556460_2314.csv',index_col=0)#bug读取很慢
WT_df=datadf.loc[datadf['variant fold name'].isin(WTproteinfoldname)]#21648
Mut_df=datadf.loc[~datadf['variant fold name'].isin(WTproteinfoldname)]#423520

'''
PSnpBind训练数据集整理数据
'''

data_train_random_meta = pd.read_csv("work/training/dual-model/train-meta-random.csv", sep=",", encoding="utf-8")
data_test_random_meta = pd.read_csv("work/training/dual-model/test-meta-random.csv", sep=",", encoding="utf-8")
data_train_random = pd.read_csv("work/training/dual-model/train-random.csv", sep=",", encoding="utf-8")
data_test_random = pd.read_csv("work/training/dual-model/test-random.csv", sep=",", encoding="utf-8")
all_ligands = pd.read_csv("work/data/chembl_ligands_filtered_combined_tanimoto_smiles.tsv", sep="\t", encoding="utf-8")