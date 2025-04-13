'''
原论文中机器学习模型预测突变蛋白质-配体结合亲和力
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import scipy
import multiprocessing
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#数据处理
def prepareDataFrame(df, categorical_features):
    for ft in categorical_features:
        cat_ft = pd.get_dummies(df[ft], prefix='cat')
        df = pd.concat([df, cat_ft], axis=1)
        df = df.drop(ft, 1)
    print("Dataframe after dummy variables:")
    print(df.shape)
    Y = df[['ba']]
    Y = Y.values.ravel()
    X = df.drop('ba', axis=1)
    return X, Y

#将原始的PSnpBind-ML中数据的蛋白质特征改为大模型特征，并保存到相应的-PLM文件中。
def getfeature(trainortest,splitname,dropcolumns,WTprotein_ProtBertfeature):
    train = pd.read_csv("./data/PsnpBind/training/dual-model/"+trainortest+"-"+splitname+".csv", sep=",", encoding="utf-8")  # 6738*126
    train2 = train.drop(dropcolumns, axis=1)  # 6738*66
    trainmeta = pd.read_csv("./data/PsnpBind/training/dual-model/"+trainortest+"-meta-"+splitname+".csv", sep=",", encoding="utf-8")
    train2['pdb'] = trainmeta['pdb'].astype('str') + '_protein_Repair_WT'
    train3 = pd.merge(train2, WTprotein_ProtBertfeature, on='pdb')
    train3.drop(columns='pdb', axis=1, inplace=True)
    train3.to_csv("./data/PsnpBind/training/dual-model/"+trainortest+"-"+splitname+"-PLM.csv")
    return train3

def getproteinBertfeature(model_type):
    # 蛋白质PsnpBind特征，26个野生型蛋白质的62维特征
    WTprotein_emb60df = pd.read_csv('./data/PsnpBind/data/protein_descriptors.tsv', sep='\t')
    dropcolumns=list(set(WTprotein_emb60df.columns.tolist())-set(['uniprot', 'sequence']))
    # 蛋白质ProtBert特征，731个野生和突变蛋白质的1024维度特征
    protein_ProtBertfeature = pd.read_csv('data/PsnpBind/process/731protein_ProtBert1024.csv', index_col=0)
    WTprotein_ProtBertfeature=protein_ProtBertfeature.loc[protein_ProtBertfeature.index.str.contains('WT')]
    WTprotein_ProtBertfeature['pdb']=WTprotein_ProtBertfeature.index

    #将原来蛋白质特征替换为大模型的蛋白质特征
    if (model_type == "dual_model_pl_split_random"):
        train=getfeature('train', 'random', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'random', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_protein"):
        train=getfeature('train', 'protein', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'protein', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_pocket"):
        train=getfeature('train', 'pocket', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'pocket', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_ligand_weight"):
        train=getfeature('train', 'ligand-weight', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'ligand-weight', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_ligand_diversity"):
        train=getfeature('train', 'ligand-diversity', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'ligand-diversity', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_ligand_tpsa"):
        train=getfeature('train', 'ligand-tpsa', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'ligand-tpsa', dropcolumns, WTprotein_ProtBertfeature)

    elif (model_type == "dual_model_pl_split_ligand_volume"):
        train=getfeature('train', 'ligand-volume', dropcolumns, WTprotein_ProtBertfeature)
        test=getfeature('test', 'ligand-volume', dropcolumns, WTprotein_ProtBertfeature)

    return train,test

def model_selection_and_tuning(model_type, model_name, proteinfeature):
    categorical_features = []

    if (model_type == "dual_model_pl_split_random"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-random"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_protein"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-protein"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_pocket"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-pocket"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_weight"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-weight"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_diversity"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-diversity"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_tpsa"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-tpsa"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_volume"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-volume"+proteinfeature+".csv", index_col=0)

    print(train.shape)

    train_X, train_Y = prepareDataFrame(train, categorical_features)
    train_X = np.array(train_X)

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=456)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=789)

    if (model_name == 'RandomForest'):

        param_grid = {'max_features': [ round(train_X.shape[1] / 2)],
                      'n_estimators': [200, 300, 400, 500],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'min_samples_split': [2, 5, 10]
                      }
        regressor = RandomForestRegressor(n_jobs=-1, criterion='squared_error')

    elif (model_name == 'DecisionTree'):

        param_grid = {'max_features': ['auto', 'sqrt', 'log2', round(train_X.shape[1] / 2)],
                      'max_depth': [None, 2, 5, 10],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'min_samples_split': [2, 5, 10]
                      }
        regressor = DecisionTreeRegressor(criterion='squared_error')

    elif (model_name == 'LassoRegression'):

        param_grid = {'alpha': [0.01, 0.02, 0.4, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0]}
        regressor = Lasso(max_iter=10000)

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)

    elif (model_name == 'RidgeRegression'):

        param_grid = {'alpha': [0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0]}
        regressor = Ridge()

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)

    tuned_model = GridSearchCV(estimator=regressor, param_grid=param_grid, n_jobs=-1, verbose=3, cv=inner_cv,
                               scoring='r2')

    tuned_model.fit(train_X, train_Y)

    tuned_outer_cv = cross_val_score(tuned_model, train_X, train_Y, cv=outer_cv)
    tuned_outer_cv = np.round(tuned_outer_cv, 2)

    return tuned_outer_cv.mean(), tuned_model.best_params_

def comparePS(model_name,proteinfeature):
    categorical_features = []
    train = pd.read_csv("./data/PsnpBind/training/dual-model/train-protein" + proteinfeature + ".csv", index_col=0)
    test = pd.read_csv("./data/PsnpBind/training/dual-model/test-protein" + proteinfeature + ".csv", index_col=0)

    train_X, train_Y = prepareDataFrame(train, categorical_features)
    test_X, test_Y = prepareDataFrame(test, categorical_features)

    train_X = np.array(train_X)
    test_X = np.array(test_X)

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=456)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=789)
    if (model_name == 'RandomForest'):
        param_grid = {'max_features': [ round(train_X.shape[1] / 2)],
                      'n_estimators': [200, 300, 400, 500],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'min_samples_split': [2, 5, 10]
                      }
        regressor = RandomForestRegressor(n_jobs=-1, criterion='squared_error')

    elif (model_name == 'DecisionTree'):

        param_grid = {'max_features': ['auto', 'sqrt', 'log2', round(train_X.shape[1] / 2)],
                      'max_depth': [None, 2, 5, 10],
                      'min_samples_leaf': [1, 2, 5, 10],
                      'min_samples_split': [2, 5, 10]
                      }
        regressor = DecisionTreeRegressor(criterion='squared_error')

    elif (model_name == 'LassoRegression'):

        param_grid = {'alpha': [0.01, 0.02, 0.4, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0]}
        regressor = Lasso(max_iter=10000)

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)

        scaler = StandardScaler().fit(test_X)
        test_X = scaler.transform(test_X)

    elif (model_name == 'RidgeRegression'):

        param_grid = {'alpha': [0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0]}
        regressor = Ridge()

        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        scaler = StandardScaler().fit(test_X)
        test_X = scaler.transform(test_X)

    tuned_model = GridSearchCV(estimator=regressor, param_grid=param_grid, n_jobs=-1, verbose=3, cv=inner_cv,
                               scoring='r2')

    tuned_model.fit(train_X, train_Y)

    tuned_outer_cv = cross_val_score(tuned_model, train_X, train_Y, cv=outer_cv)
    tuned_outer_cv = np.round(tuned_outer_cv, 2)
    print(tuned_outer_cv.mean(), tuned_model.best_params_)

    #5-fold train
    regressor = RandomForestRegressor(n_jobs=-1, max_features=544, n_estimators=500, random_state=123,
                                      min_samples_leaf=1, min_samples_split=2, criterion='squared_error')

    regressor = Ridge(criterion='squared_error',alpha=0.001)
    regressor = Lasso(max_iter=10000,alpha=0.04)
    regressor = DecisionTreeRegressor(criterion='squared_error',max_depth= None, max_features=62, min_samples_leaf=10,
                                      min_samples_split= 5)
    kf = KFold(n_splits=5, random_state=123, shuffle=True)

    RMSE_sum = 0
    RMSE_length = 5
    MSE_sum = 0
    MSE_length = 5
    MAE_sum = 0
    MAE_length = 5
    R2_sum = 0
    R2_length = 5

    for loop_number, (train_k, test_k) in enumerate(kf.split(train_X)):
        ## Get Training Matrix and Y Vector

        training_X_array = train_X[train_k]
        training_y_array = train_Y[train_k].reshape(-1, 1)

        training_y_array = training_y_array.ravel()

        ## Get Testing Matrix and Y vector

        X_test_array = train_X[test_k]
        y_actual_values = train_Y[test_k]

        ## Fit the Random Forest Regression Model

        lr_model = regressor.fit(training_X_array, training_y_array)

        ## Compute the predictions for the test data

        prediction = lr_model.predict(X_test_array)
        predicted_Y = np.array(prediction)

        ## Calculate the RMSE

        RMSE_cross_fold = np.sqrt(metrics.mean_squared_error(y_actual_values, predicted_Y))
        RMSE_sum = RMSE_cross_fold + RMSE_sum

        MSE_cross_fold = metrics.mean_squared_error(y_actual_values, predicted_Y)
        MSE_sum = MSE_cross_fold + MSE_sum

        MAE_cross_fold = metrics.mean_absolute_error(y_actual_values, predicted_Y)
        MAE_sum = MAE_cross_fold + MAE_sum

        R2_cross_fold = metrics.r2_score(y_actual_values, predicted_Y)
        R2_sum = R2_cross_fold + R2_sum

    ## Calculate the average and print

    RMSE_cross_fold_avg = RMSE_sum / RMSE_length
    MSE_cross_fold_avg = MSE_sum / MSE_length
    MAE_cross_fold_avg = MAE_sum / MAE_length
    R2_cross_fold_avg = R2_sum / R2_length

    print('The Mean R2 across all folds is: ', R2_cross_fold_avg)
    print('The Mean MAE across all folds is: ', MAE_cross_fold_avg)
    print('The Mean MSE across all folds is: ', MSE_cross_fold_avg)
    print('The Mean RMSE across all folds is: ', RMSE_cross_fold_avg)

    # Now, lets train the model on the whole dataset

    regressor.fit(train_X, train_Y)

    print("Evaluation with the test set: ")
    final_metrics = evaluate(regressor, test_X, test_Y, model_type)

    final_metrics = [R2_cross_fold_avg, MAE_cross_fold_avg, MSE_cross_fold_avg, RMSE_cross_fold_avg] + final_metrics

    return final_metrics, regressor

#评估
#非常费时间，最后评估时候使用即可
def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])
    if pair is not 0:
        return summ / pair
    else:
        return 0

def evaluate(model, test_features, test_labels, model_type):
    predictions = model.predict(test_features)

    pred_dict = {'observed': test_labels, 'predicted': predictions}

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv("./predictions/" + model_type + ".csv", encoding="utf-8", index=False)

    pearson_r = scipy.stats.pearsonr(test_labels, predictions)
    print('Pearson R: ', pearson_r)
    print('Pearson R squared: ', pearson_r[0] ** 2)

    r2 = metrics.r2_score(test_labels, predictions)
    mae = metrics.mean_absolute_error(test_labels, predictions)
    mse = metrics.mean_squared_error(test_labels, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(test_labels, predictions))
    rae = metrics.mean_absolute_error(test_labels, predictions) / metrics.mean_absolute_error(test_labels, [
        test_labels.mean()] * len(test_labels))
    rrse = np.sqrt(metrics.mean_squared_error(test_labels, predictions)) / np.sqrt(
        metrics.mean_squared_error(test_labels, [test_labels.mean()] * len(test_labels)))

    print('R Squared Error:', r2)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('Relative absolute Error:', rae)
    print('Root relateive squared Error:', rrse)
    return [pearson_r, r2, mae, mse, rmse, rae, rrse]


def train_test(model_type,proteinfeature):
    categorical_features = []

    if (model_type == "dual_model_pl_split_random"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-random"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-random"+proteinfeature+".csv", index_col=0)
        if proteinfeature=='_PLM':
            regressor = RandomForestRegressor(n_jobs=-1, max_features=544, n_estimators=200, random_state=123,
                                          min_samples_leaf=1, min_samples_split=2, criterion='squared_error')
    elif (model_type == "dual_model_pl_split_protein"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-protein"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-protein"+proteinfeature+".csv", index_col=0)
        if proteinfeature == '_PLM':
            regressor = RandomForestRegressor(n_jobs=-1, max_features=544, n_estimators=500, random_state=123,
                                              min_samples_leaf=1, min_samples_split=2, criterion='squared_error')

    elif (model_type == "dual_model_pl_split_pocket"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-pocket"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-pocket"+proteinfeature+".csv", index_col=0)
        if proteinfeature == '_PLM':
            regressor = RandomForestRegressor(n_jobs=-1, max_features=544, n_estimators=500, random_state=123,
                                              min_samples_leaf=1, min_samples_split=2, criterion='squared_error')

    elif (model_type == "dual_model_pl_split_ligand_weight"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-weight"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-ligand-weight"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_diversity"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-diversity"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-ligand-diversity"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_tpsa"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-tpsa"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-ligand-tpsa"+proteinfeature+".csv", index_col=0)

    elif (model_type == "dual_model_pl_split_ligand_volume"):
        train = pd.read_csv("./data/PsnpBind/training/dual-model/train-ligand-volume"+proteinfeature+".csv", index_col=0)
        test = pd.read_csv("./data/PsnpBind/training/dual-model/test-ligand-volume"+proteinfeature+".csv", index_col=0)

    print(train.shape)
    print(test.shape)

    train_X, train_Y = prepareDataFrame(train, categorical_features)
    test_X, test_Y = prepareDataFrame(test, categorical_features)

    train_X = np.array(train_X)
    test_X = np.array(test_X)

    num_of_features = round(train_X.shape[1] / 2)

    regressor = RandomForestRegressor(n_jobs=-1, max_features=num_of_features, n_estimators=500, random_state=123,
                                      min_samples_leaf=1,min_samples_split=2,criterion='squared_error')

    kf = KFold(n_splits=5, random_state=123, shuffle=True)

    RMSE_sum = 0
    RMSE_length = 5
    MSE_sum = 0
    MSE_length = 5
    MAE_sum = 0
    MAE_length = 5
    R2_sum = 0
    R2_length = 5

    for loop_number, (train_k, test_k) in enumerate(kf.split(train_X)):
        ## Get Training Matrix and Y Vector

        training_X_array = train_X[train_k]
        training_y_array = train_Y[train_k].reshape(-1, 1)

        training_y_array = training_y_array.ravel()

        ## Get Testing Matrix and Y vector

        X_test_array = train_X[test_k]
        y_actual_values = train_Y[test_k]

        ## Fit the Random Forest Regression Model

        lr_model = regressor.fit(training_X_array, training_y_array)

        ## Compute the predictions for the test data

        prediction = lr_model.predict(X_test_array)
        predicted_Y = np.array(prediction)

        ## Calculate the RMSE

        RMSE_cross_fold = np.sqrt(metrics.mean_squared_error(y_actual_values, predicted_Y))
        RMSE_sum = RMSE_cross_fold + RMSE_sum

        MSE_cross_fold = metrics.mean_squared_error(y_actual_values, predicted_Y)
        MSE_sum = MSE_cross_fold + MSE_sum

        MAE_cross_fold = metrics.mean_absolute_error(y_actual_values, predicted_Y)
        MAE_sum = MAE_cross_fold + MAE_sum

        R2_cross_fold = metrics.r2_score(y_actual_values, predicted_Y)
        R2_sum = R2_cross_fold + R2_sum

    ## Calculate the average and print

    RMSE_cross_fold_avg = RMSE_sum / RMSE_length
    MSE_cross_fold_avg = MSE_sum / MSE_length
    MAE_cross_fold_avg = MAE_sum / MAE_length
    R2_cross_fold_avg = R2_sum / R2_length

    print('The Mean R2 across all folds is: ', R2_cross_fold_avg)
    print('The Mean MAE across all folds is: ', MAE_cross_fold_avg)
    print('The Mean MSE across all folds is: ', MSE_cross_fold_avg)
    print('The Mean RMSE across all folds is: ', RMSE_cross_fold_avg)

    # Now, lets train the model on the whole dataset

    regressor.fit(train_X, train_Y)

    print("Evaluation with the test set: ")
    final_metrics = evaluate(regressor, test_X, test_Y, model_type)

    final_metrics = [R2_cross_fold_avg, MAE_cross_fold_avg, MSE_cross_fold_avg, RMSE_cross_fold_avg] + final_metrics

    return final_metrics, regressor

def train_test_WT(model_type):
    trainrandomdf = pd.read_csv("./JJH/data/WT/train_fold0.csv", index_col=0)
    testrandomdf = pd.read_csv("./JJH/data/WT/test_fold0.csv", index_col=0)

    train_Y = trainrandomdf[['ba']]
    train_Y = train_Y.values.ravel()  # 转为ndarray的一维数组形式
    train_X = trainrandomdf.iloc[:, 6:]
    train_X = np.array(train_X)

    test_Y = testrandomdf[['ba']]
    test_Y = test_Y.values.ravel()  # 转为ndarray的一维数组形式
    test_X = testrandomdf.iloc[:, 6:]
    test_X = np.array(test_X)

    num_of_features = round(train_X.shape[1] / 2)

    regressor = RandomForestRegressor(n_jobs=-1, max_features=num_of_features, n_estimators=400, random_state=123,
                                      criterion='mse')#最好的实验结果

    kf = KFold(n_splits=5, random_state=123, shuffle=True)

    RMSE_sum = 0
    RMSE_length = 5
    MSE_sum = 0
    MSE_length = 5
    MAE_sum = 0
    MAE_length = 5
    R2_sum = 0
    R2_length = 5

    for loop_number, (train_k, test_k) in enumerate(kf.split(train_X)):
        ## Get Training Matrix and Y Vector

        training_X_array = train_X[train_k]
        training_y_array = train_Y[train_k].reshape(-1, 1)

        training_y_array = training_y_array.ravel()

        ## Get Testing Matrix and Y vector

        X_test_array = train_X[test_k]
        y_actual_values = train_Y[test_k]

        ## Fit the Random Forest Regression Model

        lr_model = regressor.fit(training_X_array, training_y_array)

        ## Compute the predictions for the test data

        prediction = lr_model.predict(X_test_array)
        predicted_Y = np.array(prediction)

        ## Calculate the RMSE

        RMSE_cross_fold = np.sqrt(metrics.mean_squared_error(y_actual_values, predicted_Y))
        RMSE_sum = RMSE_cross_fold + RMSE_sum

        MSE_cross_fold = metrics.mean_squared_error(y_actual_values, predicted_Y)
        MSE_sum = MSE_cross_fold + MSE_sum

        MAE_cross_fold = metrics.mean_absolute_error(y_actual_values, predicted_Y)
        MAE_sum = MAE_cross_fold + MAE_sum

        R2_cross_fold = metrics.r2_score(y_actual_values, predicted_Y)
        R2_sum = R2_cross_fold + R2_sum

    ## Calculate the average and print

    RMSE_cross_fold_avg = RMSE_sum / RMSE_length
    MSE_cross_fold_avg = MSE_sum / MSE_length
    MAE_cross_fold_avg = MAE_sum / MAE_length
    R2_cross_fold_avg = R2_sum / R2_length

    print('The Mean R2 across all folds is: ', R2_cross_fold_avg)
    print('The Mean MAE across all folds is: ', MAE_cross_fold_avg)
    print('The Mean MSE across all folds is: ', MSE_cross_fold_avg)
    print('The Mean RMSE across all folds is: ', RMSE_cross_fold_avg)

    #The Mean R2 across all folds is:  0.8549044329075197
    # The Mean MAE across all folds is:  0.38690981348465187
    # The Mean MSE across all folds is:  0.289365456534253
    # The Mean RMSE across all folds is:  0.5378689896207234
    #
    # Now, lets train the model on the whole dataset

    regressor.fit(train_X, train_Y)
    pickle.dump(regressor, open("./JJH/models/" + model_type + ".mdl", 'wb'))

    print("Evaluation with the test set: ")
    final_metrics = evaluate(regressor, test_X, test_Y, model_type)
    #Pearson R:  (0.9291608304100414, 0.0)
    # Pearson R squared:  0.8633398487682776
    # R Squared Error: 0.8626140500152696
    # Mean Absolute Error: 0.3871923041389502
    # Mean Squared Error: 0.2814033127194194
    # Root Mean Squared Error: 0.5304746108150883
    # Relative absolute Error: 0.3455254158209834
    # Root relateive squared Error: 0.3706561074429105

    final_metrics = [R2_cross_fold_avg, MAE_cross_fold_avg, MSE_cross_fold_avg, RMSE_cross_fold_avg] + final_metrics

    return final_metrics, regressor

if __name__ == "__main__":
    model_types = ['dual_model_pl_split_random',
                   'dual_model_pl_split_pocket',
                   'dual_model_pl_split_protein',
                   'dual_model_pl_split_ligand_diversity',
                   'dual_model_pl_split_ligand_weight',
                   'dual_model_pl_split_ligand_volume']
    proteinfeature=['-PLM','']
    tuning_metrics_df = pd.DataFrame()
    tuning_metrics_df['dataset'] = model_types
    for model_name in ['RandomForest', 'DecisionTree', 'LassoRegression', 'RidgeRegression']:
        tuned_outer_cv_arr = []
        model_type_best_params_df = pd.DataFrame()
        for model_type in model_types:
            tuned_outer_cv, tuned_model_best_params = model_selection_and_tuning(model_type, model_name,proteinfeature[0])
            tuned_outer_cv_arr.append(tuned_outer_cv)
            tmp_key_list = list()
            tmp_value_list = list()

            for key, value in tuned_model_best_params.items():
                tmp_key_list.append(key)
                tmp_value_list.append(value)

            model_type_best_params_df['param'] = pd.Series(tmp_key_list)
            model_type_best_params_df[model_type] = pd.Series(tmp_value_list)

            # temporary saving a snapshot of the file
            model_type_best_params_df.to_csv('./data/PsnpBind/training/dual-model/' + model_name + '_best_params_tuning.csv', encoding='utf-8',
                                             index=False)

        tuning_metrics_df[model_name] = pd.Series(tuned_outer_cv_arr)

        model_type_best_params_df.to_csv('./data/PsnpBind/training/dual-model/' + model_name + '_best_params_tuning.csv', encoding='utf-8',
                                         index=False)

        # temporary saving a snapshot of the file
        tuning_metrics_df.to_csv('./data/PsnpBind/training/dual-model/tuning_metrics_df.csv', encoding='utf-8', index=False)

    tuning_metrics_df.to_csv('./data/PsnpBind/training/dual-model/tuning_metrics_df.csv', encoding='utf-8', index=False)
