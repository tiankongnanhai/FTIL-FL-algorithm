import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from FTIR_show import getdata
from FTIR_Pretreatment import mean_centralization,standardlize,sg,msc,snv,D1,D2,get_baseline,tsd,ti
from FTIR_K_S import ks
from FTIR_Dim_reduction import dim_pca,dim_tsne
from multiprocessing import cpu_count
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from FTIR_Cars import cars
from FTIR_UVE import UVE
from FITR_SPA import SPA
from FTIR_RM import RM
from FTIR_Iforest import Iforest
from FTIR_Dim_reduction import Lasso_select
import warnings
warnings.filterwarnings("ignore")
# print(cpu_count())#View the number of cpu cores = 6




# ######Read in data
# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #Merge into a data set
# FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
# FTIR_DATA= FTIR_DATA.T
# # print(FTIR_DATA.shape)

# #Create label, 0 means 2B, 1 means A549
# Label = [0 for i in range(DATA_2B.shape[1]-1)] + [1 for i in range(DATA_A549.shape[1]-1)]
# Label = np.array(Label)
# # print(len(Label))
# absorb = FTIR_DATA.iloc[1:]




#########SVM
def SVM_search(x,y):
    # Partition data set
    train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(x,y, random_state=1,\
        train_size=0.7,test_size=0.3)

    # min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_data = min_max_scaler.fit_transform(train_data)
    # test_data = min_max_scaler.transform(test_data)

    print('data already')

    #Train svm classifier
    param_grid = [      {'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
                        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000, 5000 ,10000, 50000 ,100000],
                        'kernel': ['rbf']
                        },
                        {
                        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                        'kernel': ['linear']
                        },
                        {
                        'degree':[1, 3, 5, 7, 9, 11],
                        'kernel': ['poly']
                        },
                        {
                        'gamma':[1, 2, 3, 4],
                        'coef0':[0.2, 0.4, 0.6, 0.8, 1],
                        'kernel': ['sigmoid']
                        }]

    clf = GridSearchCV(SVC(class_weight='balanced',decision_function_shape='over'),\
        param_grid=param_grid, scoring='roc_auc', n_jobs=6, cv=10)

    clf = clf.fit(train_data, train_label)
    # print(clf.best_params_, clf.best_score_)

    #Return the best parameters、 maximum AUC value、 training accuracy、 test accuracy
    return [clf.best_params_, clf.best_score_, clf.best_estimator_.score(train_data, train_label), clf.best_estimator_.score(test_data, test_label)]




##########Without Pretreatment
# select_ans = SVM_search(absorb,Label)
# print(select_ans)



###########Wavelet deal
# wave_let_absorb = get_baseline(absorb)



##########cars
# mc_times = 500
# csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//cars.csv'
# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'
# wave_name = list(FTIR_DATA.iloc[0,:])
# RMSECV_min,best_subset,car_select = cars(mc_times,wave_name,wave_let_absorb,Label,csv_dir,save_dir)



#########lasso
# lasso_select = Lasso_select(FTIR_DATA,absorb,Label)



# #########UVE
# # UVE_select = UVE(FTIR_DATA,wave_let_absorb,Label)


#########spa
# wave_name = np.array(FTIR_DATA.iloc[0,:])
# wave_name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
# Xcal, Xval, ycal, yval = sklearn.model_selection.train_test_split(wave_let_absorb, Label, test_size=0.1, random_state=0)
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) 
# Xcal = min_max_scaler.fit_transform(Xcal)
# Xval = min_max_scaler.transform(Xval)
# spa_select, var_sel_phase2 = SPA().spa(Xcal, ycal, save_dir,FTIR_DATA, m_min=1, m_max=50, Xval=Xval, yval=yval, autoscaling=1)



# ########公共特征波长
# union_wave = list(set(lasso_select).union(spa_select))
# print(union_wave)
# print(wave_name[union_wave])
# subset = wave_let_absorb.iloc[:,union_wave]




# RM_result = RM(subset,Label,2,len(union_wave)-1)
# print(RM_result)
# best_subset = wave_let_absorb.iloc[:,RM_result[3]]
# Outlier,Remain = Iforest(absorb,save_dir)
# Remain_Label = Label[Remain]
# select_ans = SVM(subset,Label)
# print(select_ans)


# #RF
# # 训练集、测试机划分
# train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(subset,Label, random_state=1,train_size=0.7,test_size=0.3)
# #分类型决策树
# clf = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0, n_jobs=6)
# #训练模型
# clf.fit(train_data,train_label)
# #评估模型准确率
# print('训练集准确率：%s' % clf.score(train_data,train_label))
# print('测试集准确率：%s' % clf.score(test_data,test_label))
# #各特征重要性
# print('各特征重要性：%s' % clf.feature_importances_)






##########Pretreatment
# pre_ways = [mean_centralization,standardlize,sg,msc,snv,D1,D2]
# Get all permutations and combinations
# posible_combination = []
# select_ans = list()
# best_pre_combination = list()
# Take AUC value as the evaluation standard of the model
# best_test_score = float('-inf')

# for i in range(1,len(pre_ways)+1):
#     posible_combination+=list(itertools.combinations(pre_ways,i))

# print(len(posible_combination))
# for combination in posible_combination:
#     cur_data = absorb
#     for way in combination:
#         cur_data = way(cur_data)
#     cur_ans = SVM(cur_data,Label)
#     if cur_ans[1]>best_test_score:
#         best_test_score = cur_ans[1]
#         select_ans = cur_ans
#         best_pre_combination = combination

# print(select_ans)
# print(best_pre_combination)