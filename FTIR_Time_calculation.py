import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from FTIR_Iforest import Iforest
from FTIR_show import getdata
from FTIR_Pretreatment import get_baseline
from sklearn.model_selection import LeaveOneOut
from FTIR_UVE import UVE
import datetime
import warnings
warnings.filterwarnings("ignore")




#正确率（accuracy = （TP+TN）/ (TP+FP+FN+TN) 被分对的样本数除以所有的样本数，通常来说，正确率越高，分类器越好
#灵敏度/召回率（sensitive = TP / (TP+FN)  表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力
#特异度（specificity) = TN / (FP+TN)    表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力
#精度（precision）= TP/（TP+FP） 精度是精确性的度量，表示被分为正例的示例中实际为正例的比例
def model_evaluation(model_parameters,subset,Label):
    #训练svm分类器,['LASSO',{'C': 7, 'gamma': 1000, 'kernel': 'rbf'}]
    Y_pred_record = []
    Oneleft = LeaveOneOut()
    classifier=svm.SVC(C=model_parameters['C'],kernel=model_parameters['kernel'],gamma=model_parameters['gamma'],decision_function_shape='ovr') # ovr:一对多策略
    #开始时间
    start = datetime.datetime.now()
    for train,test in Oneleft.split(subset):
        classifier.fit(subset.iloc[train,:],Label[train])
        Y_pred = classifier.predict(subset.iloc[test,:])
        Y_pred_record.append(Y_pred)
    #结束时间
    end = datetime.datetime.now()
    run_time = (end-start).microseconds
    #计算指标
    TP, TN, FP, FN = 0, 0, 0, 0
    # for i in range(len(Y_pred_record)):
    #     if int(Y_pred_record[i]) == int(Label[i]):
    #         count += 1
    for i in range(len(Y_pred_record)):
        if int(Label[i]) == 0:
            if Y_pred_record[i] == 0:
                TN +=1
            else:
                FP +=1
        else:
            if Y_pred_record[i] == 1:
                TP +=1
            else:
                FN +=1
    accuracy = float('%.04f'%( (TP+TN) / (TP+FP+FN+TN) ))
    sensitive = float('%.04f'%( TP / (TP+FN) ))
    specificity = float('%.04f'%( TN / (FP+TN) ))
    precision = float('%.04f'%( TP/(TP+FP) ))
    # time_record.append(['LASSO',(end-start).microseconds])
    return [run_time, accuracy,sensitive,specificity,precision]





# #读入数据
# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'
# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #波长名
# wave_name = DATA_2B['wave']

# #图片保存路径
# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据/测试结果//'

# #csv保存地址
# csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'

# #去除离群样本
# DATA_2B = DATA_2B.drop(['wave'],axis=1)
# DATA_2B = DATA_2B.T
# DATA_A549 = DATA_A549.drop(['wave'],axis=1)
# DATA_A549 = DATA_A549.T
# Outlier_2B,Remain_2B = Iforest(DATA_2B,save_dir,'2B')
# Outlier_A549,Remain_A549 = Iforest(DATA_A549,save_dir,'A549')
# DATA_2B = DATA_2B.iloc[Remain_2B,]
# DATA_A549 = DATA_A549.iloc[Remain_A549,]

# absorb = pd.concat([DATA_2B,DATA_A549])

# #样本名
# sample_name = absorb._stat_axis.values.tolist()

# #标签
# Label = [0 for i in range(DATA_2B.shape[0])] + [1 for i in range(DATA_A549.shape[0])]
# Label = np.array(Label)





# #小波消噪
# wave_let_absorb = get_baseline(absorb)



# model_performance = []

# #cars
# CARS_select = [6931,6917,6919,6921,6924,6922,6927,6923,6926,6928,6929,6937,6939,6941,6944,6932,6948,6949,6953,6947,6952,6946,6938,6925]
# subset = absorb.iloc[:,CARS_select]
# performance = model_evaluation({'C': 23, 'gamma': 1000, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['CARS',performance])


# #lasso
# LASSO_select = [428,429,430,741,742,743,744,745,746,747,748,1420,1421,1422,1423,1424,1582,1583,2596,6934,6935,6936,6937]
# subset = wave_let_absorb.iloc[:,LASSO_select]
# performance = model_evaluation({'C': 7, 'gamma': 1000, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['LASSO',performance])


# #UVE
# UVE_select = UVE(wave_let_absorb,Label,wave_name)
# subset = wave_let_absorb.iloc[:,UVE_select]
# performance = model_evaluation({'C': 29, 'gamma': 1, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['UVE',performance])


# #spa
# SPA_select = [933,228,338,2922,3650,1586,6937]
# subset = wave_let_absorb.iloc[:,SPA_select]
# performance = model_evaluation({'C': 3, 'gamma': 5000, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['SPA',performance])


# #union
# union_select = CARS_select+LASSO_select+SPA_select
# subset = wave_let_absorb.iloc[:,union_select]
# performance = model_evaluation({'C': 27, 'gamma': 100, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['UNION',performance])



# #baseline
# performance = model_evaluation({'C': 21, 'gamma': 1, 'kernel': 'rbf'},wave_let_absorb,Label)
# print(performance)
# model_performance.append(['Baseline',performance])


# print(model_performance)