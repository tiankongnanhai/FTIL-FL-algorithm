from FTIR_Pretreatment import mean_centralization,standardlize,sg,msc,snv,D1,D2,get_baseline,tsd,ti
import copy
from FTIR_show import getdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,r2_score
from collections import Counter
import warnings
warnings.filterwarnings("ignore")





#读入数据
#######Read in data
DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

DATA_2B = getdata(DATA_2B_dir)
DATA_A549 = getdata(DATA_A549_dir)

#Merge into a data set
FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
FTIR_DATA= FTIR_DATA.T
# print(FTIR_DATA.shape)

#Create label, 0 means 2B, 1 means A549
Label = [0 for i in range(DATA_2B.shape[1]-1)] + [1 for i in range(DATA_A549.shape[1]-1)]
Label = np.array(Label)
# print(len(Label))
absorb = FTIR_DATA.iloc[1:]





#去噪和去除背景操作
smoth_ops = [sg,get_baseline,tsd,ti,None]
background_ops = [msc,snv,standardlize,None]

#smoth_ops 、derivation_ops 和 background_ops选出一个
combination_record = dict()
Oneleft = LeaveOneOut()
cur_ops = []
count = 0
for smoth in smoth_ops:
    if smoth != None:
        cur_ops.append(smoth)
    for background in background_ops:
        if background != None:
            cur_ops.append(background)
        if len(cur_ops) != 0:
            cur_absorb = absorb.copy()
            # cur_ops = [ti,msc]
            print(cur_ops)
            for ops in cur_ops:
                cur_absorb = ops(cur_absorb)

            #若某一列全为空缺值，则跳过该组合
            # print(Counter(cur_absorb.isnull().all()))
            if True in list(cur_absorb.isnull().all()):
                continue
            #使用每一列的平均值进行补全
            for col in range(cur_absorb.shape[1]):
                cur_absorb.iloc[:,col].fillna(cur_absorb.iloc[:,col].mean(),inplace = True)

            cur_absorb = pd.DataFrame(cur_absorb,index=absorb._stat_axis.values.tolist())
            

        #留一法加偏最小二乘预测
        Y_pred_record = []
        Oneleft = LeaveOneOut()
        for train,test in Oneleft.split(cur_absorb):
            pls_model = PLSRegression(copy=True, max_iter=100, n_components=2, scale=True,tol=1e-06)
            pls_model.fit(cur_absorb.iloc[train,:],Label[train])
            Y_pred = pls_model.predict(cur_absorb.iloc[test,:])
            if Y_pred > 0.5:
                Y_pred = 1
            else:
                Y_pred = 0
            Y_pred_record.append(float(Y_pred))
        
        #计算acc,precision,f1,recall,auc
        acc = accuracy_score(Label, Y_pred_record)
        precision = precision_score(Label,Y_pred_record, average='binary')
        f1 = f1_score(Label, Y_pred_record, average='binary')
        recall = recall_score(Label, Y_pred_record, average='binary')
        auc = roc_auc_score(Label,Y_pred_record)
        combination_record[count] = [[acc,precision,f1,recall,auc],cur_ops.copy()]
        count+=1

        if background != None: 
            cur_ops.pop()
    if smoth != None:
        cur_ops.pop()

#整理指标
evaluation = []
for i in range(len(combination_record)):
    evaluation.append(combination_record[i][0])

pd_evaluation = pd.DataFrame(evaluation,columns=['acc','precision','f1','recall','auc'])


#保存结果到csv文件
with open('D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//Pretreatment_combination_select.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    writer.writerows(combination_record.items())

#画图
x_index = [i for i in range(len(evaluation))]

plt.figure()
plt.plot(x_index,pd_evaluation['acc'],color='green',marker = "o",linewidth=1,linestyle='-',label='acc')
plt.plot(x_index,pd_evaluation['precision'],color='blue',marker = "v",linewidth=1,linestyle='-',label='precision')
plt.plot(x_index,pd_evaluation['f1'],color='red',marker = "s",linewidth=1,linestyle='-',label='f1')
plt.plot(x_index,pd_evaluation['recall'],color='magenta',marker = "p",linewidth=1,linestyle='-',label='recall')
plt.plot(x_index,pd_evaluation['auc'],color='black',marker = "D",linewidth=1,linestyle='-',label='auc')
plt.title("Pretreatment combination select")
plt.legend()
plt.show()