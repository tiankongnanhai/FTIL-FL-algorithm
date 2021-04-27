import numpy as np
import pandas as pd 
import operator
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import linear_model
from random import randint, sample
from FTIR_show import getdata
import copy





# #读入数据
# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #Merge into a data set
# FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
# FTIR_DATA= FTIR_DATA.T

# #Check the stitching result
# # print(FTIR_DATA._stat_axis.values.tolist())
# # print(FTIR_DATA.shape)

# #Create label, 0 means 2B, 1 means A549
# Label = [0 for i in range(DATA_2B.shape[1]-1)] + [1 for i in range(DATA_A549.shape[1]-1)]
# Label = np.array(Label)

# #Convert tags into dummy variables
# # dummy_Label = pd.get_dummies(Label,prefix='type')

# # print(len(Label))
# absorb = FTIR_DATA.iloc[1:]







#函数
def RSE(X,Y,X_sub_index,n,n_sample):
    '''
    输入：原始光谱；当前子集序列list；子集变量数
    返回：残差标准误差
    '''
    X_sub = X.iloc[:,X_sub_index]

    Y_pred_record = []
    #留一法计算预测值
    Oneleft = LeaveOneOut()
    regr = LinearRegression()
    for train,test in Oneleft.split(X_sub):
        regr.fit(X_sub.iloc[train,:],Y[train])
        Y_pred = regr.predict(X_sub.iloc[test,:])
        Y_pred_record.append(Y_pred) 
    Y_pred_record = np.array(Y_pred_record).reshape(1,n_sample)
    Y_error = Y - Y_pred_record
    return (np.sum(np.square(Y_error))/(n-2))**(1/2)





def RM(absorb,Label,n_min,n_max):
    #原始光谱、标签
    X = absorb
    Y = Label

    #样本数
    n_sample = X.shape[0]

    #变量数
    n_var = X.shape[1]

    #结果记录
    #变量数；路径；rse；子集
    result = []

    #随机选择（n_min，n_max）数量的变量
    for n in range(n_min,n_max+1):
        Original_subset = sample([i for i in range(n_var)],n)
        print('Original_subset:',Original_subset)

        rse_min = RSE(X,Y,Original_subset,n,n_sample)
        new_best_sub = Original_subset.copy()

        for old in range(len(Original_subset)):
            X_sub_index = Original_subset.copy()
            print('Original_X_sub_index:',X_sub_index)
            #检测器
            best_sub = [None]*len(new_best_sub)

            #如果没有产生新的best_sub，该路径结束
            while not operator.eq(best_sub,new_best_sub):
                #开始下一轮
                best_sub = new_best_sub.copy()
                #选择子集中一个变量进行替换，保留S值最好的模型
                for new in range(n_var):
                    if X_sub_index[old] == new:
                        continue
                    else:
                        X_sub_index[old] = new
                        rse = RSE(X,Y,X_sub_index,n,n_sample)
                        # print(rse)
                        if rse < rse_min:
                            rse_min = rse
                            best_sub = X_sub_index.copy()
                
                #计算当前子集回归系数的标准误差
                results = sm.OLS(Y, X.iloc[:,best_sub]).fit()
                st_error = results.bse

                #st_error进行排序
                sort_st = sorted(enumerate(st_error), key=lambda x: x[1],reverse=True)
                sort_st_index = list(map(lambda x: x[0], sort_st))

                #根据st_error从大到小进行变量替换
                X_sub_index = best_sub.copy()

                for st in range(len(sort_st_index)):
                    st_index =sort_st_index[st]
                    #如果是本次需要替换的old，跳过
                    if st_index == old:
                        continue
                    else:
                        for new_2 in range(n_var):
                            if new_2 not in X_sub_index:
                                X_sub_index[st_index] = new_2
                                rse = RSE(X,Y,X_sub_index,n,n_sample)
                                # print(rse)
                                if rse < rse_min:
                                    rse_min = rse
                                    new_best_sub = X_sub_index.copy()
            result.append((n,old,rse,best_sub))
            print('Result_subset:',new_best_sub)
            print('Result_rse:',rse_min)

    #按照rse将subset排序
    result = sorted(result, key=lambda x: x[2])
    return result[0]