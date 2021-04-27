import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
import math
import copy
import random
from sklearn.model_selection import LeaveOneOut
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from FTIR_Simulation_data import Generate_simulation_data
from FTIR_show import Show_select_ans
from FTIR_Pretreatment import getdata, get_baseline_single, get_baseline
from FTIR_Iforest import Iforest






def scars(K_num,N_num,Mc_times,wave_name,absorb,Label,csv_dir,save_dir):


    #样本数
    Sample_num = absorb.shape[0]

    #变量数
    Var_num = absorb.shape[1]

    # #波长名字
    name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
    name = list(name)

    #参数
    a = (Var_num/2)**(1/(K_num-1))
    k = (math.log(Var_num/2))/(K_num-1)

    #RMSECV结果
    RMSECV_record = []

    #子集结果
    subset = []

    #采样变量数量
    sample_var = []

    #每个变量的回归系数
    var_coef = []

    #开始cars算法
    for iter in range(1,K_num+1):

        #计算系数矩阵(Mc_times, Var_num)
        coef_matrix = np.zeros((Mc_times,Var_num))
        for mc in range(Mc_times):
            #随机选取样本
            Select_sample = random.sample([i for i in range(Sample_num)], N_num)
            train_data = absorb.iloc[Select_sample,:]
            train_label = Label[Select_sample]
            #pls
            pls_model = PLSRegression(copy=True, max_iter=100, n_components=2, scale=True, tol=1e-06)
            pls_model.fit(train_data, train_label)
            coef = pls_model.coef_
            coef = coef.flatten()
            coef_list = coef.tolist()
            coef_matrix[mc,:] = coef_list

        #计算C值
        C_value = np.zeros((Var_num,2))
        for col in range(Var_num):
            cur_mean = np.mean(coef_matrix[:,col])
            cur_std = np.std(coef_matrix[:,col])
            C_value[col][0] = name[col]
            C_value[col][1] = abs(cur_mean/cur_std)
        
        #计算变量保留率
        r = a*math.exp(-k*iter)

        #利用指数衰减函数强行去除|bi|值较小波长
        C_value_sort = C_value[C_value[:,1].argsort()][::-1,:]
        C_value_sort = C_value_sort[:int(r*len(C_value_sort)),:]

        #利用ARS采样技术提取新变量子集
        ars_weight = list(C_value_sort[:,1])
        ars_weight_sum = sum(ars_weight)
        probabilities = [None]*len(ars_weight)
        for i in range(len(ars_weight)):
            probabilities[i]=ars_weight[i]/(ars_weight_sum)
        
        #概率抽样
        ars_result = np.random.choice(a = C_value_sort[:,0],size = len(probabilities),p = probabilities,replace=True)
        ars_result = set(ars_result)
        ars_result = list(map(lambda x: "%.4f" % x , ars_result))

        #保留该次采样变量数
        sample_var.append(len(ars_result))

        #如果特征数小于2，则aras算法结束
        if len(ars_result) < 2:
            #如果该次跳过，则回归系数全为0，RMSECV维持跟上次结果一致
            var_coef.append([0 for _ in range(Var_num)])
            RMSECV_record.append(RMSECV)
            subset.append([])
            continue

        #建立pls筛选最优子集
        ars_result_index = list(map(lambda x: name.index(x) , ars_result))
        train_data_subset = train_data.iloc[:,ars_result_index]
        train_data_subset = np.array(train_data_subset,dtype=float)

        #计算该抽样下的变量回归系数
        pls_model.fit(train_data_subset,train_label)
        coef_2 = pls_model.coef_
        coef_2 = coef_2.flatten()
        coef_2_list = coef_2.tolist()
        cur_var_coef = [0 for _ in range(Var_num)]
        for index,var in zip(ars_result_index,coef_2_list):
            cur_var_coef[index] = var
        print(len(ars_result))
        var_coef.append(cur_var_coef)

        #建立子集pls模型，并计算RMSECV
        temp_sum = 0
        Oneleft = LeaveOneOut()
        for train,test in Oneleft.split(train_data_subset):
            pls_model.fit(train_data_subset[train,:],train_label[train])
            Y_pred = pls_model.predict(train_data_subset[test,:])
            #将输出结果变成0,1离散值，阈值为0.5
            if Y_pred > 0.5:
                Y_pred = 1
            else:
                Y_pred = 0
            temp_sum += (Y_pred-train_label[test])**2

        RMSECV = (temp_sum/len(train_label))**(1/2)

        #保留四位小数
        RMSECV = float('%.4f' % RMSECV)

        #保存该次结果
        RMSECV_record.append(RMSECV)
        subset.append(ars_result)
        

    #将subset结果写入csv文件
    with open(csv_dir, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in zip(RMSECV_record,subset):
            writer.writerow([key, value])
        

    #将var_coef整理成narry
    var_coef_2 = np.array(var_coef).astype('float64')

    #将RMSECV进行小波去噪
    RMSECV_record = get_baseline_single(RMSECV_record)
    RMSECV_record = list(RMSECV_record)

    #输出最小RMSECV及对应的subset
    RMSECV_min = float('inf')
    best_subset = list()
    for index,key in enumerate(RMSECV_record):
        if key < RMSECV_min:
            RMSECV_min = key
            best_subset = subset[index]
        #相同RMSECV，输出波长数较小的子集
        if key==RMSECV_min and len(subset[index])<len(best_subset):
            best_subset = subset[index]
    
    RMSECV_min_index = RMSECV_record.index(RMSECV_min)


    #画图
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplot(3,1,1)
    plt.plot([i for i in range(K_num)], sample_var)
    plt.xlabel('K_num')
    plt.ylabel('Sample_var')
    plt.subplot(3, 1, 2)
    for wave in range(Var_num):
        plt.plot([i for i in range(K_num)],var_coef_2[:,wave])
    plt.xlabel('K_num')
    plt.ylabel('Coef')
    plt.subplot(3, 1, 3)
    plt.plot([i for i in range(K_num)],RMSECV_record)
    plt.axvline(x=RMSECV_min_index, c="r", ls="--",lw=2)
    plt.xlabel('K_num')
    plt.ylabel('RMSECV')
    plt.tight_layout()
    plt.savefig(save_dir + 'Cars'+'.jpg')
    plt.show()


    best_subset_index = list(map(lambda x: name.index(x) , best_subset))

    return [RMSECV_min,best_subset,best_subset_index]





#读入数据
DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'
DATA_2B = getdata(DATA_2B_dir)
DATA_A549 = getdata(DATA_A549_dir)

#波长名
wave_name = DATA_2B['wave']

#图片保存路径
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据/测试结果//'

#csv保存地址
csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'

#去除离群样本
DATA_2B = DATA_2B.drop(['wave'],axis=1)
DATA_2B = DATA_2B.T
DATA_A549 = DATA_A549.drop(['wave'],axis=1)
DATA_A549 = DATA_A549.T
Outlier_2B,Remain_2B = Iforest(DATA_2B,save_dir,'2B')
Outlier_A549,Remain_A549 = Iforest(DATA_A549,save_dir,'A549')
DATA_2B = DATA_2B.iloc[Remain_2B,]
DATA_A549 = DATA_A549.iloc[Remain_A549,]

absorb = pd.concat([DATA_2B,DATA_A549])

#样本名
sample_name = absorb._stat_axis.values.tolist()

#标签
Label = [0 for i in range(DATA_2B.shape[0])] + [1 for i in range(DATA_A549.shape[0])]
Label = np.array(Label)



#小波消噪
wave_let_absorb = get_baseline(absorb)



K_num = 500
N_num = 200
Mc_times = 200
#csv保存地址
csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//模拟数据//'
#图片保存路径
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//模拟数据//'
# name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
RMSECV_min,best_subset,CARS_select = scars(K_num,N_num,Mc_times,wave_name,wave_let_absorb,Label,csv_dir+'scars.csv',save_dir)
print(wave_name[CARS_select])
#画图
Show_select_ans(wave_let_absorb,wave_name,CARS_select,'simulation_SCARS',save_dir)