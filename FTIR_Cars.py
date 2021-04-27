import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
import math
import copy
from sklearn.model_selection import LeaveOneOut
from FTIR_Pretreatment import standardlize
from FTIR_show import getdata
import csv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from  scipy.stats import ttest_rel
from FTIR_Pretreatment import get_baseline_single,tsd_single



def cars(mc_times,wave_name,absorb,Label,csv_dir,save_dir):
    '''
    输入：mc_times(int); wave_name(narray); absorb(dataframe[N_sample,N_var])
        Label(narray); csv_dir(str); save_dir(str)

    输出：[RMSECV_min(int),best_subset(dataframe),best_subset_index(list)]
    '''

    #蒙特卡洛采样次数
    mc_num = mc_times

    #样本数
    sample_num = absorb.shape[0]

    #波长数
    wave_num = absorb.shape[1]

    # #波长名字
    name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
    name = list(name)

    #参数
    a = (wave_num/2)**(1/(mc_num-1))
    k = (math.log(wave_num/2))/(mc_num-1)

    #RMSECV结果
    RMSECV_record = []

    #子集结果
    subset = []

    #采样变量数量
    sample_var = []

    #每个变量的回归系数
    var_coef = []

    #开始cars算法
    for iter in range(1,mc_num+1):
        #随机9:1将样本划分为训练集和测试集
        train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(absorb,Label,\
            train_size=0.9,test_size=0.1)

        #pls
        pls_model = PLSRegression(copy=True, max_iter=100, n_components=2, scale=True, tol=1e-06)
        pls_model.fit(train_data, train_label)
        coef = pls_model.coef_
        coef = coef.flatten()
        coef_list = coef.tolist()

        #计算波长权重
        weights = np.zeros((len(coef),2))
        abs_coef = list(map(lambda x:abs(x),coef_list))
        sum_abs_coef = sum(abs_coef)
        for i in range(len(name)):
            weights[i][0] = name[i]
            weights[i][1] = abs_coef[i]/sum_abs_coef
        
        #计算变量保留率
        r = a*math.exp(-k*iter)

        #利用指数衰减函数强行去除|bi|值较小波长
        weights_sort = weights[weights[:,1].argsort()][::-1,:]
        weights_sort = weights_sort[:int(r*len(weights_sort)),:]

        #利用ARS采样技术提取新变量子集
        ars_weight = list(weights_sort[:,1])
        ars_weight_sum = sum(ars_weight)
        probabilities = [None]*len(ars_weight)
        for i in range(len(ars_weight)):
            probabilities[i]=ars_weight[i]/(ars_weight_sum)
        
        #概率抽样
        ars_result = np.random.choice(a = weights_sort[:,0],size = len(probabilities),p = probabilities,replace=True)
        ars_result = set(ars_result)
        ars_result = list(map(lambda x: "%.4f" % x , ars_result))

        #保留该次采样变量数
        sample_var.append(len(ars_result))

        #如果特征数小于2，则aras算法结束
        if len(ars_result) < 2:
            #如果该次跳过，则回归系数全为0，RMSECV维持跟上次结果一致
            var_coef.append([0 for _ in range(wave_num)])
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
        cur_var_coef = [0 for _ in range(wave_num)]
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
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['axes.unicode_minus'] = False  
    plt.subplot(3,1,1)
    plt.plot([i for i in range(mc_num)], sample_var)
    plt.xlabel('MC_num', fontsize=12,weight="bold")
    plt.ylabel('Sample_var', fontsize=12,weight="bold")
    plt.xticks(fontsize=11,weight="bold")
    plt.yticks(fontsize=11,weight="bold")
    plt.subplot(3, 1, 2)
    for wave in range(wave_num):
        plt.plot([i for i in range(mc_num)],var_coef_2[:,wave])
    plt.xlabel('MC_num', fontsize=12,weight="bold")
    plt.ylabel('Coef', fontsize=12,weight="bold")
    plt.subplot(3, 1, 3)
    plt.plot([i for i in range(mc_num)],RMSECV_record)
    plt.xticks(fontsize=11,weight="bold")
    plt.yticks(fontsize=11,weight="bold")
    plt.axvline(x=RMSECV_min_index, c="r", ls="--",lw=2)
    plt.xlabel('MC_num', fontsize=12,weight="bold")
    plt.ylabel('RMSECV', fontsize=12,weight="bold")
    plt.xticks(fontsize=11,weight="bold")
    plt.yticks(fontsize=11,weight="bold")
    plt.tight_layout()
    plt.savefig(save_dir + 'Cars'+'.jpg')
    plt.show()


    best_subset_index = list(map(lambda x: name.index(x) , best_subset))

    # plt.figure()
    # plt.plot(wave_name,absorb.iloc[0,:])
    # plt.scatter(wave_name[best_subset_index], absorb.iloc[0, best_subset_index], marker='s', color='r')
    # plt.legend(['First calibration object', 'Selected variables'])
    # plt.xlabel('Variable index')
    # plt.grid(True)
    # plt.savefig(save_dir + 'Cars_2'+'.jpg')
    # plt.show()
    
    return [RMSECV_min,best_subset,best_subset_index]



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

# #Standardize absorb
# sd_absorb = standardlize(absorb)



# mc_times = 500
# csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//cars.csv'
# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'
# wave_name = list(FTIR_DATA.iloc[0,:])
# RMSECV_min,best_subset,best_subset_index = cars(mc_times,wave_name,sd_absorb,csv_dir,save_dir) 

# print(best_subset)