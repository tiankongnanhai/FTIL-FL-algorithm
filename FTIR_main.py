import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import pandas as pd
import csv
import sklearn
import matplotlib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from FTIR_show import getdata,show_all,show_mean,Show_select_ans
from FTIR_Pretreatment import mean_centralization,standardlize,sg,msc,snv,D1,D2,get_baseline,tsd,ti,msc
from FTIR_K_S import ks
from FTIR_Dim_reduction import dim_pca,dim_tsne
from multiprocessing import cpu_count
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from FTIR_Cars import cars
from FTIR_UVE import UVE
from FITR_SPA import SPA
from FTIR_RM import RM
from FTIR_Time_calculation import model_evaluation
from FTIR_SVM_GridSearch import SVM_search
from FTIR_Iforest import Iforest
from FTIR_Dim_reduction import Lasso_select
import warnings
import datetime
warnings.filterwarnings("ignore")




#读入数据
DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'
DATA_2B = getdata(DATA_2B_dir)
DATA_A549 = getdata(DATA_A549_dir)

#波长名
wave_name = DATA_2B['wave']

#图片保存路径
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'

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

#效果
# show_all(wave_let_absorb,sample_name,wave_name,save_dir,'Wavelet',['green','red'])
# show_mean(wave_let_absorb,sample_name,wave_name,save_dir,'Wavelet mean')

# #多元散射校正
# msc_absorb = msc(wave_let_absorb)

# #效果
# show_all(msc_absorb,sample_name,wave_name,save_dir,'Msc',['green','red'])
# show_mean(msc_absorb,sample_name,wave_name,save_dir,'Msc mean')

# print('pretreament success')




# #结果记录
# result = []

# #模型性能
# model_performance = []





# #CARS
# mc_times = 200
# # name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
# RMSECV_min,best_subset,CARS_select = cars(mc_times,wave_name,wave_let_absorb,Label,csv_dir+'cars.csv',save_dir)
# print(wave_name[CARS_select])
# #画图
# Show_select_ans(wave_let_absorb,wave_name,CARS_select,'CARS',save_dir)





# LASSO
LASSO_select = Lasso_select(wave_name,wave_let_absorb,Label,save_dir)
print(wave_name[LASSO_select])

#画图
Show_select_ans(wave_let_absorb,wave_name,LASSO_select,'LASSO',save_dir)






# #UVE
# UVE_select = UVE(wave_let_absorb,Label,wave_name)
# # print('UVE选择波长数量:', len(UVE_select))

# #画图
# Show_select_ans(wave_let_absorb,wave_name,UVE_select,'UVE',save_dir)





# #SPA
# Xcal, Xval, ycal, yval = sklearn.model_selection.train_test_split(wave_let_absorb, Label, test_size=0.1, random_state=0)
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) 
# Xcal = min_max_scaler.fit_transform(Xcal)
# Xval = min_max_scaler.transform(Xval)
# SPA_select, var_sel_phase2 = SPA().spa(Xcal, ycal, save_dir, wave_name, m_min=1, m_max=50, Xval=Xval, yval=yval, autoscaling=1)

# print(wave_name[SPA_select])
# #画图
# Show_select_ans(wave_let_absorb,wave_name,SPA_select,'SPA',save_dir)








# #结果交集
# union_select = list(set(LASSO_select).union(CARS_select,SPA_select))

# #画图
# plt.figure()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.plot(wave_name,wave_let_absorb.iloc[0, :],label='First calibration object')
# plt.scatter(wave_name[union_select],wave_let_absorb.iloc[0, union_select], marker='s', color='red')
# plt.title('UNION '+'('+str(len(union_select))+' variables'+')')
# plt.legend(['First calibration object', 'Selected variables'])
# plt.xlabel("Wavenumber(cm-1)")
# plt.ylabel("Absorbance(a.u)")
# plt.grid(True)
# plt.savefig(save_dir + 'UNION_selection'+'.jpg')
# plt.show()

# subset = wave_let_absorb.iloc[:,union_select]
# # select_ans = SVM_search(wave_let_absorb,Label)
# # print(select_ans)
# # result.append(['UNION',select_ans,wave_name[union_select]])
# #计算时间
# performance = model_evaluation({'C': 27, 'gamma': 100, 'kernel': 'rbf'},subset,Label)
# print(performance)
# model_performance.append(['UNION',performance])




# #基线
# performance = model_evaluation({'C': 2, 'gamma': 10, 'kernel': 'rbf'},wave_let_absorb,Label)
# print(performance)
# model_performance.append(['Baseline',performance])


# #打印结果
# print(result)
# print(model_performance)