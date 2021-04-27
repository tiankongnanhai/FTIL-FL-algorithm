import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.stats as stats
from FTIR_Cars import cars
from FITR_SPA import SPA
from FTIR_UVE import UVE
from FTIR_Dim_reduction import Lasso_select
from FTIR_show import Show_select_ans



def Generate_simulation_data(N_sample,N_var):
    '''
    输入：样本数，变量数
    返回：模拟数据SIMUI(narray):(N_sample,N_var*2), Label(narray):N_sample
    '''
    #生成从0到1的随机数矩阵(N_sample,N_var)
    S1 = np.random.rand(N_sample,N_var)

    pca = PCA(n_components=5)
    pca.fit(S1)
    Score_vectors = pca.transform(S1)
    Loading_vectors = pca.components_

    #得到无噪声矩阵
    SIM = np.dot(Score_vectors,Loading_vectors)

    #生成无信息变量矩阵
    low = np.min(SIM)
    high = np.max(SIM)
    UI = np.random.uniform(low,high,(N_sample,N_var))

    #合并无噪声矩阵和无信息变量矩阵
    SIMUI = np.c_[SIM,UI]

    #样本标签
    Label = [0]*(N_sample)
    for i in range(N_sample):
        Label[i] = Score_vectors[i][0]*5 + Score_vectors[i][1]*4 + Score_vectors[i][2]*3 + Score_vectors[i][3]*2 + Score_vectors[i][4]*1

    Label = np.array(Label)

    #给每一个样本添加(0,0.005)正态分布噪声
    low, high = 0, 0.005
    mu = (low+high)/2
    sigma = (high-low)/6
    # Noise_matrix = np.zeros((N_sample, N_var))
    for row in range(N_sample):
        SIMUI[row,:] +=  np.random.normal(mu, sigma, N_var*2)
    
    SIMUI = pd.DataFrame(SIMUI)
    return SIMUI, Label




# N_sample = 25
# N_var = 100
# wave_name = np.array([i for i in range(N_var*2)])
# absorb,Label = Generate_simulation_data(N_sample,N_var)
# #csv保存地址
# csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//模拟数据//'
# #图片保存路径
# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//模拟数据//'

# #CARS
# mc_times = 500
# absorb, Label = Generate_simulation_data(N_sample,N_var)
# # name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))
# RMSECV_min,best_subset,CARS_select = cars(mc_times,wave_name,absorb,Label,csv_dir+'cars.csv',save_dir)
# print(wave_name[CARS_select])
# #画图
# Show_select_ans(absorb,wave_name,CARS_select,'m_CARS',save_dir)

# #SPA
# Xcal, Xval, ycal, yval = sklearn.model_selection.train_test_split(absorb, Label, test_size=0.1, random_state=0)
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) 
# Xcal = min_max_scaler.fit_transform(Xcal)
# Xval = min_max_scaler.transform(Xval)
# SPA_select, var_sel_phase2 = SPA().spa(Xcal, ycal, save_dir, wave_name, m_min=1, m_max=20, Xval=Xval, yval=yval, autoscaling=1)
# Show_select_ans(absorb,wave_name,SPA_select,'m_SPA',save_dir)

# LASSO
# LASSO_select = Lasso_select(wave_name,absorb,Label,save_dir)
# Show_select_ans(absorb,wave_name,LASSO_select,'m_LASSO',save_dir)

#UVE
# UVE_select = UVE(absorb,Label,wave_name)
# Show_select_ans(absorb,wave_name,UVE_select,'m_UVE',save_dir)