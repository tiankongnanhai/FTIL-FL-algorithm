import numpy as np
import pandas as pd 
from FTIR_show import getdata
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import math
from sklearn.model_selection import LeaveOneOut




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



def UVE(absorb,Label,wave_name):

    #原始光谱矩阵
    X = np.array(absorb)

    #尺度系数
    # scale = 10**(-4)

    #样本数
    n = X.shape[0]

    #变量数
    p = X.shape[1]

    #生成(0,1)随机矩阵
    random_matrix = np.random.rand(n,p)
    # random_matrix = random_matrix*scale

    #将原始矩阵与随机矩阵沿列拼接
    XR = np.c_[X,random_matrix]

    #将XR转为dataframe
    col_names = [x for x in range(2*p)]
    XR = pd.DataFrame(XR,columns=col_names)
    # print(XR.head(5))

    #RMSEP值监视器
    RMSEP = float('inf')
    drop = False

    #当XR的列数不大于p时强制停止,若RMSEP出现极小值则退出循环
    while XR.shape[1] > p:
        # print(XR.shape)
        #建立回归系数矩阵[n,2p]
        coef_matrix = np.zeros((n,XR.shape[1]))

        #建立pls模型，并计算RMSEP
        pls_model = PLSRegression(copy=True, max_iter=100, n_components=2, scale=True,tol=1e-06)
        mse = 0
        Oneleft = LeaveOneOut()
        for train,test in Oneleft.split(XR):
            pls_model.fit(XR.iloc[train,:],Label[train])
            coef = pls_model.coef_
            coef = coef.flatten()
            #保存coef
            coef_matrix[int(test),:] = coef
            Y_pred = pls_model.predict(XR.iloc[test,:])
            # #将输出结果变成0,1离散值，阈值为0.5
            # if Y_pred > 0.5:
            #     Y_pred = 1
            # else:
            #     Y_pred = 0
            mse += (Y_pred-Label[test])**2
        # print(Y_pred_record)
        #判断RMSEP是否达到极小值
        new_RMSEP = (mse/p)**(1/2)
        if new_RMSEP == RMSEP:
            break
        elif new_RMSEP > RMSEP:
            if drop == True:
                break
        else:
            RMSEP = min(RMSEP,new_RMSEP)
            drop = True
        # print(RMSEP)

        #计算C值
        C_value = np.zeros(XR.shape[1])
        for col in range(XR.shape[1]):
            cur_mean = np.mean(coef_matrix[:,col])
            cur_std = np.std(coef_matrix[:,col])
            C_value[col] = cur_mean/cur_std

        #计算随机矩阵的C_max
        C_value = np.abs(C_value)
        # print((XR.shape[1]*(1/2)))
        C_max = np.max(C_value[int((XR.shape[1]*(1/2))):])

        #删除原始矩阵中c值小于C_max的列
        cur_col_name = XR.columns.values.tolist()
        delete_col = []
        for i in range(int(XR.shape[1]*(1/2))):
            if C_value[i] < C_max:
                delete_col.append(cur_col_name[i])#删除X中var
                delete_col.append(cur_col_name[i+int(XR.shape[1]*(1/2))])#删除对应R中的var
        
        #更新
        XR_new = XR.drop(columns=delete_col,inplace=False)
        XR = XR_new.copy()

    #无信息变量消除结果
    XR_index = XR.columns.values.tolist() 
    var_num = int(XR.shape[1]*(1/2))
    selet_index = XR_index[:var_num]
    # result_wave = list(FTIR_DATA.iloc[0,selet_index])

    save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据/测试结果//'

    # print(type(FTIR_DATA.iloc[0,:]))
    # print(FTIR_DATA.iloc[0,:])
    # plt.figure()
    # plt.plot(wave_name,np.array(X[0, :]))
    # plt.scatter(wave_name[selet_index], np.array(X[0, selet_index]), marker='s', color='r')
    # plt.title('UVE')
    # plt.legend(['First calibration object', 'Selected variables'])
    # plt.xlabel('Variable index')
    # plt.grid(True)
    # plt.savefig(save_dir + 'UVE'+'.jpg')
    # plt.show()

    return selet_index


# wave_name = FTIR_DATA.iloc[0,:]
# result = UVE(absorb,Label,wave_name)
# print(result)