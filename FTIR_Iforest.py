import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from FTIR_show import getdata
from FTIR_Dim_reduction import dim_pca




######Read in data
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




######pca
# absorb_pca = dim_pca(absorb)



def Iforest(absorb,save_dir,name):
    '''
    返回离群者,保存者
    '''
    #####Iforest
    clf = IsolationForest(n_jobs=-1,max_features=3,random_state=46) 
    clf.fit(absorb)  
    scores_pred = clf.decision_function(absorb)
    # print(scores_pred)
    # print(len(scores_pred))

    ####draw
    Outlier = []
    Remain = []
    for i in range(len(scores_pred)):
        if scores_pred[i] < -0.09:
            Outlier.append(i)
        else:
            Remain.append(i)
    print(name+'离群者:',Outlier)
    # plt.figure(dpi=600)
    # plt.rcParams['font.sans-serif'] = ['Arial'] 
    # plt.rcParams['axes.unicode_minus'] = False  
    # plt.scatter([i for i in range(len(scores_pred))],scores_pred)
    # plt.xlabel('Sample number', fontsize=14,weight="bold")
    # plt.ylabel('Score', fontsize=14,weight="bold")
    # plt.title(name + ' Iforest Detection', fontsize=16,weight="bold")
    # plt.axhline(y=-0.09, c="r", ls="--",lw=2)
    # plt.xticks(fontsize=13,weight="bold")
    # plt.yticks(fontsize=13,weight="bold")
    # plt.savefig(save_dir + 'Iforest_'+ name +'_.jpg')
    # plt.show()
    return Outlier,Remain


# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'
# Outlier,Remain = Iforest(absorb.iloc[:120,:],save_dir,'2B')
# print(Outlier,Remain)