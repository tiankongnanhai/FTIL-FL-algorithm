import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from FTIR_show import getdata
from FTIR_Pretreatment import get_baseline




######Read in data
DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

DATA_2B = getdata(DATA_2B_dir)
DATA_A549 = getdata(DATA_A549_dir)

#Merge into a data set
FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
FTIR_DATA= FTIR_DATA.T

#Check the stitching result
# print(FTIR_DATA._stat_axis.values.tolist())
# print(FTIR_DATA.shape)

#Create label, 0 means 2B, 1 means A549
Label = [0 for i in range(DATA_2B.shape[1]-1)] + [1 for i in range(DATA_A549.shape[1]-1)]
Label = np.array(Label)

#Convert tags into dummy variables
# dummy_Label = pd.get_dummies(Label,prefix='type')

wave_name = np.array(FTIR_DATA.iloc[0,:])
wave_name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))

absorb = pd.DataFrame(FTIR_DATA.iloc[1:].values,columns=wave_name)
# print(absorb.shape)





def mean_sta(absorb):
    '''
    输入：(样本，变量)
    输出：(均值，标准差)
    '''
    #样本量
    n_sample = absorb.shape[0]

    Mean_Std_matrix = np.zeros((n_sample,2))
    #计算每个样本的均值和标准差
    for raw in range(n_sample):
        cur_mean = np.mean(absorb.iloc[raw,:])
        cur_std = np.std(absorb.iloc[raw,:])
        Mean_Std_matrix[raw,:] = [cur_mean,cur_std]
    return Mean_Std_matrix


Original_matrix = mean_sta(absorb)

#LASSO
LASSO_absorb = absorb[['849.5037','509.6058','562.6395','1808.4500','2159.4370','1164.3310','3744.1800']]

matrix = mean_sta(LASSO_absorb)

#样本量
n_sample = absorb.shape[0]

####查看数据分布
# 二维
plt.figure(figsize=(8, 8), facecolor='w')
plt.rcParams['axes.unicode_minus'] =False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(2,1,1)
plt.scatter(Original_matrix[:n_sample//2, 0], Original_matrix[:n_sample//2, 1], c = "red", marker='o', label='2B')
plt.scatter(Original_matrix[n_sample//2:, 0], Original_matrix[n_sample//2:, 1], c = "blue", marker='o', label='A549')
plt.title('Original distributed')
plt.legend()
plt.subplot(2,1,2)
plt.scatter(matrix[:n_sample//2, 0], matrix[:n_sample//2, 1], c = "red", marker='o', label='2B')
plt.scatter(matrix[n_sample//2:, 0], matrix[n_sample//2:, 1], c = "blue", marker='o', label='A549')
plt.title('LASSO distributed')
plt.legend()
plt.show()


# ########Kmeans
# #构造聚类器
# estimator = KMeans(n_clusters=2)
# estimator.fit(absorb[['1511.9430','1512.4250','1512.9070']])
# #获取聚类标签
# label_pred = estimator.labels_

# #计算准确率：
# Truth = 0
# for i in range(n_sample):
#     if label_pred[i] == Label[i]:
#         Truth += 1
# print('Kmeans accuracy %4.2f' % (Truth/n_sample))

# x0 = absorb[label_pred == 0]
# x1 = absorb[label_pred == 1]

# fig = plt.figure(figsize=(12, 6), facecolor='w')
# plt.rcParams['axes.unicode_minus'] =False
# plt.rcParams['font.sans-serif'] = ['SimHei']
# #样本原本分布
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(absorb.iloc[:n_sample//2, 0], absorb.iloc[:n_sample//2, 1], absorb.iloc[:n_sample//2, 2],c = "red",alpha=0.4, marker='o', label='2B' )
# ax.scatter(absorb.iloc[n_sample//2:, 0], absorb.iloc[n_sample//2:, 1], absorb.iloc[n_sample//2:, 2],c = "blue",alpha=0.4, marker='o', label='A549')
# plt.title('Original distributed')
# plt.legend(loc='lower right')
# #聚类结果
# ax2 = fig.add_subplot(212, projection='3d')
# ax2.scatter(x0.iloc[:, 0], x0.iloc[:, 1], x0.iloc[:, 2],c = "red", marker='*',alpha=0.4, label='2B' )
# ax2.scatter(x1.iloc[:, 0], x1.iloc[:, 1], x1.iloc[:, 2],c = "blue", marker='*',alpha=0.4, label='A549')
# plt.title('Kmeans result')
# plt.legend(loc='lower right')

# plt.show()

# #########层次聚类
# Z = linkage(absorb, 'complete')
# f = fcluster(Z,4,'distance')
# fig = plt.figure(figsize=(20, 12))
# dn = dendrogram(Z)
# plt.show()