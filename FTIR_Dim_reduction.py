from FTIR_show import getdata
from FTIR_Pretreatment import mean_centralization,standardlize,sg,msc,snv,D1,D2
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn
import numpy as np
import pandas as pd 
np.set_printoptions(threshold=10000) # 显示多少行
np.set_printoptions(linewidth=100) # 横向多宽



# #######Read in data

# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #Merge into a data set
# FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
# FTIR_DATA = FTIR_DATA.T

# #Create label, 0 means 2B, 1 means A549
# Label = [0 for i in range(DATA_2B.shape[1]-1)] + [1 for i in range(DATA_A549.shape[1]-1)]
# Label = np.array(Label)
# # print(len(Label))





#######sg+msc+snv
# st_absorb = standardlize(absorb)
# D2_absorb = D2(st_absorb)





########dimention reduce
def dim_pca(absorb):
    pca = PCA(n_components=3)
    absorb_pca = pca.fit_transform(absorb)
    return absorb_pca


def dim_tsne(absorb):
    '''
    Enter the original data and return the data after tsne dimensionality reduction
    '''
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000, learning_rate=500)
    absorb_tsne = tsne.fit_transform(absorb)
    return absorb_tsne



def vif(x, thres=10.0):
    '''
    每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量
    '''
    X_m = np.matrix(x)
    VIF_list = [variance_inflation_factor(X_m, i) for i in range(X_m.shape[1])]
    maxvif=pd.DataFrame(VIF_list,index=x.columns,columns=["vif"])
    col_save=list(maxvif[maxvif.vif<=float(thres)].index)
    col_delete=list(maxvif[maxvif.vif>float(thres)].index)
    print(len(col_delete))
    print(maxvif)
    print('delete Variables:', col_delete)
    return x[col_save]


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train_data, train_label, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)




def Lasso_select(wave_name,absorb,Label,save_dir):
    '''
    输出系数不为0的波长
    '''
    ############lasso
    #图片保存路径
    # save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'

    #数据集划分
    train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(absorb,Label, random_state=1,train_size=0.9,test_size=0.1)
    #将特征放缩到（-1，1）
    # min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    # train_data = min_max_scaler.fit_transform(train_data)
    # test_data = min_max_scaler.transform(test_data)

    #调用LassoCV函数，并进行交叉验证，cv=10
    model_lasso = LassoCV(alphas = [1, 0.1, 0.01, 0.001, 0.0005], cv=10).fit(train_data, train_label)

    #模型所选择的最优正则化参数alpha
    print(model_lasso.alpha_)

    #各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
    # print(model_lasso.coef_)

    #输出看模型最终选择了几个特征向量，剔除了几个特征向量
    coef = pd.Series(model_lasso.coef_, index = train_data.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    #输出所选择的最优正则化参数情况下的残差平均值，因为是10折，所以看平均值
    # print(rmse_cv(model_lasso).mean())

    #画出特征变量的重要程度
    Lasso_picked_var = []
    var_coef = []
    for index,value in coef.items():
        if value != 0:
            Lasso_picked_var.append(index)
            var_coef.append(value)

    #波长名
    # wave_name = np.array(FTIR_DATA.iloc[0,:])
    # wave_name = np.array(list(map(lambda x: "%.4f" % x , wave_name)))

    plt.figure(dpi=300)
    plt.rcParams['axes.unicode_minus'] =False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.bar(wave_name[Lasso_picked_var],var_coef)
    plt.tick_params(labelsize=4)
    plt.title("Coefficients in the Lasso Model")
    plt.savefig(save_dir + 'Lasso_var'+'.jpg')
    plt.show() 

    # #选择波长分布
    # plt.figure()
    # plt.plot(wave_name,absorb.iloc[0, :])
    # plt.scatter(wave_name[Lasso_picked_var], absorb.loc[0, Lasso_picked_var], marker='s', color='r')
    # plt.title('Lasso')
    # plt.legend(['First calibration object', 'Selected variables'])
    # plt.xlabel('Variable index')
    # plt.grid(True)
    # plt.savefig(save_dir + 'Lasso_1'+'.jpg')
    # plt.show()

    #输出系数不为0的波长
    print(wave_name[Lasso_picked_var])
    return Lasso_picked_var

# absorb = FTIR_DATA.iloc[1:]
# result = Lasso_select(FTIR_DATA,absorb,Label)
# print(result)



########PCA
# candidate_components = range(1, 30, 1)
# explained_ratios = []
# for c in candidate_components:
#     pca = PCA(n_components=c)
#     X_pca = pca.fit_transform(st_absorb)
#     explained_ratios.append(np.sum(pca.explained_variance_ratio_))

# plt.figure(figsize=(10, 6), dpi=144)
# plt.grid()
# plt.plot(candidate_components, explained_ratios)
# plt.xlabel('Number of PCA Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained variance ratio for PCA')
# plt.yticks(np.arange(0.5, 1.05, .05))
# plt.xticks(np.arange(0, 30, 1))
# plt.show()


# scatter
# absorb_tsne = dim_tsne(absorb)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(absorb_tsne[:,0], absorb_tsne[:,1], absorb_tsne[:,2])
# plt.show()


# plt.figure()
# plt.scatter(absorb_tsne[:,0],absorb_tsne[:,1])
# plt.show()