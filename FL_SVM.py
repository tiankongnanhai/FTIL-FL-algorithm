import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from FL_show import getdata
# print(cpu_count())#View the number of cpu cores = 6




######Read in data
# FL_DATA = pd.read_json("FL_DATA.json")
# Label = []
# for i in range(4):
#     Label += [i]*21
# Label = np.array(Label)
# # print(len(Label))
# FL_DATA= FL_DATA.T
# absorb = FL_DATA.iloc[1:]
# print(absorb.shape)




#########SVM  #best 'C': 1, 'gamma': 1, 'kernel': 'rbf'
def SVM_GridSearch(x,y):
    # Partition data set
    # train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(x,y, random_state=1,\
    #     train_size=0.7,test_size=0.3)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = min_max_scaler.fit_transform(x)
    # test_data = min_max_scaler.transform(test_data)

    print('data already')

    #Train svm classifier
    param_grid = [      {'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000, 5000 ,10000, 50000 ,100000],
                        'kernel': ['rbf']
                        },
                        {
                        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                        'kernel': ['linear']
                        },
                        {
                        'degree':[1, 3, 5, 7, 9, 11],
                        'kernel': ['poly']
                        },
                        {
                        'gamma':[1, 2, 3, 4],
                        'coef0':[0.2, 0.4, 0.6, 0.8, 1],
                        'kernel': ['sigmoid']
                        }]

    clf = GridSearchCV(SVC(class_weight='balanced',decision_function_shape='ovo'),\
        param_grid=param_grid, scoring='f1_micro', n_jobs=6, cv=10)

    clf = clf.fit(x, y)
    # print(clf.best_params_, clf.best_score_)

    #Return the best parameters、 maximum accuracy value、 training accuracy、 test accuracy
    return [clf.best_params_, clf.best_score_]
    # return [clf.best_params_, clf.best_score_, clf.best_estimator_.score(train_data, train_label), clf.best_estimator_.score(test_data, test_label)]




#########Without Pretreatment
# select_ans = SVM_GridSearch(absorb,Label)
# print(select_ans)






#划分数据集
# x = np.array(absorb)
# y = Label

# train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(x,y, random_state=1,\
#     train_size=0.7,test_size=0.3)


# classifier = SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
# classifier.fit(train_data,train_label)

# #训练集： 0.896551724137931
# #测试集： 0.8076923076923077
# print("训练集：",classifier.score(train_data,train_label))
# print("测试集：",classifier.score(test_data,test_label))

# #确定坐标轴范围
# x1_min, x1_max=x[:,0].min(), x[:,0].max() #第0维特征的范围
# x2_min, x2_max=x[:,1].min(), x[:,1].max() #第1维特征的范围
# x1,x2=np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j ] #生成网络采样点
# grid_test=np.stack((x1.flat,x2.flat) ,axis=1) #测试点
# #指定默认字体
# matplotlib.rcParams['font.sans-serif']=['SimHei']
# #设置颜色
# cm_light=matplotlib.colors.ListedColormap(['lightgreen', 'lighrcoral', 'royalblue','gold','silver'])
# cm_dark=matplotlib.colors.ListedColormap(['green','red','blue','yellow','black'] )

# grid_hat = classifier.predict(grid_test)       # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c = y, s=30,cmap=cm_dark)  # 样本
# plt.scatter(test_data[:,0],test_data[:,1], c=test_label, s=30, edgecolors='k', zorder=2, cmap=cm_dark) #圈中测试集样本点
# plt.xlabel('First Var', fontsize=13)
# plt.ylabel('Second Var', fontsize=13)
# plt.xlim(x1_min,x1_max)
# plt.ylim(x2_min,x2_max)
# plt.title('SVM Classify')
# plt.show()