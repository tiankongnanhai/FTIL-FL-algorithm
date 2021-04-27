import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from FTIR_Iforest import Iforest
from FTIR_show import getdata
from FTIR_Pretreatment import get_baseline
from sklearn.model_selection import LeaveOneOut
import datetime
import warnings
warnings.filterwarnings("ignore")


def Time_calculation(model_parameters,subset,Label):
    #训练svm分类器,['LASSO',{'C': 7, 'gamma': 1000, 'kernel': 'rbf'}]
    Oneleft = LeaveOneOut()
    classifier=svm.SVC(C=model_parameters[0],kernel=[1],gamma=model_parameters[2],decision_function_shape='ovr') # ovr:一对多策略
    #开始时间
    start = datetime.datetime.now()
    for train,test in Oneleft.split(subset):
        classifier.fit(subset.iloc[train,:],Label[train])
        Y_pred = classifier.predict(subset.iloc[test,:])
        # Y_pred_record.append(Y_pred)
    #结束时间
    end = datetime.datetime.now()
    # time_record.append(['LASSO',(end-start).microseconds])
    return (end-start).microseconds





# #读入数据
# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'
# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #波长名
# wave_name = DATA_2B['wave']

# #图片保存路径
# save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'

# #csv保存地址
# csv_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//预处理测试结果//'

# #去除离群样本
# DATA_2B = DATA_2B.drop(['wave'],axis=1)
# DATA_2B = DATA_2B.T
# DATA_A549 = DATA_A549.drop(['wave'],axis=1)
# DATA_A549 = DATA_A549.T
# Outlier_2B,Remain_2B = Iforest(DATA_2B,save_dir,'2B')
# Outlier_A549,Remain_A549 = Iforest(DATA_A549,save_dir,'A549')
# DATA_2B = DATA_2B.iloc[Remain_2B,]
# DATA_A549 = DATA_A549.iloc[Remain_A549,]

# absorb = pd.concat([DATA_2B,DATA_A549])

# #样本名
# sample_name = absorb._stat_axis.values.tolist()

# #标签
# Label = [0 for i in range(DATA_2B.shape[0])] + [1 for i in range(DATA_A549.shape[0])]
# Label = np.array(Label)



# #小波消噪
# wave_let_absorb = get_baseline(absorb)

# Y_pred_record = []
# time_record = []