from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import sklearn
from FTIR_show import getdata
import warnings
warnings.filterwarnings("ignore")





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

# print(len(Label))
absorb = FTIR_DATA.iloc[1:]






#训练集、测试机划分
train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(absorb,Label, random_state=1,train_size=0.7,test_size=0.3)

#分类型决策树
clf = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0, n_jobs=6)

#训练模型
clf.fit(train_data,train_label)

#评估模型准确率
print('训练集准确率：%s' % clf.score(train_data,train_label))
print('测试集准确率：%s' % clf.score(test_data,test_label))

#各特征重要性
# print('各特征重要性：%s' % clf.feature_importances_)