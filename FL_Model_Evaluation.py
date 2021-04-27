import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
import datetime
from sklearn import metrics
import warnings





#正确率（accuracy = （TP+TN）/ (TP+FP+FN+TN) 被分对的样本数除以所有的样本数，通常来说，正确率越高，分类器越好
#灵敏度/召回率（sensitive = TP / (TP+FN)  表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力
#特异度（specificity) = TN / (FP+TN)    表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力
#精度（precision）= TP/（TP+FP） 精度是精确性的度量，表示被分为正例的示例中实际为正例的比例
def model_evaluation(model_parameters,subset,Label):
    #训练svm分类器,['LASSO',{'C': 7, 'gamma': 1000, 'kernel': 'rbf'}]
    Y_pred_record = []
    Oneleft = LeaveOneOut()
    classifier=svm.SVC(C=model_parameters['C'],kernel=model_parameters['kernel'],gamma=model_parameters['gamma'],decision_function_shape='ovr') # ovr:一对多策略
    #开始时间
    start = datetime.datetime.now()
    for train,test in Oneleft.split(subset):
        classifier.fit(subset[train,:],Label[train])
        Y_pred = classifier.predict(subset[test,:])
        Y_pred_record.append(Y_pred)
    #结束时间
    end = datetime.datetime.now()
    run_time = (end-start).microseconds
    #计算指标
    classification_report = metrics.classification_report(Label, Y_pred_record)
    # time_record.append(['LASSO',(end-start).microseconds])
    return classification_report





# #读入数据
FL_DATA = pd.read_json("FL_DATA.json")
Label = []
for i in range(4):
    Label += [i]*21
Label = np.array(Label)
# print(len(Label))
FL_DATA= FL_DATA.T
#波长名
wave_name = FL_DATA.iloc[0]
absorb = FL_DATA.iloc[1:]

CARS_select = np.load('CARS_select.npy')
x = np.array(absorb.iloc[:,CARS_select])
y = Label
performance = model_evaluation({'C': 3, 'gamma': 10, 'kernel': 'rbf'},x,y)
print(performance)#0.8690476190476191