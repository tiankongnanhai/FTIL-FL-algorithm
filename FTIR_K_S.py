import numpy as np
import sklearn
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_auc_score
from FTIR_show import getdata
from FTIR_Pretreatment import mean_centralization,standardlize,sg,msc,snv,D1,D2



def ks(x, y, test_size=0.3):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]             
    N = round((1-test_size) * M)
    samples = np.arange(M)     
 
    D = np.zeros((M, M))       
 
    for i in range((M-1)):
        xa = x.iloc[i]
        for j in range((i+1), M):
            xb = x.iloc[j]
            D[i, j] = np.linalg.norm(xa-xb) 
 
    maxD = np.max(D, axis=0)             
    index_row = np.argmax(D, axis=0)    
    index_column = np.argmax(maxD)      
 
    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)                   
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]  
 
    for i in range(2, N):  
        pool = np.delete(samples, m[:i]) 
        dmin = np.zeros((M-i))        
        for j in range((M-i)):        
            indexa = pool[j]         
            d = np.zeros(i)           
            for k in range(i):         
                indexb = m[k]         
                if indexa < indexb:   
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)    
        dminmax[i] = np.max(dmin)   
        index = np.argmax(dmin)     
        m[i] = pool[index]          
 
    m_complement = np.delete(np.arange(x.shape[0]), m)    
 
    spec_train = x.iloc[m]
    target_train = y[m]
    spec_test = x.iloc[m_complement]
    target_test = y[m_complement]
    return spec_train, spec_test, target_train, target_test




#######Read in data
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



# #########pretreatment
# st_absorb = standardlize(absorb)
# D2_absorb = D2(st_absorb)



# #########Use ks algorithm for sample partitioning
# train_data,test_data,train_label,test_label = ks(absorb,Label)
# # print('split success')



# #########SVM
# clf = svm.SVC(C=17,kernel='rbf',gamma=10,class_weight='balanced',decision_function_shape='over',probability=True) # ovr:一对多策略
# clf.fit(train_data, train_label)
# predict_prob_y = clf.predict_proba(train_data)[:,1]

# print("训练集准确率：",clf.score(train_data,train_label))
# print("测试集准确率：",clf.score(test_data,test_label))
# print('AUC:',roc_auc_score(train_label,predict_prob_y))