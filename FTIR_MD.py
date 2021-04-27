import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from FTIR_show import getdata




#######Read in data

# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #Merge into a data set
# FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
# FTIR_DATA = FTIR_DATA.T
# wave = FTIR_DATA.iloc[0]
# absorb = FTIR_DATA.iloc[1:]



def md(absorb):
    '''
    :param absorb: shape (n_samples, n_features)
    :param threshold: float
    :return : 
            outline_sample(sample_id)
    '''
    sample_num = absorb.shape[0]
    threshold = sample_num*0.005

    #Perform pca dimensionality reduction on the sample to 3 dimensions
    pca = PCA(n_components=3)
    absorb_pca = pca.fit_transform(absorb)

    #Solve for average spectrum
    mean_absorb = np.mean(absorb_pca,axis=0)

    #Find the covariance matrix
    cov_matrix = np.cov(absorb_pca.T)

    #The inverse of the covariance matrix
    cov_matrix_inverse = np.linalg.inv(cov_matrix)

    #Calculate the Mahalanobis distance from all samples to the average spectrum
    md_distance = np.zeros(sample_num)

    for i in range(sample_num):
        delta = absorb_pca[i]-mean_absorb
        d=np.sqrt(np.dot(np.dot(delta,cov_matrix_inverse),delta.T))
        md_distance[i]=d

    #Calculate the average and standard deviation of Mahalanobis distance
    mean_md = np.mean(md_distance)
    md_distance.reshape(sample_num,1)
    std_md = np.std(md_distance,axis=0)

    #Check for abnormal samples
    outline_sample = []
    for i in range(sample_num):
        if abs((md_distance[i]-mean_md)/std_md) > threshold:
            outline_sample.append(i)
        else:
            pass
    
    return outline_sample


ans = md(absorb)
print(ans)