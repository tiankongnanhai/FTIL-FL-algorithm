from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FTIR_Pretreatment import sg

#The result shows that window_size=51, polynomial_order=3 works well

filename = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//X-A-5.CSV'
df1 = pd.read_csv(filename,header=None,names=['wave','Absorb'])

Y0 = df1['wave']
X0 = df1['Absorb']
x=np.array(Y0)
y=np.array(X0)

for window_size in range(9,99,2):
    for polynomial_order in range(1,4,1):
        zs1=sg(y, window_size, polynomial_order)
        plt.figure()
        plt.plot(x,y,color='red',lw=0.5)
        plt.title('Savitzky-Golay'+'-'+str(window_size)+'-'+str(polynomial_order))
        plt.plot(x,zs1,color='green',lw=0.5)
        # plt.show()
        plt.savefig('D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//SG平滑实验结果//'+'Savitzky-Golay'+\
        '-'+str(window_size)+'-'+str(polynomial_order)+'.jpg')