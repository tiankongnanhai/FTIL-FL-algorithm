import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm



#######Define function
def getdata(list_dir):
    '''
    Enter the file address name and return to the dataframe
    '''
    draw_dir = os.listdir(list_dir)
    #Read the first csv for subsequent stitching
    df1 = pd.read_csv(list_dir + draw_dir[0],header=None)
    #name
    df1.columns = ['wave',draw_dir[0][:-4]]
    #Traverse folders, merge data
    for csv in draw_dir:
        if csv.endswith('.CSV'):
            if csv[:-4] == draw_dir[0][:-4]:#Avoid the first repetition
                continue
            filename = list_dir + csv
            df2 = pd.read_csv(filename,header=None)
            df2.columns = ['wave',csv[:-4]]
            df1[csv[:-4]] = df2[csv[:-4]] #Combine df2 data with df1
    #Complete missing values
    df1.fillna(method='ffill',inplace=True)
    # #Keep the spectrum in the range of 500-2000
    # df1 = df1[(df1.wave>1400)&(df1.wave<1800)]
    return df1

def show_all(df1,names,wave_name,save_dir,pic_name,colors):
    '''
    Enter dataframe and return to FTIR diagram
    '''
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['axes.unicode_minus'] = False  
    for i in range(len(names)):
        if names[i].startswith('X-2B'):
            color = colors[0]
        else:
            color = colors[1]
        plt.plot(wave_name,df1.loc[names[i]],color=color,linewidth=0.5,linestyle='-')
    plt.xlabel("Wavenumber(cm-1)",weight="bold",fontsize=15)
    plt.ylabel("Absorbance(a.u)",weight="bold",fontsize=15)
    # plt.show()
    plt.title(pic_name,weight="bold",fontsize=15)
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.grid(True)
    plt.savefig(save_dir + pic_name+'.jpg')
    plt.show()


def show_mean(df1,sample_name,wave_name,save_dir,pic_name_2):
    '''
    Enter dataframe and return the average FTIR graph
    '''
    list_2B = []
    list_A549 = []
    for name in sample_name:
        if name.startswith('X-2B'):
            list_2B.append(name)
        else:
            list_A549.append(name)
    df1 = df1.T
    df1['2B_mean'] = df1[list_2B].mean(axis=1)
    df1['A549_mean'] = df1[list_A549].mean(axis=1)
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['axes.unicode_minus'] = False  
    plt.plot(wave_name,df1['2B_mean'],color='green',linewidth=2,linestyle='-',label='2B_mean')
    plt.plot(wave_name,df1['A549_mean'],color='red',linewidth=2,linestyle='-',label='A549_mean')
    plt.xlabel("Wavenumber(cm-1)", fontsize=14,weight="bold")
    plt.ylabel("Absorbance(a.u)", fontsize=14,weight="bold")
    plt.title(pic_name_2,fontsize=16,weight="bold")
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + pic_name_2+'.jpg')
    plt.show()

def Show_select_ans(absorb,wave_name,select,algorithm_name,save_dir):
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['axes.unicode_minus'] = False  
    plt.plot(wave_name,absorb.iloc[0, :],label='First calibration object')
    plt.scatter(wave_name[select],absorb.iloc[0, select], marker='s', color='red')
    plt.title(algorithm_name+' select '+'('+str(len(select))+' variables'+')',fontsize=16,weight="bold")
    plt.legend(['First calibration object', 'Selected variables'])
    plt.xlabel("Wavenumber (cm-1)", fontsize=14,weight="bold")
    plt.ylabel("Absorbance (a.u.)", fontsize=14,weight="bold")
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.grid(True)
    plt.savefig(save_dir + algorithm_name +'_selection'+'.jpg')
    plt.show()


######Read in data
DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

DATA_2B = getdata(DATA_2B_dir)
DATA_A549 = getdata(DATA_A549_dir)

#Merge into a data set
FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')

absorb = FTIR_DATA.drop(['wave'],axis=1)
absorb = absorb.T
wave_name = FTIR_DATA['wave']
sample_name = absorb._stat_axis.values.tolist()

#Draw a general picture
# names = FTIR_DATA.columns.tolist()#Send out column names in list form
# names.remove('wave')
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'
pic_name = 'Original spectra'
colors = ['green','red']
show_all(absorb,sample_name,wave_name,save_dir,pic_name,colors)

#Draw an average graph
list_2B = DATA_2B.columns.tolist()#Send out column names in list form
list_2B.remove('wave')
list_A549 = DATA_A549.columns.tolist()#Send out column names in list form
list_A549.remove('wave')
pic_name_2 = 'Original spectra mean'
show_mean(absorb,sample_name,wave_name,save_dir,pic_name_2)