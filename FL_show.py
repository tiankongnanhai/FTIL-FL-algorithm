import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
import json
import copy



#######Define function
def getdata(list_dir):
    '''
    Enter the file address name and return to the dataframe
    '''
    draw_dir = os.listdir(list_dir)
    #Read the first csv for subsequent stitching
    df1 = pd.read_excel(list_dir + draw_dir[0],header=None,skiprows=5)#skip front 5 row

    #remain first and second col --- wave and absorbance
    df1 = df1.iloc[:,[0,1]]

    #if the last col is null, drop
    if df1.iloc[:,-1].isnull().all():
        df1.drop([df1.columns.values.tolist()[-1]],axis=1,inplace=True)
    
    #name
    col_name = list_dir[-3] +'_'+ str(0)
    df1.columns = ['wave']+[col_name]
    # print(df1.head(5))
    #Traverse folders, merge data
    i = 1
    for csv in draw_dir:
        if csv.endswith('.xlsx'):
            if csv == draw_dir[0]:#Avoid the first repetition
                continue
            filename = list_dir + csv
            df2 = pd.read_excel(filename, header=None, skiprows=5)
            df2 = df2.iloc[:,[0,1]]
            if df2.iloc[:,-1].isnull().all():
                df2.drop([df2.columns.values.tolist()[-1]],axis=1,inplace=True)
            col_name = list_dir[-3] +'_'+ str(i)
            i += 1
            df2.columns = ['wave'] + [col_name]
            df1 = pd.merge(df1, df2, on='wave') #Combine df2 data with df1
            # print(df1.shape)
    #Complete missing values
    df1.fillna(method='ffill',inplace=True)
    # print(df1.shape)
    # print(df1.head(5))
    return df1

def shaw_all(df1,names,save_dir,pic_name,colors):
    '''
    Enter dataframe and return to FTIR diagram
    '''
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial']
    for i in range(len(names)):
        if names[i].startswith('4'):
            color = colors[0]
        elif names[i].startswith('6'):
            color = colors[1]
        elif names[i].startswith('8'):
            color = colors[2]
        else:
            color = colors[3]


        plt.plot(df1['wave'],df1[names[i]],color=color,linewidth=0.5,linestyle='-')
    plt.xlabel("Wave", fontsize=14)
    plt.ylabel("Absorbance", fontsize=14)
    # plt.legend()
    # plt.show()
    plt.grid(True)
    plt.xlabel("Wavenumber (nm)", fontsize=14,weight="bold")
    plt.ylabel("Absorbance (a.u.)", fontsize=14,weight="bold")
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.savefig(save_dir + pic_name+'_Absorbance'+'.jpg')
    plt.clf()

def show_mean(wave, df1,save_dir,pic_name_2, kind, colors):
    '''
    Enter dataframe and return the average FTIR graph
    '''
    # df1['6H_mean'] = df1.iloc[:,[0:21]].mean(axis=1)
    # df1['7H_mean'] = df1[list_A549].mean(axis=1)
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial']
    for i in range(len(kind)):
        plt.plot(wave,df1[kind[i]],color=colors[i],linewidth=1,linestyle='-',label=kind[i])
    plt.xlabel("Wavenumber (nm)", fontsize=14,weight="bold")
    plt.ylabel("Absorbance (a.u.)", fontsize=14,weight="bold")
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.savefig(save_dir + pic_name_2+'_Absorbance'+'.jpg')
    plt.clf()


def Show_select_ans(absorb,wave_name,select,algorithm_name,save_dir):
    plt.figure(dpi=600)
    plt.rcParams['font.sans-serif'] = ['Arial'] 
    plt.rcParams['axes.unicode_minus'] = False  
    plt.plot(wave_name,absorb.iloc[61, :],label='First calibration object')
    plt.scatter(wave_name[select],absorb.iloc[61, select], marker='s', color='red')
    plt.title(algorithm_name+' select '+'('+str(len(select))+' variables'+')',fontsize=16,weight="bold")
    plt.legend(['First calibration object', 'Selected variables'])
    plt.xlabel("Wavenumber (nm)", fontsize=14,weight="bold")
    plt.ylabel("Absorbance (a.u.)", fontsize=14,weight="bold")
    plt.xticks(fontsize=13,weight="bold")
    plt.yticks(fontsize=13,weight="bold")
    plt.grid(True)
    plt.savefig(save_dir + algorithm_name +'_selection'+'.jpg')
    plt.show()



#####Read in data
PH = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FL//花青素吸收_20210406//myq0410//myq0406//'
PH_dir = os.listdir(PH)

FL_DATA = getdata(PH + PH_dir[0] + '//')
for dir in PH_dir:
    if dir == PH_dir[0]:
        continue
    DATA = getdata(PH + dir + '//')
    FL_DATA = pd.merge(FL_DATA,DATA,on='wave')

#储存FL_DATA
FL_DATA.to_json("FL_DATA.json")
# print(FL_DATA.shape)
print(FL_DATA.columns)


#Draw a general picture
names = FL_DATA.columns.tolist()#Send out column names in list form
names.remove('wave')
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FL//'
pic_name = 'PH'
colors = ['green','red','blue','black']
shaw_all(FL_DATA,names,save_dir,pic_name,colors)

#Draw an average graph
kind = ['6.4','6.6','6.8','7']
#求每个样本的平均值
Mean_FL = pd.DataFrame()
n = FL_DATA.shape[1]
i = 0
start = 1
while start < n:
    Mean_FL[kind[i]] = FL_DATA.iloc[:,[start,start+20]].mean(axis=1)
    i += 1
    start += 21
# list_6H = DATA_PH6.columns.tolist()#Send out column names in list form
# list_6H.remove('wave')
# list_7H = DATA_PH7.columns.tolist()#Send out column names in list form
# list_7H.remove('wave')
pic_name_2 = 'PH_mean'
wave = FL_DATA['wave']
show_mean(wave,Mean_FL,save_dir,pic_name_2, kind, colors)