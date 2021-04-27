import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd




#读入结果并画图
result = [  ['CARS', [{'C': 23, 'gamma': 1000, 'kernel': 'rbf'}, 0.9334639652962634, 0.9006211180124224, 0.7714285714285715]],\
            ['SPA', [{'C': 3, 'gamma': 5000, 'kernel': 'rbf'}, 0.9551168293404317, 0.968944099378882, 0.8428571428571429]],\
            ['LASSO', [{'C': 7, 'gamma': 1000, 'kernel': 'rbf'}, 0.9806763285024155, 0.9627329192546584, 0.8285714285714286]],\
            ['UVE', [{'C': 29, 'gamma': 1, 'kernel': 'rbf'}, 0.9783348121857438, 0.9565217391304348, 0.8285714285714286]], \
            ['Union', [{'C': 27, 'gamma': 100, 'kernel': 'rbf'}, 0.9879227053140096, 0.968944099378882, 0.8857142857142857]]
        ]

#AUC值
AUC = []
Train_accuracy = []
Test_accuracy = []

for item in result:
    index = item[1]
    for i in range(len(index)):
        if i==1:
            AUC.append(index[i]*100)
        elif i==2:
            Train_accuracy.append(index[i]*100)
        elif i==3:
            Test_accuracy.append(index[i]*100)


#基线
#['Baseline', [{'C': 21, 'gamma': 1, 'kernel': 'rbf'}, 0.9912008281573498, 0.9627329192546584, 0.8428571428571429]]

#图片保存路径
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'

x_index = ['CARS','SPA','LASSO','UVE','Union']
positions = np.arange(len(x_index))


plt.figure()
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots()
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
plt.plot(x_index,AUC,color='green',marker = "o",linewidth=1,linestyle='-',label='AUC')
plt.plot(x_index,Train_accuracy,color='blue',marker = "v",linewidth=1,linestyle='-',label='Train accuracy')
plt.plot(x_index,Test_accuracy,color='red',marker = "s",linewidth=1,linestyle='-',label='Test accuracy')
plt.axhline(y=99.12008281573498, c="green", ls="--",lw=2,label='AUC baseline')
plt.axhline(y=96.27329192546584, c="blue", ls="--",lw=2,label='Train accuracy baseline')
plt.axhline(y=84.28571428571429, c="red", ls="--",lw=2,label='Test accuracy baseline')
plt.title("Feature selection  + SVM grid search ",fontsize=16,weight="bold")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.grid()
plt.savefig(save_dir + 'SVM'+'.jpg')
plt.show()



# #波长减少率
# wave_reduce = [99.69, 55.46, 99.91, 99.65, 99.29]
#'CARS'    'SPA'     'LASSO'     'UVE'    'Union'
accuracy = [84.42,89.61,92.21,90.50,94.40]
sensitive = [78.26,86.09,87.83,90.43,90.43]
specificity = [90.52,93.10,96.55,93.10,97.41]
precision = [89.11,92.52,96.19,92.86,97.20]
time = [361358,253676,362409, 532786, 383300]


plt.rcdefaults()
plt.figure()
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig, ax1 = plt.subplots()

# 成绩直方图
ax1.bar(positions, time, width=0.6, align='center', color='r', label=u"Time")
ax1.axhline(y=682139, c="red", ls="--",lw=2, label=' Time baseline')
ax1.set_xticks(positions)
ax1.set_xticklabels(x_index,fontsize=13)
# ax1.set_xlabel(u"名字")
ax1.set_ylabel(u"Time(microseconds)",fontsize=13)
max_score = max(time)
ax1.set_ylim(250000, int(max_score * 1.5))
# 成绩标签
for x,y in zip(positions, time):
    ax1.text(x, y + max_score * 0.02, y, ha='center', va='center', fontsize=13)

# 变动折线图
ax2 = ax1.twinx()
ax2.plot(positions, accuracy, 'o-', label=u"Accuracy")
ax2.axhline(y=92.64, ls="--", lw=2, label=' Accuracy baseline')
max_all_accuracy = max(accuracy)
# 变化率标签
for x,y in zip(positions, accuracy):
    ax2.text(x, y + max_all_accuracy * 0.02, ('%.1f%%' %y), ha='center', va= 'bottom', fontsize=13)

# 设置纵轴格式
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax2.yaxis.set_major_formatter(yticks)
ax2.set_ylim(50, 100)
ax2.set_ylabel(u"Accuracy",fontsize=13)

# 图例
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2, loc='center left')
plt.title(u'Model performance',fontsize=16,weight="bold")
plt.savefig(save_dir + 'performance'+'.jpg')
plt.show()