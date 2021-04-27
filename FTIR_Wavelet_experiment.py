import pywt
import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.signal import savgol_filter


# 求SureShrink法阈值
def sure_shrink(var, coeffs):
    N = len(coeffs)
    sqr_coeffs = []
    for coeff in coeffs:
        sqr_coeffs.append(math.pow(coeff, 2))
    sqr_coeffs.sort()
    pos = 0
    r = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
        if r == 0 or r > new_r:
            r = new_r
            pos = idx
    thre = math.sqrt(var) * math.sqrt(sqr_coeffs[pos])
    return thre


# 求VisuShrink法阈值
def visu_shrink(var, coeffs):
    N = len(coeffs)
    thre = math.sqrt(var) * math.sqrt(2 * math.log(N))
    return thre


# 求HeurSure法阈值
def heur_sure(var, coeffs):
    N = len(coeffs)
    s = 0
    for coeff in coeffs:
        s += math.pow(coeff, 2)
    theta = (s - N) / N
    miu = math.pow(math.log2(N), 3/2) / math.pow(N, 1/2)
    if theta < miu:
        return visu_shrink(var, coeffs)
    else:
        return min(visu_shrink(var, coeffs), sure_shrink(var, coeffs))


# 求Minimax法阈值
def mini_max(var, coeffs):
    N = len(coeffs)
    if N > 32:
        return math.sqrt(var) * (0.3936 + 0.1829 * math.log2(N))
    else:
        return 0


# 平移操作
def right_shift(data, n):
    copy1 = list(data[n:])
    copy2 = list(data[:n])
    return copy1 + copy2


# 逆平移操作
def back_shift(data, n):
    p = len(data) - n
    copy1 = list(data[p:])
    copy2 = list(data[:p])
    return copy1 + copy2


# 获取噪声方差
def get_var(cD):
    coeffs = cD
    abs_coeffs = []
    for coeff in coeffs:
        abs_coeffs.append(math.fabs(coeff))
    abs_coeffs.sort()
    pos = math.ceil(len(abs_coeffs) / 2)
    var = abs_coeffs[pos] / 0.6745
    return var

# 获取近似基线
def get_baseline(data, wavelets_name='sym8', level=5):
    '''
    :param data: signal
    :param wavelets_name: wavelets name in PyWavelets, 'sym8' as default
    :param level: deconstruct level, 5 as default
    :return: baseline signal
    '''
    # 创建小波对象
    wave = pywt.Wavelet(wavelets_name)
    # 分解
    coeffs = pywt.wavedec(data, wave, level=level)
    # 除最高频外小波系数置零
    for i in range(1, len(coeffs)):
        coeffs[i] *= 0
    # 重构
    baseline = pywt.waverec(coeffs, wave)
    if len(baseline)!=len(data):
        baseline = baseline[:min(len(baseline),len(data))]
    return baseline


# 阈值收缩去噪法
def tsd(data, method='sureshrink', mode='soft', wavelets_name='sym8', level=5):
    '''
    :param data: signal
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'sureshrink' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'sym8' as default
    :param level: deconstruct level, 5 as default
    :return: processed data
    '''
    methods_dict = {'visushrink': visu_shrink, 'sureshrink': sure_shrink, 'heursure': heur_sure, 'minmax': mini_max}
    # 创建小波对象
    wave = pywt.Wavelet(wavelets_name)

    # 分解 阈值处理
    data_ = data[:]

    (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    var = get_var(cD)

    coeffs = pywt.wavedec(data=data, wavelet=wavelets_name, level=level)

    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            continue
        # 求阈值thre
        thre = methods_dict[method](var, coeff)
        # 处理cD
        coeffs[idx] = pywt.threshold(coeffs[idx], thre, mode=mode)

    # 重构信号
    thresholded_data = pywt.waverec(coeffs, wavelet=wavelets_name)
    if len(thresholded_data)!=len(data):
        thresholded_data = thresholded_data[:min(len(thresholded_data),len(data))]
    return thresholded_data


# 小波平移不变消噪
def ti(data, step=100, method='heursure', mode='soft', wavelets_name='sym5', level=5):
    '''

    :param data: signal
    :param step: shift step, 100 as default
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'heursure' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'sym5' as default
    :param level: deconstruct level, 5 as default
    :return: processed data
    '''
    # 循环平移
    num = math.ceil(len(data)/step)
    final_data = [0]*len(data)
    for i in range(num):
        temp_data = right_shift(data, i*step)
        temp_data = tsd(temp_data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        temp_data = temp_data.tolist()
        temp_data = back_shift(temp_data, i*step)
        final_data = list(map(lambda x, y: x+y, final_data, temp_data))

    final_data = list(map(lambda x: x/num, final_data))
    if len(final_data)!=len(data):
        final_data = final_data[:min(len(final_data),len(data))]
    return final_data


#Data path
path ='D:\\正常细胞与癌细胞分类\\光谱法\\实验数据\\FTIR\\FTIR总数据\\2B\\X-2B-40.CSV'

#Extract data
Raw_data = pd.read_csv(path,header=None)
data = Raw_data.iloc[:, 1]

#wavelet deal
datarec = get_baseline(data)
# #sg
# datarec = savgol_filter(data, window_length=51, polyorder=3)


#drawing
save_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//测试结果//'

plt.figure(dpi=600)
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False  
plt.subplot(3,1,1)
plt.plot(Raw_data.iloc[:,0], data)
plt.xlabel("Wavenumber(cm-1)", fontsize=12,weight="bold")
plt.ylabel("Absorbance(a.u)", fontsize=12,weight="bold")
plt.xticks(fontsize=11,weight="bold")
plt.yticks(fontsize=11,weight="bold")
plt.title("Raw signal", fontsize=13,weight="bold")
plt.subplot(3, 1, 2)
plt.plot(Raw_data.iloc[:,0], datarec)
plt.xlabel("Wavenumber(cm-1)", fontsize=12,weight="bold")
plt.ylabel("Absorbance(a.u)", fontsize=12,weight="bold")
plt.xticks(fontsize=11,weight="bold")
plt.yticks(fontsize=11,weight="bold")
plt.title("De-noised signal using Wavelet techniques", fontsize=13,weight="bold")
plt.subplot(3, 1, 3)
plt.plot(Raw_data.iloc[:,0],data-datarec)
plt.xlabel("Wavenumber(cm-1)", fontsize=12,weight="bold")
plt.ylabel('Error(a.u)', fontsize=12,weight="bold")
plt.xticks(fontsize=11,weight="bold")
plt.yticks(fontsize=11,weight="bold")
plt.tight_layout()
plt.savefig(save_dir + 'wavelet_deal' +'.jpg')
plt.show()