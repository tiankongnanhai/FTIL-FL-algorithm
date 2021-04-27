from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import pywt
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from FL_show import getdata


def PlotSpectrum(wave,spec):
    """
    :param spec: shape (n_samples, n_features)
    :return: plt
    """
    plt.figure()
    fonts = 8
    # plt.xlim(350, 2550)
    # plt.ylim(0, 1)
    for i in range(spec.shape[0]):
        plt.plot(wave, spec.iloc[i], linewidth=0.6)
    plt.xlabel('Wavelength (nm)', fontsize=fonts)
    plt.ylabel('Absorbance (AU)', fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.xticks(fontsize=fonts)
    plt.tight_layout(pad=1.5)
    plt.grid(True)
    plt.show()


def mean_centralization(sdata):
    """
    Mean centralization
    """
    temp1 = np.mean(sdata, axis=0)# axis=0, calculate the mean of each column
    temp2 = np.tile(temp1, sdata.shape[0]).reshape((sdata.shape[0], sdata.shape[1]))# Copy sample rows
    return sdata - temp2


def standardlize(sdata):
    """
    standardization
    """
    st_data = preprocessing.scale(sdata) 
    st_data = pd.DataFrame(st_data,index=sdata._stat_axis.values.tolist())
    return st_data


def sg(sdata,window_size=51,polynomial_order=3):
    '''
    Input raw data,window_size,polynomial_order
    Output smoothed data
    window must be odd and smaller than polynomial order
    '''
    for i in range(sdata.shape[0]):
        sdata.iloc[i]=savgol_filter(sdata.iloc[i], window_size, polynomial_order)
    return sdata


def msc(sdata):
    '''
    Multi-element scatter correction processing
    '''
    np_sdata = np.array(sdata)
    n = np_sdata.shape[0]  # Number of samples
    k = np.zeros(np_sdata.shape[0])
    b = np.zeros(np_sdata.shape[0])

    M = np.mean(np_sdata, axis=0)#The average of all spectra is regarded as the ideal spectrum

    #Calculate the linear regression equation between the spectrum of each sample and the "ideal spectrum"
    for i in range(n):
        y = np_sdata[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_
 
    spec_msc = np.zeros_like(np_sdata)
    for i in range(n):
        bb = np.repeat(b[i], np_sdata.shape[1])
        kk = np.repeat(k[i], np_sdata.shape[1])
        temp = (np_sdata[i, :] - bb)/kk
        spec_msc[i, :] = temp
    
    spec_msc = pd.DataFrame(spec_msc,index=sdata._stat_axis.values.tolist())
    return spec_msc


def snv(sdata):
    """
    Standard normal variable transformation
    """
    temp1 = np.mean(sdata, axis=1)
    temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
    temp3 = np.std(sdata, axis=1)
    temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
    return (sdata - temp2)/temp4


def D1(sdata):
    """
    First difference
    """
    temp1 = pd.DataFrame(sdata)
    temp2 = temp1.diff(axis=1)
    temp3 = temp2.values
    # temp4 = np.delete(temp3, 0, axis=1) #The first point remains unchanged
    D1_msc = pd.DataFrame(temp3,index=sdata._stat_axis.values.tolist())
    D1_msc = D1_msc.fillna(0)# Set the vacancy value to zero
    return D1_msc


def D2(sdata):
    """
    Second order difference
    """
    temp2 = (pd.DataFrame(sdata)).diff(axis=1)  
    # temp3 = np.delete(temp2.values, 0, axis=1) #The first point remains unchanged
    temp4 = (pd.DataFrame(temp2)).diff(axis=1)  
    # spec_D2 = np.delete(temp4.values, 0, axis=1) #The first point remains unchanged
    D2_spec = pd.DataFrame(temp4,index=sdata._stat_axis.values.tolist())
    D2_spec = D2_spec.fillna(0)# Set the vacancy value to zero
    return D2_spec


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
def get_baseline(absorb, wavelets_name='sym8', level=5):
    '''
    :param data: signal
    :param wavelets_name: wavelets name in PyWavelets, 'sym8' as default
    :param level: deconstruct level, 5 as default
    :return: baseline signal
    '''
    # 创建空白narray
    baseline_absorb = np.zeros((absorb.shape[0],absorb.shape[1]))
    for i in range(absorb.shape[0]):
        data = absorb.iloc[i,:]
        # 创建小波对象
        wave = pywt.Wavelet(wavelets_name)
        # 分解
        coeffs = pywt.wavedec(data, wave, level=level)
        # 除最高频外小波系数置零
        for k in range(1, len(coeffs)):
            coeffs[k] *= 0
        # 重构
        baseline = pywt.waverec(coeffs, wave)
        if len(baseline)!=len(data):
            baseline = baseline[:min(len(baseline),len(data))]
        
        baseline_absorb[i,:] = baseline

    baseline_absorb = pd.DataFrame(baseline_absorb,index=absorb._stat_axis.values.tolist())
    return baseline_absorb

def get_baseline_single(data, wavelets_name='sym8', level=5):
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
    for k in range(1, len(coeffs)):
        coeffs[k] *= 0
    # 重构
    baseline = pywt.waverec(coeffs, wave)
    if len(baseline)!=len(data):
        baseline = baseline[:min(len(baseline),len(data))]

    return baseline

# 阈值收缩去噪法
def tsd(absorb, method='sureshrink', mode='soft', wavelets_name='sym8', level=5):
    '''
    :param data: signal
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'sureshrink' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'sym8' as default
    :param level: deconstruct level, 5 as default
    :return: processed data
    '''
    tsd_absorb = np.zeros((absorb.shape[0],absorb.shape[1]))
    for i in range(absorb.shape[0]):
        data = absorb.iloc[i,:]
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
        tsd_absorb[i,:] = thresholded_data
    
    tsd_absorb = pd.DataFrame(tsd_absorb,index=absorb._stat_axis.values.tolist())
    return tsd_absorb

def tsd_single(data, method='sureshrink', mode='soft', wavelets_name='sym8', level=5):
    '''
    对于单个样本进行消噪
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
def ti(absorb, step=100, method='heursure', mode='soft', wavelets_name='sym5', level=5):
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
    ti_absorb = np.zeros((absorb.shape[0],absorb.shape[1]))
    for i in range(absorb.shape[0]):
        data = absorb.iloc[i,:]
        num = math.ceil(len(data)/step)
        final_data = [0]*len(data)
        for j in range(num):
            temp_data = right_shift(data, j*step)
            temp_data = tsd_1(temp_data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
            temp_data = temp_data.tolist()
            temp_data = back_shift(temp_data, j*step)
            final_data = list(map(lambda x, y: x+y, final_data, temp_data))

        final_data = list(map(lambda x: x/num, final_data))
        if len(final_data)!=len(data):
            final_data = final_data[:min(len(final_data),len(data))]
        ti_absorb[i,:] = final_data
    
    ti_absorb = pd.DataFrame(ti_absorb,index=absorb._stat_axis.values.tolist())
    return ti_absorb



# ######Read in data
# DATA_2B_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//2B//'
# DATA_A549_dir = 'D://正常细胞与癌细胞分类//光谱法//实验数据//FTIR//FTIR总数据//A549//'

# DATA_2B = getdata(DATA_2B_dir)
# DATA_A549 = getdata(DATA_A549_dir)

# #Merge into a data set
# FTIR_DATA = pd.merge(DATA_2B,DATA_A549,on='wave')
# FTIR_DATA = FTIR_DATA.T

# data = mean_centralization(FTIR_DATA.iloc[1:])
# PlotSpectrum(FTIR_DATA.iloc[0],data)