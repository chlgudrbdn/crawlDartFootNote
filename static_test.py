import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
from pandas.api.types import is_numeric_dtype

np.random.seed(42)


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh
    # 출처: https://pythonanalysis.tistory.com/7 [Python 데이터 분석]


def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

    return result


def normal_dist_test(df):
    for col in list(df.columns):
        # col = df.columns[0]
        print(col)
        x1 = df.loc[df[col].notnull(), col]
        print(x1.shape)
        x1 = x1[x1 != 0.0]
        print(x1.shape)
        x1 = removeOutliers(x1, 1.5)
        print(x1.shape)
        sns.distplot(x1, rug=True, label=col)

        # N = x1.shape[0]
        # x2 = sp.stats.norm(0, 1).rvs(N)
        # sns.distplot(x2, rug=True, label='norm')

        plt.legend(title="compare dist")

        plt.figure(figsize=(12, 6))
        plt.title(col)
        plt.show()

        # plt.hist(x1, 12)

        # print(sp.stats.ks_2samp(x1, x2).pvalue < 0.05)  # Kolmogorov-Smirnov 검정 0.05, 이하라면 둘은 다른 분포.
        print(sp.stats.shapiro(x1)[1] < 0.05)  # shapiro wilks 검정 0.05, 이하라면 정규분포가 아니다.
