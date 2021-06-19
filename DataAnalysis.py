# -*- coding: utf-8 -*

%matplotlib inline
import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

from util.feature import add_feature, fillna
from util.metric import mse
from util import variables

sns.set(style='white')
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   #用来正常显示负号


#---------------------加载数据-------------------------
train_data = pd.read_csv('d_train_20180102.csv', encoding='gb18030')
test_data = pd.read_csv('d_test_A_20180102.csv', encoding='gb18030')
test_data['血糖'] = -1
data = pd.concat([train_data, test_data], ignore_index=True)


# import pandas as pd
#
# file = 'd_train_20180102.csv'
# data = pd.read_csv(file, encoding='gb18030')
# data.info()


#---------------------打印各字段信息-------------------------
train_data.info()


#---------------------统计空值情况-------------------------
na_data = train_data.isna().sum() / train_data.shape[0]
fig_na, ax_na = plt.subplots(figsize=(15,9))
na_data.plot.barh(ax=ax_na, color='g')


#---------------------相关性分析-------------------------
train_data.describe().to_csv('descibe.csv')
corr = train_data.loc[:, '年龄':].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig_corr, ax_corr = plt.subplots(figsize=(15, 13))
cmap = sns.diverging_palette(100, 3, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=False)



#---------------------数据分布情况-------------------------
fig_boxt, ax_boxt = plt.subplots(figsize=(15, 9))
XTrain = train_data.loc[:, [column for column in train_data.columns if column not in
                        ['id', '年龄', '体检日期', '乙肝表面抗原',
                         '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体',
                         'dayofyear']]]
sns.violinplot(data=XTrain, orient="h", ax=ax_boxt)



#---------------------男女生血糖分布情况-------------------------
from scipy.special import boxcox1p
fig_gluhist, ax_gluhist = plt.subplots(figsize=(15, 9))
n, bins, pathes = ax_gluhist.hist([np.log1p(train_data.loc[train_data['性别'] == '男', '血糖']),
                                   np.log1p(train_data.loc[train_data['性别'] == '女', '血糖'])],
                                   bins=20, label=['男','女'])
plt.legend()



#---------------------特征分布散点图-------------------------
origin_feature = [column for column in train_data.columns if column not in ['id', '性别', '血糖', '体检日期', '乙肝表面抗原',
                         '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']]
print(len(origin_feature))
fig_f2o, ax_f2o = plt.subplots(7, 5, figsize=(20, 15))
for idx, col in enumerate(origin_feature):
    ax_f2o[idx//5, idx%5].plot(train_data[col], train_data['血糖'], 'g.')
    ax_f2o[idx//5, idx%5].set_title(col)
fig_f2o.tight_layout()



#---------------------特征重要性分析-------------------------
import xgboost as xgb
from dateutil.parser import parse
y_train = train_data['血糖']
X_train = train_data.drop(['血糖', 'id', '性别', '体检日期'], axis=1)
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=60, height=0.8, ax=ax, color='green')
plt.show()





