# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pandas import read_csv
from os import listdir
#from sklearn import tree
#from sklearn import cluster
#from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
#import graphviz 
import pandas as pd
#import random
#import xgboost as xgb
#from scipy.stats.mstats import zscore
#from xgboost import XGBRegressor
#from xgboost import plot_tree
import os

#os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/graphviz-2.38/release/bin/'

#import math
#from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm


composite_cycle = None
training_cycle = listdir('FCEV_csv')
randomindex = []
composite_cycle = None


for cyc in training_cycle:
    if '.csv' in cyc:
        print("process cycle name " + cyc)
        cycle_to_process = read_csv('FCEV_csv/' + cyc, sep = ',')
        cycle_to_process = cycle_to_process[cycle_to_process['Acceleration(mph/s)'].abs() < 10]
        cycle_to_process['fuel_rate(J)'] = cycle_to_process['fuelrate(kg/s)'] * 142000000.0 * 0.1
    #    cutpoint_cycle = cycle_to_process['tracpower(watt)'].quantile(0.9)
    #    cycle_to_process = cycle_to_process[cycle_to_process['tracpower(watt)'].abs() <= cutpoint_cycle]
    
        if cycle_to_process.size > 0:
            if composite_cycle is None:
                composite_cycle = cycle_to_process
            else:
                composite_cycle = pd.concat([composite_cycle, cycle_to_process])

quartile_1, quartile_3 = composite_cycle['tracpower(watt)'].quantile(0.25), composite_cycle['tracpower(watt)'].quantile(0.75)
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)

composite_cycle = composite_cycle[(composite_cycle['tracpower(watt)'] >= lower_bound) & (composite_cycle['tracpower(watt)'] <= upper_bound)]

#quartile_1, quartile_3 = composite_cycle['fuelrate(kg/s)'].quantile(0.25), composite_cycle['fuelrate(kg/s)'].quantile(0.75)
#iqr = quartile_3 - quartile_1
#lower_bound = quartile_1 - (iqr * 1.5)
#upper_bound = quartile_3 + (iqr * 1.5)
#composite_cycle = composite_cycle[(composite_cycle['fuelrate(kg/s)'] >= lower_bound) & (composite_cycle['fuelrate(kg/s)'] <= upper_bound)]

#composite_cycle = composite_cycle.loc[~composite_cycle['elec_energy(J)'].isnull()]
composite_cycle = composite_cycle.loc[composite_cycle['mot_command'] != 0]
composite_cycle = composite_cycle.loc[composite_cycle['fc_on'] != 0]

# <codecell>
plt.style.use('ggplot')
sample_cycle = composite_cycle.sample(n=30000)
sample_cycle = sample_cycle[['time(s)', 'road_type', 'SOC', 'Speed(mph)','Acceleration(mph/s)', 'road_grade(rad)', 'VSP', 'torque_demand(Nm)', 
                             'fc_on', 'fc_temp_coeff', 'fc_command_pwr(watt)', 'ess_current', 'ess_volt', 'mot_torque(Nm)', 'mot_speed(rad/s)', 'fuelrate(kg/s)', 'elec_energy(J)']]
sample_cycle.loc[:, 'SOC_high'] = 1 * (sample_cycle.loc[:, 'SOC'] - 0.6 >= 0) + 0 * (sample_cycle.loc[:, 'SOC'] - 0.6 < 0)

to_plot = composite_cycle[['SOC', 'Speed(mph)', 'Acceleration(mph/s)','VSP', 'fc_on', 'fuel_rate(J)',  'elec_energy(J)']].sample(n=10000)
to_plot['elec_energy(J)'] *= 10
to_plot['fuel_rate(J)'] *= 10

to_plot.columns = ['SOC', 'Speed (mph)', 'Acceleration (mph/s)','VSP', 'FC on','fuel rate (Watt)', 'Elec. rate (Watt)']
sns.set(font_scale=1)
sns.pairplot(to_plot[['SOC', 'Speed (mph)', 'Acceleration (mph/s)','VSP', 'FC on','fuel rate (Watt)', 'Elec. rate (Watt)']],  kind='reg', plot_kws={'line_kws':{'color':'navy'}, 'scatter_kws': {'alpha': 0.1}})
plt.savefig('FCEV_variable_pair_plot.jpg', bbox_inches = 'tight', dpi=500)

#corr = sample_cycle.corr(method = 'spearman')
#fig, ax = plt.subplots(figsize=(10, 10))
#cax = ax.matshow(corr, cmap = 'coolwarm')
#cbar = fig.colorbar(cax)
#cbar.set_clim(-1.0, 1.0)
#plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
#plt.yticks(range(len(corr.columns)), corr.columns)
#plt.savefig('FCEV_Corr_matrix_noparam.jpg', bbox_inches = 'tight', dpi=500)
#plt.show()
#plt.ylim([-0.001, 0.002])
#g1= sns.jointplot(sample_cycle['VSP'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})
g1= plt.scatter(sample_cycle['VSP'], sample_cycle['elec_energy(J)']*10, c=sample_cycle['SOC'], cmap='jet_r', alpha=0.2)
cbar = plt.colorbar(g1)
plt.xlabel('VSP (watt/tonne)')
plt.ylabel("Electricity rate (Watt)")
plt.ylim([-40000, 30000])
cbar.ax.get_yaxis().labelpad = 15
cbar.set_label('SOC', rotation=270)
#plt.ylim([0, 0.016])
plt.savefig('FCEV_ELEC_VS_SOC_VSP.jpg', bbox_inches = 'tight', dpi = 200)
plt.show()
#g2 = sns.lmplot(x='VSP', y='fuelrate(kg/s)', col='SOC_high', data=sample_cycle, scatter_kws={"alpha":.05})
#g2.set(ylim=(0, 0.0016))
#sns.jointplot(sample_cycle['Speed(mph)'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})

# <codecell>
#sample_cycle_lowvsp = sample_cycle.loc[(sample_cycle['VSP'] <= 10000) & (sample_cycle['VSP'] > 0)  & (sample_cycle['SOC'] > 0.6)]
##sns.pairplot(sample_cycle_lowvsp[['SOC', 'Speed(mph)','Acceleration(mph/s)', 'road_grade(rad)', 'fc_temp_coeff', 'VSP', 'fuelrate(kg/s)']], kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
#lowvsp_features = sample_cycle_lowvsp.describe()
#total_features = composite_cycle.describe()
#
#plt.scatter(sample_cycle_lowvsp['VSP'], sample_cycle_lowvsp['fuelrate(kg/s)'], c=sample_cycle_lowvsp['elec_energy(J)'], cmap='RdBu',  alpha=0.3)
#plt.ylim([0, 0.0016])
#plt.show()

#sns.jointplot(sample_cycle['VSP'], sample_cycle['elec_energy(J)'], joint_kws={"alpha":.05})
#sns.jointplot(sample_cycle['SOC'], sample_cycle['elec_energy(J)'], joint_kws={"alpha":.05})
#sns.jointplot(sample_cycle['fc_temp_coeff'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})

# <codecell>

### linear regression and cross validation
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

kf = KFold(n_splits=10)
#lm = linear_model.LinearRegression()

x = composite_cycle[['SOC', 'VSP', 'fuel_rate(J)']]
#x.loc[:, 'VSP_Square'] = x.loc[:, 'VSP'] ** 2
y = composite_cycle['fuel_rate(J)'] * 10
y2 = composite_cycle['elec_energy(J)'] * 10
#x_scaled = preprocessing.scale(x)
#y_scaled = preprocessing.scale(y)
##sample_y.loc[:, 'elec_energy(J)'] *= 10
#clf = linear_model.LassoCV()
#
## Set a minimum threshold of 0.25
#sfm = SelectFromModel(clf, threshold=1e-5)
#sfm.fit(x_scaled, y_scaled)
#x_transform = sfm.transform(x_scaled)
#n_features = sfm.transform(x_scaled).shape[1]


# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
#while n_features > 3:
#    sfm.threshold += 0.1
#    
#    n_features = x_transform.shape[1]
    

# <codecell>
est = sm.OLS(y, x)
est2 = est.fit()
print(est2.summary())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#OLS_accuracy = []
piecewise_accuracy = []
for train_index, test_index in kf.split(X_train):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
    lm = linear_model.LinearRegression()
#    model = lm.fit(X_train, y_train)
#    predictions = lm.predict(X_test)
#    accuracy = metrics.r2_score(y_test, predictions)
#    OLS_accuracy.append(accuracy)
#    print("OLS Cross-Predicted Accuracy:", accuracy)
    
    X_train_pos = X_train_cv.loc[X_train_cv['VSP'] >= 0]
    X_train_neg = X_train_cv.loc[X_train_cv['VSP'] < 0]

    y_train_pos = y_train_cv.loc[X_train_cv['VSP'] >= 0]
    y_train_neg = y_train_cv.loc[X_train_cv['VSP'] < 0]

    results_pos = lm.fit(X_train_pos, y_train_pos)
#    a1 = results_pos.coef_
#    b1 = results_pos.intercept_
    X_test_pos = X_test_cv.loc[X_test_cv['VSP'] >= 0]
    y_test_pos = y_test_cv.loc[X_test_cv['VSP'] >= 0]
    y_pred_pos = results_pos.predict(X_test_pos)
    
#    results_neg = lm.fit(X_train_neg, y_train_neg)
#    a2 = results_neg.coef_
#    b2 = results_neg.intercept_   
    X_test_neg = X_test_cv.loc[X_test_cv['VSP'] < 0]   
    y_test_neg = y_test_cv.loc[X_test_cv['VSP'] < 0] 
    y_pred_neg = y_test_neg
    y_pred_neg.loc[:] = y_train_neg.mean()
    y_pred_cv =  np.concatenate((y_pred_pos, y_pred_neg))
    y_test_cv = pd.concat([y_test_pos, y_test_neg])
    piecewise_mse = metrics.mean_squared_error(y_test_cv, y_pred_cv)
    piecewise_accuracy.append(piecewise_mse)
    print("piecewise Cross-Predicted Accuracy:", piecewise_mse)
#print(np.mean(OLS_accuracy))
print(np.mean(piecewise_accuracy))
#    break
X_train_pos = X_train.loc[X_train['VSP'] >= 0]
X_train_neg = X_train.loc[X_train['VSP'] < 0]

y_train_pos = y_train.loc[X_train['VSP'] >= 0]
y_train_neg = y_train.loc[X_train['VSP'] < 0]

results_pos = lm.fit(X_train_pos, y_train_pos)
a1 = results_pos.coef_
b1 = results_pos.intercept_
X_test_pos = X_test.loc[X_test['VSP'] >= 0]
y_test_pos = y_test.loc[X_test['VSP'] >= 0]
y_pred_pos = results_pos.predict(X_test_pos)

results_neg = lm.fit(X_train_neg, y_train_neg)
a2 = results_neg.coef_
b2 = results_neg.intercept_   
X_test_neg = X_test.loc[X_test['VSP'] < 0]   
y_test_neg = y_test.loc[X_test['VSP'] < 0] 
y_pred_neg = y_test_neg
y_pred_neg.loc[:] = y_train_neg.mean()
y_pred =  np.concatenate((y_pred_pos, y_pred_neg))
y_test = pd.concat([y_test_pos, y_test_neg])
piecewise_r2 = metrics.r2_score(y_test, y_pred)
piecewise_mse = metrics.mean_squared_error(y_test, y_pred)
print(piecewise_r2)
print(np.sqrt(piecewise_mse))

# <codecell>
X_train, X_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.2, random_state=42)

#OLS_accuracy = []
piecewise_accuracy = []
for train_index, test_index in kf.split(X_train):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y2_train_cv, y2_test_cv = y2_train.iloc[train_index], y2_train.iloc[test_index]
    lm = linear_model.LinearRegression()
    
    X_train_pos = X_train_cv.loc[X_train_cv['VSP'] >= 0]
    X_train_neg = X_train_cv.loc[X_train_cv['VSP'] < 0]

    y2_train_pos = y2_train_cv.loc[X_train_cv['VSP'] >= 0]
    y2_train_neg = y2_train_cv.loc[X_train_cv['VSP'] < 0]

    results_pos = lm.fit(X_train_pos, y2_train_pos)
#    a1 = results_pos.coef_
#    b1 = results_pos.intercept_
    X_test_pos = X_test_cv.loc[X_test_cv['VSP'] >= 0]
    y2_test_pos = y2_test_cv.loc[X_test_cv['VSP'] >= 0]
    y2_pred_pos = results_pos.predict(X_test_pos)
    
#    results_neg = lm.fit(X_train_neg, y_train_neg)
#    a2 = results_neg.coef_
#    b2 = results_neg.intercept_   
    X_test_neg = X_test_cv.loc[X_test_cv['VSP'] < 0]   
    y2_test_neg = y2_test_cv.loc[X_test_cv['VSP'] < 0] 
    result_neg = lm.fit(X_train_neg, y2_train_neg)
    y2_pred_neg = results_pos.predict(X_test_neg)
    y2_pred_cv =  np.concatenate((y2_pred_pos, y2_pred_neg))
    y2_test_cv = pd.concat([y2_test_pos, y2_test_neg])
    piecewise_mse = metrics.mean_squared_error(y2_test_cv, y2_pred_cv)
    piecewise_accuracy.append(piecewise_mse)
    print("piecewise Cross-Predicted Accuracy:", piecewise_mse)
#print(np.mean(OLS_accuracy))
print(np.mean(piecewise_accuracy))
#    break
X_train_pos = X_train.loc[X_train['VSP'] >= 0]
X_train_neg = X_train.loc[X_train['VSP'] < 0]

y2_train_pos = y2_train.loc[X_train['VSP'] >= 0]
y2_train_neg = y2_train.loc[X_train['VSP'] < 0]

results_pos = lm.fit(X_train_pos, y2_train_pos)
a1 = results_pos.coef_
b1 = results_pos.intercept_
X_test_pos = X_test.loc[X_test['VSP'] >= 0]
y2_test_pos = y2_test.loc[X_test['VSP'] >= 0]
y2_pred_pos = results_pos.predict(X_test_pos)

results_neg = lm.fit(X_train_neg, y2_train_neg)
a2 = results_neg.coef_
b2 = results_neg.intercept_   
X_test_neg = X_test.loc[X_test['VSP'] < 0]   
y2_test_neg = y2_test.loc[X_test['VSP'] < 0] 
y2_pred_neg = results_neg.predict(X_test_neg)
y2_pred =  np.concatenate((y2_pred_pos, y2_pred_neg))
y2_test = pd.concat([y2_test_pos, y2_test_neg])
piecewise_r2 = metrics.r2_score(y2_test, y2_pred)
piecewise_mse = metrics.mean_squared_error(y2_test, y_pred)
print(piecewise_r2)
print(np.sqrt(piecewise_mse))


# <codecell>

comb_piecewise_r2 = metrics.r2_score(y_test+y2_test, y_pred+y2_pred)
comb_piecewise_mse = metrics.mean_squared_error(y_test+y2_test, y_pred+y2_pred)
print(comb_piecewise_r2)
print(np.sqrt(comb_piecewise_mse))


