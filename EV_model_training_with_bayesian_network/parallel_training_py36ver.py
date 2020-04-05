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



#import math
#from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm


composite_cycle = None
training_cycle = listdir('Fusion_HEV_csv')
randomindex = []
composite_cycle = None


for cyc in training_cycle:
    print("process cycle name " + cyc)
    cycle_to_process = read_csv('Fusion_HEV_csv/' + cyc, sep = ',')
    cycle_to_process = cycle_to_process[cycle_to_process['Acceleration(mph/s)'].abs() < 10]
    cycle_to_process['fuel_rate(J)'] = cycle_to_process['fuelrate(kg/s)'] * 46400000.0 * 0.1
    cycle_to_process['rolling_VSP'] = cycle_to_process['VSP'].rolling(window=5).mean()
    cycle_to_process.head(4)['rolling_VSP'] = float(cycle_to_process.iloc[4]['rolling_VSP'])
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

#composite_cycle = composite_cycle.loc[~composite_cycle['elec_energy(J)'].isnull()]
#composite_cycle = composite_cycle.loc[(composite_cycle['mot_command'] > 0) | (composite_cycle['eng_on'] != 0) | (composite_cycle['VSP'] <= 0)]

composite_cycle = composite_cycle.loc[(composite_cycle['fuel_rate(J)'] + composite_cycle['elec_energy(J)'] > 0.1 * composite_cycle['tracpower(watt)'])]
#composite_cycle = composite_cycle.loc[composite_cycle['fc_on'] != 0]
composite_cycle.loc[:, 'eng_on'] = 1 * ((composite_cycle.loc[:, 'eng_on'] == 1) | (composite_cycle.loc[:, 'fuelrate(kg/s)'] > 0.0)) +  0 * ((composite_cycle.loc[:, 'eng_on'] == 0) | (composite_cycle.loc[:, 'fuelrate(kg/s)'] == 0.0))
#composite_cycle.loc[:, 'mot_command'] = 1 * ((composite_cycle.loc[:, 'mot_command'] == 1) | (composite_cycle.loc[:, 'mot_speed(rad/s)'] > 0.0)) +  0 * ((composite_cycle.loc[:, 'eng_on'] == 0) | (composite_cycle.loc[:, 'fuelrate(kg/s)'] == 0.0))
composite_cycle['control'] = 1 * ((composite_cycle.loc[:, 'eng_on'] == 1) & (composite_cycle.loc[:, 'mot_command'] >= 0)) + \
2 * ((composite_cycle.loc[:, 'eng_on'] == 1) & (composite_cycle.loc[:, 'mot_command'] < 0)) + \
3 * ((composite_cycle.loc[:, 'eng_on'] == 0) & (composite_cycle.loc[:, 'mot_command'] >= 0)) + \
4 * ((composite_cycle.loc[:, 'eng_on'] == 0) & (composite_cycle.loc[:, 'mot_command'] < 0))

# <codecell>

control_type_dict = {1: 'eng_on, mot1_pos', 
                     2: 'eng_on, mot1_neg', 
                     3: 'eng_off, mot1_pos', 
                     4: 'eng_off, mot1_neg'} 

        
composite_cycle["control_type"] = composite_cycle["control"].map(control_type_dict)
plt.style.use('ggplot')

sample_cycle = composite_cycle.sample(n=100000)
#sample_cycle = sample_cycle[['time(s)', 'road_type', 'SOC', 'Speed(mph)','Acceleration(mph/s)', 'road_grade(rad)', 'VSP', 'torque_demand(Nm)', 'mot_command', 
#                             'eng_on', 'ess_current', 'ess_volt', 'mot_torque(Nm)', 'mot_speed(rad/s)', 'fuel_rate(J)', 'elec_energy(J)']]
##sample_cycle.loc[:, 'SOC_high'] = 1 * (sample_cycle.loc[:, 'SOC'] - 0.3 >= 0) + 0 * (sample_cycle.loc[:, 'SOC'] - 0.3 < 0)
#corr = sample_cycle.corr(method = 'pearson')
#fig, ax = plt.subplots(figsize=(10, 10))
#cax = ax.matshow(corr, cmap = 'coolwarm')
#cbar = fig.colorbar(cax)
#cbar.set_clim(-1.0, 1.0)
#plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
#plt.yticks(range(len(corr.columns)), corr.columns)
#plt.savefig('Fusion_Corr_matrix.jpg', bbox_inches = 'tight', dpi=500)
#plt.show()
#plt.ylim([-0.001, 0.002])
#g1= sns.jointplot(sample_cycle['VSP'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})
#

#g1= plt.scatter(sample_cycle.loc[sample_cycle["control"].isin([1,2]), 'VSP'], sample_cycle.loc[sample_cycle["control"].isin([1,2]), 'fuel_rate(J)']*10, c=sample_cycle.loc[sample_cycle["control"].isin([1,2]),'SOC'], cmap='jet_r', alpha=0.2)
#cbar = plt.colorbar(g1)
##control_id = sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'control'].unique()[0]
#plt.xlabel('VSP (watt/tonne)')
#plt.ylabel("fuel (J/s)")
##plt.title(ctrl)
#cbar.ax.get_yaxis().labelpad = 15
#cbar.set_label('SOC', rotation=270)
##plt.ylim([0, 0.016])
#plt.savefig('PAR_HEV_PLOT/Fusion_HEV_fuel_VS_SOC_VSP_eng_on.jpg', bbox_inches = 'tight', dpi = 200)
#plt.show()
for ctrl in sample_cycle["control_type"].unique():
    g1= plt.scatter(sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'VSP'], sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'fuel_rate(J)']*10, c=sample_cycle.loc[sample_cycle["control_type"] == ctrl,'SOC'], cmap='jet_r', alpha=0.2)
    cbar = plt.colorbar(g1)
    control_id = sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'control'].unique()[0]
    plt.xlabel('VSP (watt/tonne)')
    plt.ylabel("fuel (J/s)")
    plt.title(ctrl)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('SOC', rotation=270)
    #plt.ylim([0, 0.016])
    plt.savefig('PAR_HEV_PLOT/Fusion_HEV_fuel_VS_SOC_VSP_' + str(control_id) + '.jpg', bbox_inches = 'tight', dpi = 200)
    plt.show()
#g2 = sns.lmplot(x='VSP', y='fuelrate(kg/s)', col='SOC_high', data=sample_cycle, scatter_kws={"alpha":.05})
#g2.set(ylim=(0, 0.0016))
#sns.jointplot(sample_cycle['Speed(mph)'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})

# <codecell>

#engine on scenario
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import tree
import graphviz 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import numpy as np


os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/graphviz-2.38/release/bin/'

composite_cycle.loc[:, 'SOC_high'] = 1 * (composite_cycle.loc[:, 'SOC'] - 0.5 >= 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 0.5 < 0)
composite_cycle.loc[:, 'SOC_medium'] = 1 * ((composite_cycle.loc[:, 'SOC'] - 0.45 >= 0) & (composite_cycle.loc[:, 'SOC'] - 0.5 < 0)) + 0 * ((composite_cycle.loc[:, 'SOC'] - 0.5 >= 0) | (composite_cycle.loc[:, 'SOC'] - 0.45 < 0))
composite_cycle.loc[:, 'SOC_low'] = 1 * (composite_cycle.loc[:, 'SOC'] - 0.45 < 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 0.45 >= 0)
composite_cycle.loc[:, 'VSP_pos'] = composite_cycle.loc[:, 'VSP'] * (composite_cycle.loc[:, 'VSP'] >= 0) + 0 * (composite_cycle.loc[:, 'VSP'] < 0)
composite_cycle.loc[:, 'VSP_neg'] = 1 * (composite_cycle.loc[:, 'VSP'] < 0) + 0 * (composite_cycle.loc[:, 'VSP'] > 0)
composite_cycle.loc[:, 'low_speed'] = 1 * (composite_cycle.loc[:, 'Speed(mph)'] - 4 < 0) + 0 * (composite_cycle.loc[:, 'Speed(mph)'] - 4 >= 0)
#composite_cycle.loc[:, 'high_speed'] = 1 * (composite_cycle.loc[:, 'Speed(mph)'] - 3 >= 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 3 < 0)
composite_cycle['mot1_control'] = 1 * (composite_cycle['control'].isin([1, 3])) + 0 * (composite_cycle['control'].isin([2, 4]))

clf = LogisticRegression()
class_tree = tree.DecisionTreeClassifier(max_depth=3)
class_nb = GaussianNB()
class_qda = QuadraticDiscriminantAnalysis()


X = composite_cycle[['SOC_high', 'SOC_medium', 'SOC_low', 'VSP']]
#X = X.fillna(method='ffill')
y = composite_cycle['eng_on']
scores = cross_val_score(clf, X, y, cv=10)
scores_tree = cross_val_score(class_tree, X, y, cv=10)

print(np.mean(scores))
print(np.mean(scores_tree))




composite_cycle_eng_on = composite_cycle.loc[composite_cycle['eng_on'] == 1]
X2 = composite_cycle_eng_on[['SOC', 'Speed(mph)', 'VSP']]
#X2.loc['interac'] = X2['low_speed'] * X2['VSP_neg']
X3 = composite_cycle_eng_on[['SOC', 'Speed(mph)', 'VSP']]
y2 = composite_cycle_eng_on['mot1_control']
scores = cross_val_score(clf, X2, y2, cv=10)
scores_tree = cross_val_score(class_tree, X3, y2, cv=10)

print(np.mean(scores))
print(np.mean(scores_tree))


tree_result = class_tree.fit(X3, y2)
dot_data = tree.export_graphviz(tree_result, out_file=None)  
graph = graphviz.Source(dot_data)  
graph 

# <codecell>

### linear regression and cross validation
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np

#import statsmodels.api as sm
#from sklearn.feature_selection import SelectFromModel
#from sklearn import preprocessing

kf = KFold(n_splits=10)
composite_cycle['mot1_control'] = 1 * (composite_cycle['control'].isin([1, 3])) + 0 * (composite_cycle['control'].isin([2, 4]))

#lm = linear_model.LinearRegression()

#composite_cycle.iloc[0]['rolling_VSP'] = float(composite_cycle.iloc[2]['rolling_VSP'])
#print(composite_cycle['rolling_VSP'].head(5))
X = composite_cycle[['SOC_high', 'SOC_medium', 'SOC_low', 'SOC', 'low_speed', 'VSP_pos', 'Speed(mph)', 'VSP']]
#X = X.fillna(method='ffill')
y = composite_cycle[['eng_on', 'mot1_control', 'fuel_rate(J)', 'elec_energy(J)']]
    
y.loc[:, 'fuel_rate(J)'] *= 10
y.loc[:, 'elec_energy(J)'] *= 10

y.loc[:, 'fuel_rate_pred'] = 0
y.loc[:, 'mot_control_pred'] = 1
y.loc[:, 'elec_rate_pred'] = 0

fuel_accuracy = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#OLS_accuracy = []
piecewise_accuracy = []

#1. fit engine control model
logit = LogisticRegression()
eng_control = logit.fit(X_train[['SOC_high', 'SOC_medium', 'SOC_low', 'VSP']], y_train['eng_on'])
print(logit.coef_, logit.intercept_)
y_train.loc[:, 'eng_on_pred'] = logit.predict(X_train[['SOC_high', 'SOC_medium', 'SOC_low', 'VSP']])
eng_on_training_acc = metrics.accuracy_score(y_train['eng_on'], y_train['eng_on_pred'])
y_test.loc[:, 'eng_on_pred'] = logit.predict(X_test[['SOC_high', 'SOC_medium', 'SOC_low', 'VSP']])
eng_on_testing_acc = metrics.accuracy_score(y_test['eng_on'], y_test['eng_on_pred'])
print(eng_on_training_acc, eng_on_testing_acc)

#2. fit energy model under engine off 

##2.1 Fuel
avg_fuel_rate_eng_off = y_train.loc[y_train['eng_on_pred'] == 0, 'fuel_rate(J)'].mean()
y_train.loc[y_train['eng_on_pred'] == 0, 'fuel_rate_pred'] = avg_fuel_rate_eng_off
y_test.loc[y_test['eng_on_pred'] == 0, 'fuel_rate_pred'] = avg_fuel_rate_eng_off
#
##2.2 Electric
lm = linear_model.LinearRegression()
elec_pos = lm.fit(X_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame(), y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'elec_energy(J)'])
y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame())
y_test.loc[(X_test['VSP']>=0) & (y_test['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_test.loc[(X_test['VSP']>=0) & (y_test['eng_on_pred']==0), 'VSP'].to_frame())
print(lm.coef_, lm.intercept_)
elec_neg = lm.fit(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame(), y_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'elec_energy(J)'])
y_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame())
y_test.loc[(X_test['VSP']<0) & (y_test['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_test.loc[(X_test['VSP']<0) & (y_test['eng_on_pred']==0), 'VSP'].to_frame())
b1_elec, b0_elec = lm.coef_, lm.intercept_
print(lm.coef_, lm.intercept_)

#3. fit energy model under engine on 

#3.1.1 fit motor charging/discharging
class_tree = tree.DecisionTreeClassifier(max_depth=3)
mot_control = class_tree.fit(X_train.loc[y_train['eng_on_pred'] == 1, ['SOC', 'Speed(mph)', 'VSP']], y_train.loc[y_train['eng_on_pred'] == 1, 'mot1_control'])
#print(logit.coef_, logit.intercept_)
y_train.loc[y_train['eng_on_pred'] == 1, 'mot_control_pred'] = class_tree.predict(X_train.loc[y_train['eng_on_pred'] == 1, ['SOC', 'Speed(mph)', 'VSP']])
y_test.loc[y_test['eng_on_pred'] == 1, 'mot_control_pred'] = class_tree.predict(X_test.loc[y_test['eng_on_pred'] == 1, ['SOC', 'Speed(mph)', 'VSP']])
mot_on_training_acc = metrics.accuracy_score(y_train.loc[y_train['eng_on_pred'] == 1, 'mot1_control'], y_train.loc[y_train['eng_on_pred'] == 1, 'mot_control_pred'])
mot_on_testing_acc = metrics.accuracy_score(y_test.loc[y_test['eng_on_pred'] == 1, 'mot1_control'], y_test.loc[y_test['eng_on_pred'] == 1, 'mot_control_pred'])
print(mot_on_training_acc, mot_on_testing_acc)

#3.1.1 fit fuel/elec under discharging
CD_loc = (y_train['mot_control_pred']==1) & (y_train['eng_on_pred']==1)
CD_test_loc = (y_test['mot_control_pred']==1) & (y_test['eng_on_pred']==1)
CD_low_soc_loc = (y_train['mot_control_pred']==1) & (y_train['eng_on_pred']==1) & (X_train['SOC_low'] == 1)
CD_test_low_soc_loc = (y_test['mot_control_pred']==1) & (y_test['eng_on_pred']==1)  & (X_test['SOC_low'] == 1)
CD_high_soc_loc = (y_train['mot_control_pred']==1) & (y_train['eng_on_pred']==1) & (X_train['SOC_low'] == 0)
CD_test_high_soc_loc = (y_test['mot_control_pred']==1) & (y_test['eng_on_pred']==1) & (X_test['SOC_low'] == 0)

CD_fuel = lm.fit(X_train.loc[CD_loc, ['SOC','VSP']], y_train.loc[CD_loc, 'fuel_rate(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CD_test_loc, 'fuel_rate_pred'] = lm.predict(X_test.loc[CD_test_loc, ['SOC','VSP']])

CD_elec_low_soc = lm.fit(X_train.loc[CD_low_soc_loc, ['VSP']], y_train.loc[CD_low_soc_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CD_test_low_soc_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[CD_test_low_soc_loc, ['VSP']])

CD_elec_high_soc = lm.fit(X_train.loc[CD_high_soc_loc, ['SOC','VSP']], y_train.loc[CD_high_soc_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CD_test_high_soc_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[CD_test_high_soc_loc, ['SOC','VSP']])
#

#
#3.1.1 fit fuel/elec under charging
CS_low_vsp_loc = (X_train['VSP']<0) & (y_train['mot_control_pred']==0) & (y_train['eng_on_pred']==1)
CS_high_vsp_loc = (X_train['VSP']>=0) & (y_train['mot_control_pred']==0) & (y_train['eng_on_pred']==1)
CS_test_low_vsp_loc = (X_test['VSP']<0) & (y_test['mot_control_pred']==0) & (y_test['eng_on_pred']==1)
CS_test_high_vsp_loc = (X_test['VSP']>=0) & (y_test['mot_control_pred']==0) & (y_test['eng_on_pred']==1)
#
CS_avg_fuel_low_vsp = y_train.loc[CS_low_vsp_loc, 'fuel_rate(J)'].mean()
if CS_avg_fuel_low_vsp is np.nan:
    CS_avg_fuel_low_vsp = 0
print(CS_avg_fuel_low_vsp)
y_test.loc[CS_test_low_vsp_loc, 'fuel_rate_pred'] = CS_avg_fuel_low_vsp
#
CS_high_vsp_fuel = lm.fit(X_train.loc[CS_high_vsp_loc, ['SOC','VSP']], y_train.loc[CS_high_vsp_loc, 'fuel_rate(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CS_test_high_vsp_loc, 'fuel_rate_pred'] = lm.predict(X_test.loc[CS_test_high_vsp_loc, ['SOC','VSP']])
#

CS_elec_low_vsp = lm.fit(X_train.loc[CS_low_vsp_loc, ['SOC', 'VSP']], y_train.loc[CS_low_vsp_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CS_test_low_vsp_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[CS_test_low_vsp_loc, ['SOC', 'VSP']])

CS_high_vsp_elec = lm.fit(X_train.loc[CS_high_vsp_loc, ['SOC', 'VSP']], y_train.loc[CS_high_vsp_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_test.loc[CS_test_high_vsp_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[CS_test_high_vsp_loc, ['SOC', 'VSP']])

#
#CS_avg_elec_high_vsp_high_soc = y_train.loc[CS_high_vsp_high_soc_loc, 'elec_energy(J)'].mean()
#if CS_avg_elec_high_vsp_high_soc is np.nan:
#    CS_avg_elec_high_vsp_high_soc = CS_avg_elec_low_vsp
#print(CS_avg_elec_high_vsp_high_soc)
#y_test.loc[CS_test_high_vsp_high_soc_loc, 'elec_rate_pred'] = CS_avg_elec_high_vsp_high_soc
#
#
#y_test.loc[y_test['fuel_rate_pred']>200000, 'fuel_rate_pred'] = 200000
#y_test.loc[y_test['fuel_rate_pred']<0, 'fuel_rate_pred'] = 0
##training_fuel_r2 = metrics.r2_score(y_train['fuel_rate(J)'], y_train['fuel_rate_pred'])
##training_elec_r2 = metrics.r2_score(y_train['elec_energy(J)'], y_train['elec_rate_pred'])
##print(training_fuel_r2, training_elec_r2)
testing_fuel_r2 = metrics.r2_score(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'])
testing_elec_r2 = metrics.r2_score(y_test['elec_energy(J)'], y_test['elec_rate_pred'])
testing_fuel_mse = metrics.mean_squared_error(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'])
testing_elec_mse = metrics.mean_squared_error(y_test['elec_energy(J)'], y_test['elec_rate_pred'])
print(testing_fuel_r2, testing_elec_r2)
print(np.sqrt(testing_fuel_mse), np.sqrt(testing_elec_mse))

#plt.subplot(121)
#g1=plt.scatter(y_test['elec_energy(J)'], y_test['elec_rate_pred'], c=y_test['eng_on'], cmap='jet_r', alpha=0.01)
#cbar = plt.colorbar(g1)
#plt.subplot(122)
#g2=plt.scatter(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'], c=y_test['eng_on'], cmap='jet_r', alpha=0.01)
#cbar = plt.colorbar(g2)
##print(lm.coef_, lm.intercept_)
#y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']>=0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame())
#y_test.loc[(X_test['VSP']>=0) & (y_test['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_test.loc[(X_test['VSP']>=0) & (y_test['eng_on_pred']==0), 'VSP'].to_frame())
#print(lm.coef_, lm.intercept_)
#elec_neg = lm.fit(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame(), y_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'elec_energy(J)'])
#y_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on_pred']==0), 'VSP'].to_frame())
#y_test.loc[(X_test['VSP']<0) & (y_test['eng_on_pred']==0), 'elec_rate_pred'] = lm.predict(X_test.loc[(X_test['VSP']<0) & (y_test['eng_on_pred']==0), 'VSP'].to_frame())
#print(lm.coef_, lm.intercept_)
##2.1 Predict motor 1 on

#fuel_eng_on_training_r2 = metrics.r2_score(y_train.loc[y_test['eng_on_pred'] == 0, 'fuel_rate(J)'], y_train.loc[y_test['eng_on_pred'] == 0, 'fuel_rate_pred'])
#fuel_eng_on_testing_r2 = metrics.r2_score(y_test.loc[y_test['eng_on_pred'] == 0, 'fuel_rate(J)'], y_test.loc[y_test['eng_on_pred'] == 0, 'fuel_rate_pred'])
#print(fuel_eng_on_training_r2, fuel_eng_on_testing_r2)



# <codecell>

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
print(piecewise_r2)


