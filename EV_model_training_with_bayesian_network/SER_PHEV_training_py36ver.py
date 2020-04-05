# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# load data
from pandas import read_csv
from os import listdir
import matplotlib.pyplot as plt
import pandas as pd


composite_cycle = None
training_cycle = listdir('Series_AER53_HEV_csv')
randomindex = []
composite_cycle = None


for cyc in training_cycle:
    print("process cycle name " + cyc)
    cycle_to_process = read_csv('Series_AER53_HEV_csv/' + cyc, sep = ',')
    cycle_to_process = cycle_to_process[cycle_to_process['Acceleration(mph/s)'].abs() < 10.0]
#    cycle_to_process = cycle_to_process[cycle_to_process['tracpower(watt)'].abs() < 60000.0]

    cycle_to_process['fuel_rate(J)'] = cycle_to_process['fuelrate(kg/s)'] * 46400000.0 * 0.1
#    cycle_to_process =cycle_to_process[cycle_to_process['time(s)']%1==0]
#    cycle_to_process['Time_step'] = cycle_to_process['time(s)'].astype(int)
    cycle_to_process['max_VSP'] = cycle_to_process['VSP'].rolling(window=20).max()
    cycle_to_process['max_VSP'].fillna(float(cycle_to_process.iloc[19]['max_VSP']), inplace = True)
    cycle_to_process['min_VSP'] = cycle_to_process['VSP'].rolling(window=20).min()
    cycle_to_process['min_VSP'].fillna(float(cycle_to_process.iloc[19]['min_VSP']), inplace = True)
    if cycle_to_process.size > 0:
        if composite_cycle is None:
            composite_cycle = cycle_to_process
        else:
            composite_cycle = pd.concat([composite_cycle, cycle_to_process])

print(len(composite_cycle))
quartile_1, quartile_3 = composite_cycle['tracpower(watt)'].quantile(0.25), composite_cycle['tracpower(watt)'].quantile(0.75)
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
composite_cycle = composite_cycle[(composite_cycle['tracpower(watt)'] >= lower_bound) & (composite_cycle['tracpower(watt)'] <= upper_bound)]
quartile_1, quartile_3 = composite_cycle['VSP'].quantile(0.25), composite_cycle['VSP'].quantile(0.75)
iqr = quartile_3 - quartile_1
lower_bound = quartile_1 - (iqr * 1.5)
upper_bound = quartile_3 + (iqr * 1.5)
composite_cycle['max_VSP'] = composite_cycle['max_VSP'] * (composite_cycle['max_VSP'] <= upper_bound)+ upper_bound * (composite_cycle['max_VSP'] > upper_bound)
composite_cycle['min_VSP'] = composite_cycle['min_VSP'] * (composite_cycle['min_VSP'] >= lower_bound)+ lower_bound * (composite_cycle['min_VSP'] < lower_bound)
composite_cycle = composite_cycle.loc[~composite_cycle['elec_energy(J)'].isnull()]
composite_cycle = composite_cycle.loc[(composite_cycle['mot_command'] != 0) |(composite_cycle['mot2_command'] != 0) | (composite_cycle['eng_on'] != 0)]
composite_cycle = composite_cycle.loc[(composite_cycle['fuel_rate(J)'] + composite_cycle['elec_energy(J)'] > 0.1 * composite_cycle['tracpower(watt)'])]
print(len(composite_cycle))
#composite_cycle.loc[:, 'eng_on'] = 1 * ((composite_cycle.loc[:, 'eng_on'] == 1) | (composite_cycle.loc[:, 'fuelrate(kg/s)'] > 0.0)) +  0 * ((composite_cycle.loc[:, 'eng_on'] == 0) | (composite_cycle.loc[:, 'fuelrate(kg/s)'] == 0.0))
composite_cycle.loc[:, 'eng_on'] = 1 * (composite_cycle.loc[:, 'fuelrate(kg/s)'] > 0.0) +  0 * (composite_cycle.loc[:, 'fuelrate(kg/s)'] == 0.0)

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
sample_cycle = composite_cycle.sample(n=100000)
plt.style.use('ggplot')
#g1= sns.jointplot(sample_cycle['VSP'], sample_cycle['fuelrate(kg/s)'], ylim=(0, 0.0016), joint_kws={"alpha":.05})

for ctrl in sample_cycle["control_type"].unique():
    g1= plt.scatter(sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'VSP'], sample_cycle.loc[sample_cycle["control_type"] == ctrl,'elec_energy(J)']*10, c=sample_cycle.loc[sample_cycle["control_type"] == ctrl,'SOC'], cmap='jet_r', alpha=0.2)
    cbar = plt.colorbar(g1)
    control_id = sample_cycle.loc[sample_cycle["control_type"] == ctrl, 'control'].unique()[0]
    plt.xlabel('VSP (watt/tonne)')
    plt.ylabel("Electricity (J/s)")
    plt.title(ctrl)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('SOC', rotation=270)
    #plt.ylim([0, 0.016])
    plt.savefig('SER_PHEV_PLOT/EREV_elec_' + str(ctrl) + '.png', bbox_inches = 'tight', dpi = 200)
    plt.show()
# <codecell>
#training model
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings

warnings.simplefilter('ignore')

plt.style.use('ggplot')
composite_cycle.loc[:, 'mot1_control'] = 1 * (composite_cycle.loc[:, 'control'].isin([1, 3])) + 0 * (composite_cycle.loc[:,'control'].isin([2, 4]))
#composite_cycle['mot2_control'] = 1 * (composite_cycle['control'].isin([1, 3, 5, 7])) + 0 * (composite_cycle['control'].isin([2, 4, 6, 8]))
composite_cycle.loc[:, 'SOC_high'] = 1 * (composite_cycle.loc[:, 'SOC'] - 0.3 >= 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 0.3 < 0)
composite_cycle.loc[:, 'SOC_bottom'] = 0 * (composite_cycle.loc[:, 'SOC'] - 0.3 >= 0) + composite_cycle.loc[:, 'SOC'] * (composite_cycle.loc[:, 'SOC'] - 0.3 < 0)
composite_cycle.loc[:, 'SOC_upper'] = 1 * (composite_cycle.loc[:, 'SOC'] - 0.36 >= 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 0.36 < 0)
composite_cycle.loc[:, 'SOC_medium'] = composite_cycle.loc[:, 'SOC'] * ((composite_cycle.loc[:, 'SOC'] - 0.2 >= 0) & (composite_cycle.loc[:, 'SOC'] - 0.36 < 0)) + 0 * ((composite_cycle.loc[:, 'SOC'] - 0.36 >= 0) | (composite_cycle.loc[:, 'SOC'] - 0.2 < 0))
composite_cycle.loc[:, 'SOC_lower'] = composite_cycle.loc[:, 'SOC'] * (composite_cycle.loc[:, 'SOC'] - 0.36 < 0) + 0 * (composite_cycle.loc[:, 'SOC'] - 0.36 >= 0)


kf = KFold(n_splits=10)

X = composite_cycle[['SOC_high', 'SOC_bottom', 'SOC', 'SOC_upper', 'SOC_medium', 'SOC_lower', 'Speed(mph)', 'Acceleration(mph/s)', 'VSP', 'max_VSP', 'min_VSP']] # cannot add more
#X = X.fillna(method='ffill')
y = composite_cycle[['eng_on', 'mot1_control', 'fuel_rate(J)', 'elec_energy(J)']]  # cannot add more
    
y.loc[:, 'fuel_rate(J)'] *= 10
y.loc[:, 'elec_energy(J)'] *= 10

y.loc[:, 'fuel_rate_pred'] = 0
y.loc[:, 'fuel_rate_eng_off_pred'] = 0
y.loc[:, 'fuel_rate_mot_pos_pred'] = 0
y.loc[:, 'fuel_rate_mot_neg_pred'] = 0
y.loc[:, 'elec_rate_pred'] = 0
y.loc[:, 'elec_rate_eng_off_pred'] = 0
y.loc[:, 'elec_rate_mot_pos_pred'] = 0
y.loc[:, 'elec_rate_mot_neg_pred'] = 0
y.loc[:, 'eng_on_pred'] = 1
y.loc[:, 'eng_off_pred'] = 0
y.loc[:, 'mot1_pos_pred'] = 1
y.loc[:, 'mot1_neg_pred'] = 0


fuel_accuracy = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#OLS_accuracy = []
piecewise_accuracy = []

#1. fit engine control model
logit = LogisticRegression()
eng_control = logit.fit(X_train[['SOC_high', 'Speed(mph)', 'max_VSP', 'min_VSP']], y_train['eng_on'])
print(logit.coef_, logit.intercept_)
y_train.loc[:, 'eng_on_pred'] = logit.predict(X_train[['SOC_high', 'Speed(mph)', 'max_VSP', 'min_VSP']])
eng_on_training_acc = metrics.accuracy_score(y_train['eng_on'], y_train['eng_on_pred'])
#y_test.loc[:, 'eng_on_pred'] = logit.predict(X_test[['SOC_high', 'Speed(mph)', 'max_VSP', 'min_VSP']])
y_test.loc[:, ['eng_off_pred', 'eng_on_pred']] = logit.predict_proba(X_test[['SOC_high', 'Speed(mph)', 'max_VSP', 'min_VSP']])
#eng_on_testing_acc = metrics.accuracy_score(y_test['eng_on'], y_test['eng_on_pred'])
print(eng_on_training_acc)
#confusion_matrix = confusion_matrix(y_test['eng_on'], y_test['eng_on_pred'])
#print(confusion_matrix)

#2. fit energy model under engine off 

##2.1 Fuel
avg_fuel_rate_eng_off = y_train.loc[y_train['eng_on'] == 0, 'fuel_rate(J)'].mean()
y_train.loc[y_train['eng_on'] == 0, 'fuel_rate_pred'] = avg_fuel_rate_eng_off
y_test.loc[:, 'fuel_rate_eng_off_pred'] = avg_fuel_rate_eng_off
eng_off_fuel_r2 = metrics.r2_score(y_train.loc[y_train['eng_on'] == 0, 'fuel_rate(J)'], y_train.loc[y_train['eng_on'] == 0, 'fuel_rate_pred'])
print(eng_off_fuel_r2)
#
##2.2 Electric
lm = linear_model.Ridge(alpha=0.1)
elec_pos = lm.fit(X_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'VSP'].to_frame(), y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'elec_energy(J)'])
y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'VSP'].to_frame())
print(lm.coef_, lm.intercept_)
eng_off_elec_pos_vsp_r2 = metrics.r2_score(y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'elec_energy(J)'], y_train.loc[(X_train['VSP']>=0) & (y_train['eng_on']==0), 'elec_rate_pred'])
print(eng_off_elec_pos_vsp_r2)
y_test.loc[(X_test['VSP']>=0), 'elec_rate_eng_off_pred'] = lm.predict(X_test.loc[(X_test['VSP']>=0), 'VSP'].to_frame())
#
#
elec_neg = lm.fit(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'VSP'].to_frame(), y_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'elec_energy(J)'])
y_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'elec_rate_pred'] = lm.predict(X_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'VSP'].to_frame())
print(lm.coef_, lm.intercept_)
eng_off_elec_neg_vsp_r2 = metrics.r2_score(y_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'elec_energy(J)'], y_train.loc[(X_train['VSP']<0) & (y_train['eng_on']==0), 'elec_rate_pred'])
print(eng_off_elec_neg_vsp_r2)
y_test.loc[(X_test['VSP']<0), 'elec_rate_eng_off_pred'] = lm.predict(X_test.loc[(X_test['VSP']<0), 'VSP'].to_frame())
#
#
#3. fit energy model under engine on 

#3.1.1 fit motor1 motoring(+)/generating(-)

mot1_control = logit.fit(X_train.loc[y_train['eng_on'] == 1, ['VSP']], y_train.loc[y_train['eng_on'] == 1, 'mot1_control'])
print(logit.coef_, logit.intercept_)
y_train.loc[y_train['eng_on'] == 1, 'mot1_control_pred'] = logit.predict(X_train.loc[y_train['eng_on'] == 1, ['VSP']])
mot1_on_training_acc = metrics.accuracy_score(y_train.loc[y_train['eng_on'] == 1, 'mot1_control'], y_train.loc[y_train['eng_on'] == 1, 'mot1_control_pred'])
print(mot1_on_training_acc)
#
y_test.loc[:, ['mot1_neg_pred','mot1_pos_pred']] = logit.predict_proba(X_test.loc[:, ['VSP']])
y_test.loc[:, 'mot1_neg_pred'] = y_test.loc[:, 'mot1_neg_pred'] * y_test.loc[:, 'eng_on_pred']
y_test.loc[:, 'mot1_pos_pred'] = y_test.loc[:, 'mot1_pos_pred'] * y_test.loc[:, 'eng_on_pred']
##
##3.2 fit fuel under mot1 motoring
motoring_loc = (y_train['mot1_control']==1) & (y_train['eng_on']==1)
#motoring_test_loc = (y_test['mot1_control_pred']==1) & (y_test['eng_on_pred']==1)
#motoring_low_soc_loc = (X_train['SOC']<=0.33) & (y_train['mot1_control_pred']==1) & (y_train['eng_on_pred']==1)
#motoring_low_soc_test_loc = (X_test['SOC']<=0.33) & (y_test['mot1_control_pred']==1) & (y_test['eng_on_pred']==1)
#motoring_med_soc_loc = (X_train['SOC']<=0.36) & (X_train['SOC']>0.33) & (y_train['mot1_control_pred']==1) & (y_train['eng_on_pred']==1)
#motoring_med_soc_test_loc =(X_test['SOC']<=0.36) & (X_test['SOC']>0.33) & (y_test['mot1_control_pred']==1) & (y_test['eng_on_pred']==1)
#motoring_high_soc_loc = (X_train['SOC']>=0.36) & (y_train['mot1_control_pred']==1) & (y_train['eng_on_pred']==1)
#motoring_high_soc_test_loc = (X_test['SOC']>=0.36) & (y_test['mot1_control_pred']==1) & (y_test['eng_on_pred']==1)
##
##3.2.1 fuel model under engine on/ motor 1 positive
motoring_fuel = lm.fit(X_train.loc[motoring_loc, ['SOC_upper', 'SOC_lower', 'VSP']], y_train.loc[motoring_loc, 'fuel_rate(J)'])
print(lm.coef_, lm.intercept_)
y_train.loc[motoring_loc, 'fuel_rate_pred'] = lm.predict(X_train.loc[motoring_loc, ['SOC_upper', 'SOC_lower', 'VSP']])
eng_on_mot_pos_fuel_r2 = metrics.r2_score(y_train.loc[motoring_loc, 'fuel_rate(J)'], y_train.loc[motoring_loc, 'fuel_rate_pred'])
print(eng_on_mot_pos_fuel_r2)
y_test.loc[:, 'fuel_rate_mot_pos_pred'] = lm.predict(X_test.loc[:, ['SOC_upper','SOC_lower', 'VSP']])

#
#3.2.2 electric model under engine on/ motor 1 positive
motoring_low_soc_elec = lm.fit(X_train.loc[motoring_loc, ['SOC_upper', 'SOC_lower', 'VSP']], y_train.loc[motoring_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_train.loc[motoring_loc, 'elec_rate_pred'] = lm.predict(X_train.loc[motoring_loc, ['SOC_upper', 'SOC_lower', 'VSP']])
plt.scatter(X_train.loc[motoring_loc, 'VSP'], y_train.loc[motoring_loc, 'elec_energy(J)'], alpha = 0.1)
plt.show()
eng_on_mot_pos_elec_r2 = metrics.r2_score(y_train.loc[motoring_loc, 'elec_energy(J)'], y_train.loc[motoring_loc, 'elec_rate_pred'])
print(eng_on_mot_pos_elec_r2)
y_test.loc[:, 'elec_rate_mot_pos_pred'] = lm.predict(X_test.loc[:, ['SOC_upper', 'SOC_lower', 'VSP']])
##
###motoring_medium_soc_elec = lm.fit(X_train.loc[motoring_med_soc_loc, ['VSP']], y_train.loc[motoring_med_soc_loc, 'elec_energy(J)'])
###print(lm.coef_, lm.intercept_)
###y_test.loc[motoring_med_soc_test_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[motoring_med_soc_test_loc, ['VSP']])
###
###motoring_elec = lm.fit(X_train.loc[motoring_high_soc_loc, ['Speed(mph)', 'SOC','VSP']], y_train.loc[motoring_high_soc_loc, 'elec_energy(J)'])
###print(lm.coef_, lm.intercept_)
###y_test.loc[motoring_high_soc_test_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[motoring_high_soc_test_loc, ['Speed(mph)', 'SOC','VSP']])
###
###
#3.3 fit fuel under mot1 generating
generating_loc = (y_train['mot1_control']==0) & (y_train['eng_on']==1)
##generating_test_loc = (y_test['mot1_control_pred']==0) & (y_test['eng_on_pred']==1)
##generating_pos_vsp_loc = (X_train['VSP']>=0) & (y_train['mot1_control_pred']==0) & (y_train['eng_on_pred']==1)
###generating_pos_vsp_test_loc = (X_test['VSP']>=0) & (y_test['mot1_control_pred']==0) & (y_test['eng_on_pred']==1)
##generating_neg_vsp_loc = (X_train['VSP']<0) & (y_train['mot1_control_pred']==0) & (y_train['eng_on_pred']==1)
##generating_neg_vsp_test_loc = (X_test['VSP']<0) & (y_test['mot1_control_pred']==0) & (y_test['eng_on_pred']==1)
##
#3.3.1 fit fuel model under engine on/ motor1 negative
generating_fuel = lm.fit(X_train.loc[generating_loc, ['SOC_bottom', 'SOC_high', 'VSP']], y_train.loc[generating_loc, 'fuel_rate(J)'])
print(lm.coef_, lm.intercept_)
y_train.loc[generating_loc, 'fuel_rate_pred'] = lm.predict(X_train.loc[generating_loc, ['SOC_bottom', 'SOC_high', 'VSP']])
eng_on_mot_neg_fuel_r2 = metrics.r2_score(y_train.loc[generating_loc, 'fuel_rate(J)'], y_train.loc[generating_loc, 'fuel_rate_pred'])
print(eng_on_mot_neg_fuel_r2)
y_test.loc[:, 'fuel_rate_mot_neg_pred'] = lm.predict(X_test.loc[:, ['SOC_bottom', 'SOC_high', 'VSP']])
#y_test.loc[generating_test_loc, 'fuel_rate_pred'] = lm.predict(X_test.loc[generating_test_loc, ['SOC','VSP']])
#

##3.3.2 fit electric model under engine on/ motor1 negative
generating_elec = lm.fit(X_train.loc[generating_loc, ['SOC_bottom', 'SOC_high', 'VSP']], y_train.loc[generating_loc, 'elec_energy(J)'])
print(lm.coef_, lm.intercept_)
y_train.loc[generating_loc, 'elec_rate_pred'] = lm.predict(X_train.loc[generating_loc, ['SOC_bottom', 'SOC_high', 'VSP']])
plt.scatter(y_train.loc[generating_loc, 'elec_energy(J)'], y_train.loc[generating_loc, 'elec_rate_pred'], alpha = 0.1)
plt.show()
eng_on_mot_neg_elec_vsp_pos_r2 = metrics.r2_score(y_train.loc[generating_loc, 'elec_energy(J)'], y_train.loc[generating_loc, 'elec_rate_pred'])
print(eng_on_mot_neg_elec_vsp_pos_r2)
y_test.loc[:, 'elec_rate_mot_neg_pred'] = lm.predict(X_test.loc[:, [ 'SOC_bottom', 'SOC_high', 'VSP']])
###
####generating_elec = lm.fit(X_train.loc[generating_neg_vsp_loc, ['SOC','VSP']], y_train.loc[generating_neg_vsp_loc, 'elec_energy(J)'])
####print(lm.coef_, lm.intercept_)
####y_test.loc[generating_neg_vsp_test_loc, 'elec_rate_pred'] = lm.predict(X_test.loc[generating_neg_vsp_test_loc, ['SOC','VSP']])
####
####
##3.4 Model performance assessment
y_test.loc[:, 'fuel_rate_pred'] = y_test.loc[:, 'fuel_rate_mot_neg_pred'] * y_test.loc[:, 'mot1_neg_pred'] + y_test.loc[:, 'fuel_rate_mot_pos_pred'] * y_test.loc[:, 'mot1_pos_pred'] +y_test.loc[:, 'fuel_rate_eng_off_pred'] * y_test.loc[:, 'eng_off_pred']
y_test.loc[:, 'elec_rate_pred'] = y_test.loc[:, 'elec_rate_mot_neg_pred'] * y_test.loc[:, 'mot1_neg_pred'] + y_test.loc[:, 'elec_rate_mot_pos_pred'] * y_test.loc[:, 'mot1_pos_pred'] +y_test.loc[:, 'elec_rate_eng_off_pred'] * y_test.loc[:, 'eng_off_pred']
y_test.loc[:, 'fuel_rate_pred'] = y_test.loc[:, 'fuel_rate_pred'] * (y_test.loc[:, 'fuel_rate_pred'] <=200000) + 200000 * (y_test.loc[:, 'fuel_rate_pred'] > 200000)
testing_fuel_r2 = metrics.r2_score(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'])
testing_elec_r2 = metrics.r2_score(y_test['elec_energy(J)'], y_test['elec_rate_pred'])
testing_energy_r2 = metrics.r2_score(y_test['fuel_rate(J)'] + y_test['elec_energy(J)'], y_test['fuel_rate_pred'] + y_test['elec_rate_pred'])

testing_fuel_mse = metrics.mean_squared_error(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'])
testing_elec_mse = metrics.mean_squared_error(y_test['elec_energy(J)'], y_test['elec_rate_pred'])
testing_energy_mse = metrics.mean_squared_error(y_test['fuel_rate(J)'] + y_test['elec_energy(J)'], y_test['fuel_rate_pred'] + y_test['elec_rate_pred'])

print(testing_fuel_r2, testing_elec_r2, testing_energy_r2)
print(np.sqrt(testing_fuel_mse), np.sqrt(testing_elec_mse), np.sqrt(testing_energy_mse))

##
g1=plt.scatter(y_test['elec_energy(J)'], y_test['elec_rate_pred'], c=y_test['eng_on'], cmap='jet_r', alpha=0.01)
cbar = plt.colorbar(g1)
plt.show()
#
g2=plt.scatter(y_test['fuel_rate(J)'], y_test['fuel_rate_pred'], c=y_test['eng_on'], cmap='jet_r', alpha=0.01)
cbar = plt.colorbar(g2)
plt.show()

# <codecell>

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def eng_on_fitting(SOC, Speed, max_VSP, min_VSP, eng_on_param):
    x = [(SOC>0.3), Speed, max_VSP, min_VSP, 1]
    eng_on_score = np.dot(eng_on_param, x)
#    print(x, eng_on_score)
    eng_on_prob = sigmoid(eng_on_score)
    return eng_on_prob

def mot1_pos_fitting(Speed, VSP, mot1_pos_param):
    x = [Speed, VSP, 1]
    mot_pos_score = np.dot(mot1_pos_param, x)
    mot_pos_prob = sigmoid(mot_pos_score)
    return mot_pos_prob

def update_soc(initial_soc, energy_consumption, battery_size):
    energy_available = initial_soc * battery_size
    energy_remaining = energy_available - energy_consumption
    latest_soc = energy_remaining / battery_size
    return latest_soc

def energy_model_fitting(SOC, Speed, VSP, max_VSP, min_VSP, eng_on_param, mot1_pos_param, resolution):
    eng_on_prob = eng_on_fitting(SOC, Speed, max_VSP, min_VSP, eng_on_param)
    mot1_pos_prob = mot1_pos_fitting(Speed, VSP, mot1_pos_param)
    mot1_pos_eng_on_prob = eng_on_prob * mot1_pos_prob
    mot1_neg_eng_on_prob = eng_on_prob * (1 - mot1_pos_prob)
    fuel_eng_off = 0 * resolution
    print(eng_on_prob)
    elec_eng_off = (VSP>=0) * (2.45157107 * VSP + 144.38601445994937) + (VSP<0) * (1.33505183 * VSP + 346.41790648524875)
    fuel_eng_on_mot1_pos =  -3.38597796e+05 * 1 * (SOC>=0.36) - 1.16994098e+06 * SOC * (SOC<0.36) + 3.3584077 * VSP + 363839.0015780216
    elec_eng_on_mot1_pos =  1.69653646e+05 * 1 * (SOC>=0.36) + 5.79011468e+05 * SOC * (SOC<0.36) + 5.57667998e-01 * VSP -173651.62228421587
    fuel_eng_on_mot1_neg = 1.28258355e+02 * Speed - 1.91192723e+05 * SOC * (SOC<0.3) - 2.61734567e+05 * SOC * (SOC<0.36) * (SOC>=0.3) - 6.80486243e+04 * 1 * (SOC>=0.36) + 4.11470992 * VSP + 75964.08479507004
    elec_eng_on_mot1_neg = 3.47681620e+02 * Speed - 6.92437852e+04 * SOC * (SOC<0.3) - 3.36729340e+04 * SOC * (SOC<0.36) * (SOC>=0.3) - 1.46818108e+04 * 1 * (SOC>=0.36) - 3.51383515e-01 * VSP + 3199.503220777504
    fuel_consumption = fuel_eng_off * (1 - eng_on_prob) + fuel_eng_on_mot1_pos * mot1_pos_eng_on_prob + fuel_eng_on_mot1_neg * mot1_neg_eng_on_prob
    fuel_consumption *= resolution
    elec_consumption = elec_eng_off * (1 - eng_on_prob) + elec_eng_on_mot1_pos * mot1_pos_eng_on_prob + elec_eng_on_mot1_neg * mot1_neg_eng_on_prob
    elec_consumption *= resolution
#    print(mot1_pos_eng_on_prob, mot1_neg_eng_on_prob)
    return fuel_consumption, elec_consumption


#####sample data testing #####
testing_cycle = read_csv('Power_split_HEV_ARC_hwy_Spring_4054072_3_36_1_15.csv', sep=',')
#testing_cycle['VSP'] = testing_cycle['tracpower(watt)'] / (1712 * 0.001)
testing_cycle['fuel_rate(J)'] = testing_cycle['fuelrate(kg/s)'] * 46400000.0 * 0.1
testing_cycle['max_VSP'] = testing_cycle['VSP'].rolling(window=20).max()
testing_cycle['max_VSP'].fillna(float(testing_cycle.iloc[19]['max_VSP']), inplace = True)
testing_cycle['min_VSP'] = testing_cycle['VSP'].rolling(window=20).min()
testing_cycle['min_VSP'].fillna(float(testing_cycle.iloc[19]['min_VSP']), inplace = True)
testing_cycle['SOC_est'] = 0

testing_cycle['fuel_est'] = 0
testing_cycle['elec_est'] = 0
initial_soc = testing_cycle.loc[0, 'SOC']
testing_cycle.loc[:, 'SOC_est'] = initial_soc
battery_size = 37.5684 * 60 * 3.6 / 1000 * 3.6e+06

eng_on_param = [-2.67677801, 4.24002192e-02, 9.15611588e-05, 1.65119099e-04, -1.93716152]
mot1_pos_param = [-0.11713582, 0.00028122, 0.04196995]

for index, row in testing_cycle.iterrows():
    print(index)
    testing_cycle.loc[index, 'fuel_est'], testing_cycle.loc[index, 'elec_est'] = energy_model_fitting(row['SOC_est'], testing_cycle.loc[index, 'Speed(mph)'], testing_cycle.loc[index, 'VSP'], testing_cycle.loc[index, 'max_VSP'], testing_cycle.loc[index, 'min_VSP'], eng_on_param, mot1_pos_param, resolution=0.1)
#    latest_soc = update_soc(testing_cycle.loc[index, 'SOC_est'], testing_cycle.loc[index, 'elec_est'], battery_size)
#    if index+1 in testing_cycle.index:
##        print(testing_cycle.loc[index, 'elec_est'], latest_soc) 
#        testing_cycle.loc[index + 1, 'SOC_est'] = latest_soc


testing_cycle.to_csv('Power_split_HEV_ARC_hwy_Spring_est.csv', sep=',')
#    print row["c1"], row["c2"]
#fuel_consumption, elec_consumption, latest_soc = energy_model_fitting(initial_soc, testing_cycle.loc[0, 'Speed(mph)'], testing_cycle.loc[0, 'VSP'], testing_cycle.loc[0, 'max_VSP'], testing_cycle.loc[0, 'min_VSP'], eng_on_param, mot1_pos_param, battery_size, resolution=0.1)
#print(fuel_consumption, elec_consumption)



#data_with_intercept = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
#                                 simulated_separableish_features))
#final_scores = np.dot(data_with_intercept, weights)