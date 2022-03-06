import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.preprocessing import minmax_scale
import pickle
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import random
import torch
from sklearn.preprocessing import StandardScaler
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.normalization import batch_normalization
from keras.regularizers import l2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
from keras.regularizers import l1


#loading data
train_set=pd.read_csv('../data-sets/KPI/data/train-data.csv')
df = pd.read_csv('train-data.csv')
df.head()
test_set=pd.read_hdf('../data-sets/KPI/data/test-data.hdf')
print(test_set.groupby(['KPI ID']).size())
print(train_set.groupby(['KPI ID']).size())
print(test_set)
#print(len(train_set))
#print(train_set)

print(len(train_set.groupby(['KPI ID']).size()))
class_mapping = {'05f10d3a-239c-3bef-9bdc-a2feeb0037aa':0, '0efb375b-b902-3661-ab23-9a0bb799f4e3':1,'1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0':2,
                 '43115f2a-baeb-3b01-96f7-4ea14188343c':3,'431a8542-c468-3988-a508-3afd06a218da':4,
                 '4d2af31a-9916-3d9f-8a8e-8a268a48c095':5,'54350a12-7a9d-3ca8-b81f-f886b9d156fd':6,
                 '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9':7,'57051487-3a40-3828-9084-a12f7f23ee38':8,
                 '6a757df4-95e5-3357-8406-165e2bd49360':9,'6d1114ae-be04-3c46-b5aa-be1a003a57cd':10,
                 '6efa3a07-4544-34a0-b921-a155bd1a05e8':11,'7103fa0f-cac4-314f-addc-866190247439':12,
                 '847e8ecc-f8d2-3a93-9107-f367a0aab37d':13,'8723f0fb-eaef-32e6-b372-6034c9c04b80':14,
                 '9c639a46-34c8-39bc-aaf0-9144b37adfc8':15,'a07ac296-de40-3a7c-8df3-91f642cc14d0':16,
                 'a8c06b47-cc41-3738-9110-12df0ee4c721':17,'ab216663-dcc2-3a24-b1ee-2c3e550e06c9':18,
                 'adb2fde9-8589-3f5b-a410-5fe14386c7af':19,'ba5f3328-9f3f-3ff5-a683-84437d16d554':20,
                 'c02607e8-7399-3dde-9d28-8a8da5e5d251':21,'c69a50cf-ee03-3bd7-831e-407d36c7ee91':22,
                 'da10a69f-d836-3baa-ad40-3e548ecf1fbd':23,'e0747cad-8dc8-38a9-a9ab-855b61f5551d':24,
                 'f0932edd-6400-3e63-9559-0a9860a1baa9':25,'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa':26,
                 '301c70d8-1630-35ac-8f96-bc1b6f4359ea':27,'42d6616d-c9c5-370a-a8ba-17ead74f3114':28
                 }
train=train_set
train['KPI ID']=train_set['KPI ID'].map(class_mapping)
train_set['KPI ID'] = train_set['KPI ID'].map(class_mapping)
test_set['KPI ID'] = test_set['KPI ID'].map(class_mapping)
print(train_set.groupby(['KPI ID']).size())

print(test_set.groupby(['KPI ID']).size())
print(train_set)
print(test_set)
# 05f10d3a-239c-3bef-9bdc-a2feeb0037aa    146255
# 0efb375b-b902-3661-ab23-9a0bb799f4e3      8784
# 1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0
# 301c70d8-1630-35ac-8f96-bc1b6f4359ea
# 42d6616d-c9c5-370a-a8ba-17ead74f3114
# 43115f2a-baeb-3b01-96f7-4ea14188343c
# 431a8542-c468-3988-a508-3afd06a218da
# 4d2af31a-9916-3d9f-8a8e-8a268a48c095
# 54350a12-7a9d-3ca8-b81f-f886b9d156fd
# 55f8b8b8-b659-38df-b3df-e4a5a8a54bc9
# 57051487-3a40-3828-9084-a12f7f23ee38
# 6a757df4-95e5-3357-8406-165e2bd49360
# 6d1114ae-be04-3c46-b5aa-be1a003a57cd
# 6efa3a07-4544-34a0-b921-a155bd1a05e8
# 7103fa0f-cac4-314f-addc-866190247439
# 847e8ecc-f8d2-3a93-9107-f367a0aab37d
# 8723f0fb-eaef-32e6-b372-6034c9c04b80
# 9c639a46-34c8-39bc-aaf0-9144b37adfc8
# a07ac296-de40-3a7c-8df3-91f642cc14d0
# a8c06b47-cc41-3738-9110-12df0ee4c721
# ab216663-dcc2-3a24-b1ee-2c3e550e06c9
# adb2fde9-8589-3f5b-a410-5fe14386c7af
# ba5f3328-9f3f-3ff5-a683-84437d16d554
# c02607e8-7399-3dde-9d28-8a8da5e5d251
# c69a50cf-ee03-3bd7-831e-407d36c7ee91
# da10a69f-d836-3baa-ad40-3e548ecf1fbd
# e0747cad-8dc8-38a9-a9ab-855b61f5551d
# f0932edd-6400-3e63-9559-0a9860a1baa9
# ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa


print(train_set.groupby(['label']).size())
print(test_set)

#construct the model
print(df)
features=df.loc[:,df.columns!='label'].values[:,1:]
labels=df.loc[:,'label'].values

features1=test_set.loc[:,test_set.columns!='label'].values[:,1:]
labels1=test_set.loc[:,'label'].values


print(labels[labels==1].shape[0], labels[labels==0].shape[0])
print(labels1[labels1==1].shape[0], labels1[labels1==0].shape[0])

print(train)
# scaler=MinMaxScaler((-1,1))
# x=scaler.fit_transform(features)
# y=labels

# model=XGBClassifier()
# model.fit(train_set,test_set)
labels=['normal','anomaly']
sizes=[2924512,79554]
size1=[2864287,54560]
explode = (0,0)
plt.pie(size1,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
plt.title("normal vs anomaly in train_set")
plt.show()


#begin
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_hdf('test-data.hdf')
print(train_data)
print(test_data)

id_list, id_indexes = np.unique(train_data['KPI ID'], return_index=True)
id_indexes.sort()
id_indexes = np.append(id_indexes, len(train_data))
timeseries_all = []
timeseries_label = []

for i in np.arange(len(id_indexes)-1):
    timeseries_all.append(np.asarray(train_data['value'][id_indexes[i]:id_indexes[i+1]]))
    timeseries_label.append(np.asarray(train_data['label'][id_indexes[i]:id_indexes[i+1]]))
print(timeseries_all)
print(timeseries_label)


sum1 = 0
for i in range(29):
    for j in range(len(timeseries_all[i])):
        if abs(timeseries_all[i][j] - train_data['value'][sum1+j])>1e-6:
            print(i,j,timeseries_all[i][j],train_data['value'][sum1+j])
        if timeseries_label[i][j] != train_data['label'][sum1+j]:
            print(i,j,timeseries_label[i][j],train_data['label'][sum1+j])
    sum1 += len(timeseries_all[i])
test_id_list, test_id_indexes = np.unique(test_data['KPI ID'], return_index=True)
test_id_indexes.sort()
test_id_indexes = np.append(test_id_indexes, len(test_data))
testseries_all = []

for i in np.arange(len(test_id_indexes)-1):
    testseries_all.append(np.asarray(test_data['value'][test_id_indexes[i]:test_id_indexes[i+1]]))
print(testseries_all)
file = open('train_test_all.txt','wb')
pickle.dump(timeseries_all,file)
pickle.dump(timeseries_label,file)
pickle.dump(testseries_all,file)
file.close()
for i in range(len(timeseries_all)):
    print(len(timeseries_all[i]),len(testseries_all[i]))
# sum1 = 0
# for i in range(29):
#     print(i)
#     for j in range(len(testseries_all[i])):
#         if abs(testseries_all[i][j] - test_data['value'][sum1+j])>1e-6:
#             print(i,j,testseries_all[i][j],test_data['value'][sum1+j])
#     sum1 += len(testseries_all[i])
timeseries_scaled = []
for i in range(len(timeseries_all)):
    timeseries_scaled.append(minmax_scale(timeseries_all[i]))
print(timeseries_scaled)
print(len(timeseries_scaled))

#data preprocessing
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
def get_feature_logs(time_series):
    return np.log(time_series + 1e-2)
def get_feature_SARIMA_residuals(time_series):
    predict = sm.tsa.statespace.SARIMAX(time_series,
                      trend='n',
                      order=(5,1,1),
                      measurement_error=True).fit().get_prediction()
    return time_series - predict.predicted_mean

def get_feature_AddES_residuals(time_series):
    predict = ExponentialSmoothing(time_series, trend='add').fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_SimpleES_residuals(time_series):
    predict = SimpleExpSmoothing(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_Holt_residuals(time_series):
    predict = Holt(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues


def get_timeseries_features(time_series, time_series_label, Windows, delay):
    data = []
    data_label = []
    data_label_vital = []

    start_point = 2 * max(Windows) - 1
    start_accum = sum(time_series[0:start_point])

    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    time_series_logs = get_feature_logs(time_series)

    for i in np.arange(start_point, len(time_series)):
        datum = []
        datum_label = time_series_label[i]

        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i + 1)

        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])

        datum.append(time_series_logs[i])

        datum.append(diff_plain)

        datum.append(diff_plain / (time_series[i - 1] + 1e-8))

        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))

        datum.append(time_series[i] - mean_accum)

        for k in Windows:
            mean_w = np.mean(time_series[i - k + 1:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k + 1:i + 1]) - mean_w) ** 2)

            mean_w_and_1 = mean_w + (time_series[i - k] - time_series[i]) / k
            var_w_and_1 = np.mean((np.asarray(time_series[i - k:i]) - mean_w_and_1) ** 2)

            mean_2w = np.mean(time_series[i - 2 * k + 1:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k + 1:i - k + 1]) - mean_2w) ** 2)

            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            datum.append(mean_w)

            datum.append(var_w)

            datum.append(diff_mean_1)

            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-8))

            datum.append(diff_var_1)

            datum.append(diff_var_1 / (var_w_and_1 + 1e-8))

            datum.append(diff_mean_w)

            datum.append(diff_mean_w / (mean_2w + 1e-8))

            datum.append(diff_var_w)

            datum.append(diff_var_w / (var_2w + 1e-8))

            datum.append(time_series[i] - mean_w_and_1)

            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))
        data_label.append(np.asarray(datum_label))

        if datum_label == 1 and sum(time_series_label[i - delay:i]) < delay:
            data_label_vital.append(np.asarray(1))
        else:
            data_label_vital.append(np.asarray(0))

    return data, data_label, data_label_vital


W = np.asarray([2, 5, 10, 25, 50, 100, 200, 300, 400, 500])
delay = 7
scaler_list = []
timeseries_features = None
timeseries_features_label = []
timeseries_features_label_vital = []

#prediction algorithm
for i in range(len(timeseries_all)):
    features_temp, label_temp, label_vital_temp = get_timeseries_features(timeseries_scaled[i], timeseries_label[i], W,
                                                                          delay)
    print(i)
    assert (len(features_temp) == len(label_temp))
    assert (len(label_temp) == len(label_vital_temp))
    scaler_temp = StandardScaler()
    features_temp = scaler_temp.fit_transform(features_temp)
    scaler_list.append(scaler_temp)
    if i == 0:
        timeseries_features = features_temp
    else:
        timeseries_features = np.concatenate((timeseries_features, features_temp), axis=0)

    timeseries_features_label = timeseries_features_label + label_temp
    timeseries_features_label_vital = timeseries_features_label_vital + label_vital_temp
print(timeseries_features.shape)
print(len(timeseries_features_label),len(timeseries_features_label_vital))
timeseries_features_label = np.array(timeseries_features_label)
timeseries_features_label_vital = np.array(timeseries_features_label_vital)
file = open('timeseries_features2.txt','wb')
pickle.dump(timeseries_features,file)
pickle.dump(scaler_list,file)
pickle.dump(timeseries_features_label,file)
pickle.dump(timeseries_features_label_vital,file)
file.close()
np.random.seed(5)
model = Sequential()
model.add(Dense(256, input_dim = 128))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
#model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','binary_accuracy'])
print(sum(timeseries_features_label))
print(sum(timeseries_features_label_vital))
ratio = round((len(timeseries_features_label) - sum(timeseries_features_label)) / sum(timeseries_features_label))
print(ratio)
non_anomaly = np.ones(len(timeseries_features_label)) - timeseries_features_label
print(non_anomaly,non_anomaly.shape)
sample_ratio = (4*ratio) * timeseries_features_label_vital + non_anomaly
print(sample_ratio,sum(sample_ratio))
sample_ratio = sample_ratio + ratio * timeseries_features_label
print(sample_ratio,sum(sample_ratio))
"""begin to train the first DNN"""
print('Keras: start to train DNN!')
start_time = time.time()
history = model.fit(timeseries_features, timeseries_features_label, epochs=100, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))
"""one train process show as follows"""
'''
Keras: start to train DNN!
Epoch 1/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1589 - accuracy: 0.9775 - binary_accuracy: 0.97
Epoch 2/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1596 - accuracy: 0.9772 - binary_accuracy: 0.97
Epoch 3/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1581 - accuracy: 0.9767 - binary_accuracy: 0.97
Epoch 4/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1619 - accuracy: 0.9767 - binary_accuracy: 0.97
Epoch 5/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1582 - accuracy: 0.9775 - binary_accuracy: 0.97
Epoch 6/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1591 - accuracy: 0.9773 - binary_accuracy: 0.97
Epoch 7/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1591 - accuracy: 0.9772 - binary_accuracy: 0.97
Epoch 8/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1567 - accuracy: 0.9774 - binary_accuracy: 0.97
Epoch 9/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1548 - accuracy: 0.9779 - binary_accuracy: 0.97
Epoch 10/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1527 - accuracy: 0.9780 - binary_accuracy: 0.97
Epoch 11/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1558 - accuracy: 0.9777 - binary_accuracy: 0.97
Epoch 12/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1548 - accuracy: 0.9778 - binary_accuracy: 0.97
Epoch 13/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1577 - accuracy: 0.9778 - binary_accuracy: 0.97
Epoch 14/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1477 - accuracy: 0.9786 - binary_accuracy: 0.97
Epoch 15/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1518 - accuracy: 0.9781 - binary_accuracy: 0.97
Epoch 16/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1506 - accuracy: 0.9781 - binary_accuracy: 0.97
Epoch 17/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1499 - accuracy: 0.9780 - binary_accuracy: 0.97
Epoch 18/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1502 - accuracy: 0.9777 - binary_accuracy: 0.97
Epoch 19/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1495 - accuracy: 0.9777 - binary_accuracy: 0.97
Epoch 20/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1477 - accuracy: 0.9786 - binary_accuracy: 0.97
Epoch 21/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1478 - accuracy: 0.9779 - binary_accuracy: 0.97
Epoch 22/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1440 - accuracy: 0.9786 - binary_accuracy: 0.97
Epoch 23/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1469 - accuracy: 0.9782 - binary_accuracy: 0.97
Epoch 24/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1467 - accuracy: 0.9781 - binary_accuracy: 0.97
Epoch 25/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1454 - accuracy: 0.9785 - binary_accuracy: 0.97
Epoch 26/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1478 - accuracy: 0.9776 - binary_accuracy: 0.97
Epoch 27/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1468 - accuracy: 0.9787 - binary_accuracy: 0.97
Epoch 28/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1408 - accuracy: 0.9790 - binary_accuracy: 0.97
Epoch 29/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1463 - accuracy: 0.9778 - binary_accuracy: 0.97
Epoch 30/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1456 - accuracy: 0.9782 - binary_accuracy: 0.97
Epoch 31/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1419 - accuracy: 0.9787 - binary_accuracy: 0.97
Epoch 32/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1380 - accuracy: 0.9795 - binary_accuracy: 0.97
Epoch 33/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1410 - accuracy: 0.9792 - binary_accuracy: 0.97
Epoch 34/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1405 - accuracy: 0.9789 - binary_accuracy: 0.97
Epoch 35/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1414 - accuracy: 0.9789 - binary_accuracy: 0.97
Epoch 36/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1377 - accuracy: 0.9793 - binary_accuracy: 0.97
Epoch 37/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1395 - accuracy: 0.9792 - binary_accuracy: 0.97
Epoch 38/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1417 - accuracy: 0.9789 - binary_accuracy: 0.97
Epoch 39/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1399 - accuracy: 0.9787 - binary_accuracy: 0.97
Epoch 40/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1404 - accuracy: 0.9792 - binary_accuracy: 0.97
Epoch 41/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1541 - accuracy: 0.9765 - binary_accuracy: 0.97
Epoch 42/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1468 - accuracy: 0.9780 - binary_accuracy: 0.97
Epoch 43/500
596/596 [==============================] - 17s 29ms/step - loss: 0.1387 - accuracy: 0.9789 - binary_accuracy: 0.97
Epoch 44/500
596/596 [==============================] - 19s 33ms/step - loss: 0.1352 - accuracy: 0.9795 - binary_accuracy: 0.97
Epoch 45/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1369 - accuracy: 0.9795 - binary_accuracy: 0.97
Epoch 46/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1347 - accuracy: 0.9796 - binary_accuracy: 0.97
Epoch 47/500
596/596 [==============================] - 19s 31ms/step - loss: 0.1348 - accuracy: 0.9797 - binary_accuracy: 0.97
Epoch 48/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1342 - accuracy: 0.9797 - binary_accuracy: 0.97
Epoch 49/500
596/596 [==============================] - 20s 33ms/step - loss: 0.1362 - accuracy: 0.9795 - binary_accuracy: 0.97
Epoch 50/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1365 - accuracy: 0.9795 - binary_accuracy: 0.97
Epoch 51/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1314 - accuracy: 0.9794 - binary_accuracy: 0.97
Epoch 52/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1321 - accuracy: 0.9799 - binary_accuracy: 0.97
Epoch 53/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1410 - accuracy: 0.9782 - binary_accuracy: 0.97
Epoch 54/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1347 - accuracy: 0.9796 - binary_accuracy: 0.97
Epoch 55/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1293 - accuracy: 0.9802 - binary_accuracy: 0.98
Epoch 56/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1312 - accuracy: 0.9794 - binary_accuracy: 0.97
Epoch 57/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1345 - accuracy: 0.9792 - binary_accuracy: 0.97
Epoch 58/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1305 - accuracy: 0.9797 - binary_accuracy: 0.97
Epoch 59/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1318 - accuracy: 0.9799 - binary_accuracy: 0.97
Epoch 60/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1305 - accuracy: 0.9802 - binary_accuracy: 0.98
Epoch 61/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1337 - accuracy: 0.9797 - binary_accuracy: 0.97
Epoch 62/500
596/596 [==============================] - 19s 33ms/step - loss: 0.1275 - accuracy: 0.9800 - binary_accuracy: 0.98
Epoch 63/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1305 - accuracy: 0.9802 - binary_accuracy: 0.98
Epoch 64/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1274 - accuracy: 0.9807 - binary_accuracy: 0.98
Epoch 65/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1304 - accuracy: 0.9803 - binary_accuracy: 0.98
Epoch 66/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1261 - accuracy: 0.9804 - binary_accuracy: 0.98
Epoch 67/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1258 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 68/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1270 - accuracy: 0.9806 - binary_accuracy: 0.98
Epoch 69/500
596/596 [==============================] - 19s 31ms/step - loss: 0.1261 - accuracy: 0.9804 - binary_accuracy: 0.98
Epoch 70/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1276 - accuracy: 0.9804 - binary_accuracy: 0.98
Epoch 71/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1266 - accuracy: 0.9807 - binary_accuracy: 0.98
Epoch 72/500
596/596 [==============================] - 19s 32ms/step - loss: 0.1242 - accuracy: 0.9809 - binary_accuracy: 0.98
Epoch 73/500
596/596 [==============================] - 16s 28ms/step - loss: 0.1255 - accuracy: 0.9807 - binary_accuracy: 0.98
Epoch 74/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1252 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 75/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1261 - accuracy: 0.9806 - binary_accuracy: 0.98
Epoch 76/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1232 - accuracy: 0.9809 - binary_accuracy: 0.98
Epoch 77/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1263 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 78/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1260 - accuracy: 0.9806 - binary_accuracy: 0.98
Epoch 79/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1244 - accuracy: 0.9810 - binary_accuracy: 0.98
Epoch 80/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1237 - accuracy: 0.9807 - binary_accuracy: 0.98
Epoch 81/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1255 - accuracy: 0.9809 - binary_accuracy: 0.98
Epoch 82/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1243 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 83/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1237 - accuracy: 0.9809 - binary_accuracy: 0.98
Epoch 84/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1251 - accuracy: 0.9806 - binary_accuracy: 0.98
Epoch 85/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1243 - accuracy: 0.9806 - binary_accuracy: 0.98
Epoch 86/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1237 - accuracy: 0.9810 - binary_accuracy: 0.98
Epoch 87/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1224 - accuracy: 0.9811 - binary_accuracy: 0.98
Epoch 88/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1250 - accuracy: 0.9805 - binary_accuracy: 0.98
Epoch 89/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1222 - accuracy: 0.9811 - binary_accuracy: 0.98
Epoch 90/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1239 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 91/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1201 - accuracy: 0.9808 - binary_accuracy: 0.98
Epoch 92/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1251 - accuracy: 0.9805 - binary_accuracy: 0.98
Epoch 93/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1218 - accuracy: 0.9809 - binary_accuracy: 0.98
Epoch 94/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1194 - accuracy: 0.9816 - binary_accuracy: 0.98
Epoch 95/500
596/596 [==============================] - 16s 27ms/step - loss: 0.1197 - accuracy: 0.9811 - binary_accuracy: 0.98
Epoch 96/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1224 - accuracy: 0.9812 - binary_accuracy: 0.98
Epoch 97/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1207 - accuracy: 0.9814 - binary_accuracy: 0.98
Epoch 98/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1174 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 99/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1150 - accuracy: 0.9818 - binary_accuracy: 0.98
Epoch 100/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1191 - accuracy: 0.9811 - binary_accuracy: 0.98
Epoch 101/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1213 - accuracy: 0.9812 - binary_accuracy: 0.98
Epoch 102/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1192 - accuracy: 0.9818 - binary_accuracy: 0.98
Epoch 103/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1198 - accuracy: 0.9815 - binary_accuracy: 0.98
Epoch 104/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1182 - accuracy: 0.9814 - binary_accuracy: 0.98
Epoch 105/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1191 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 106/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1203 - accuracy: 0.9815 - binary_accuracy: 0.98
Epoch 107/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1192 - accuracy: 0.9815 - binary_accuracy: 0.98
Epoch 108/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1186 - accuracy: 0.9817 - binary_accuracy: 0.98
Epoch 109/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1161 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 110/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1166 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 111/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1181 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 112/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1184 - accuracy: 0.9816 - binary_accuracy: 0.98
Epoch 113/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1190 - accuracy: 0.9812 - binary_accuracy: 0.98
Epoch 114/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1180 - accuracy: 0.9817 - binary_accuracy: 0.98
Epoch 115/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1136 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 116/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1141 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 117/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1166 - accuracy: 0.9818 - binary_accuracy: 0.98
Epoch 118/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1130 - accuracy: 0.9827 - binary_accuracy: 0.98
Epoch 119/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1177 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 120/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1149 - accuracy: 0.9820 - binary_accuracy: 0.98
Epoch 121/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1165 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 122/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1102 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 123/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1135 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 124/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1130 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 125/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1144 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 126/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1117 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 127/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1110 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 128/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1147 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 129/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1152 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 130/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1189 - accuracy: 0.9815 - binary_accuracy: 0.98
Epoch 131/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1156 - accuracy: 0.9821 - binary_accuracy: 0.98
Epoch 132/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1152 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 133/500
596/596 [==============================] - 17s 28ms/step - loss: 0.1099 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 134/500
596/596 [==============================] - 22s 37ms/step - loss: 0.1131 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 135/500
596/596 [==============================] - 23s 38ms/step - loss: 0.1112 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 136/500
596/596 [==============================] - 23s 39ms/step - loss: 0.1136 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 137/500
596/596 [==============================] - 17s 28ms/step - loss: 0.1112 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 138/500
596/596 [==============================] - 16s 27ms/step - loss: 0.1138 - accuracy: 0.9820 - binary_accuracy: 0.98
Epoch 139/500
596/596 [==============================] - 16s 27ms/step - loss: 0.1162 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 140/500
596/596 [==============================] - 16s 27ms/step - loss: 0.1129 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 141/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1106 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 142/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1085 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 143/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1070 - accuracy: 0.9829 - binary_accuracy: 0.98
Epoch 144/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1109 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 145/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1152 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 146/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1085 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 147/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1130 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 148/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1122 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 149/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1104 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 150/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1090 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 151/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1062 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 152/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1062 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 153/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1133 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 154/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1139 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 155/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1135 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 156/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1078 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 157/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1064 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 158/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1058 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 159/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1102 - accuracy: 0.9824 - binary_accuracy: 0.98
Epoch 160/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1083 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 161/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1074 - accuracy: 0.9827 - binary_accuracy: 0.98
Epoch 162/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1098 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 163/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1065 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 164/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1068 - accuracy: 0.9829 - binary_accuracy: 0.98
Epoch 165/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1071 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 166/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1104 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 167/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1084 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 168/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1056 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 169/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1049 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 170/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1058 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 171/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1093 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 172/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1040 - accuracy: 0.9829 - binary_accuracy: 0.98
Epoch 173/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1102 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 174/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1083 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 175/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1023 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 176/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1043 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 177/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1048 - accuracy: 0.9827 - binary_accuracy: 0.98
Epoch 178/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1066 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 179/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1126 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 180/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1018 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 181/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1030 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 182/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1035 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 183/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1051 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 184/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1049 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 185/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1117 - accuracy: 0.9819 - binary_accuracy: 0.98
Epoch 186/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1087 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 187/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1052 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 188/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1031 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 189/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1048 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 190/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1054 - accuracy: 0.9822 - binary_accuracy: 0.98
Epoch 191/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1058 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 192/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1046 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 193/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1075 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 194/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1049 - accuracy: 0.9827 - binary_accuracy: 0.98
Epoch 195/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1012 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 196/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1003 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 197/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1047 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 198/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1083 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 199/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1028 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 200/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1067 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 201/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1039 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 202/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1079 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 203/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1001 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 204/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1075 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 205/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1028 - accuracy: 0.9830 - binary_accuracy: 0.98
Epoch 206/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1042 - accuracy: 0.9829 - binary_accuracy: 0.98
Epoch 207/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1059 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 208/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1063 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 209/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1074 - accuracy: 0.9826 - binary_accuracy: 0.98
Epoch 210/500
596/596 [==============================] - 16s 26ms/step - loss: 0.0995 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 211/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0972 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 212/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1042 - accuracy: 0.9825 - binary_accuracy: 0.98
Epoch 213/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0983 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 214/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1012 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 215/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1011 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 216/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1046 - accuracy: 0.9829 - binary_accuracy: 0.98
Epoch 217/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1028 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 218/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0993 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 219/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1058 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 220/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1027 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 221/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1010 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 222/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0999 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 223/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0989 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 224/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1022 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 225/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1019 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 226/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1005 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 227/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1017 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 228/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0993 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 229/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0976 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 230/500
596/596 [==============================] - 16s 26ms/step - loss: 0.1032 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 231/500
596/596 [==============================] - 15s 26ms/step - loss: 0.1007 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 232/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0999 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 233/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1012 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 234/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1069 - accuracy: 0.9823 - binary_accuracy: 0.98
Epoch 235/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1016 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 236/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1012 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 237/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1008 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 238/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1010 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 239/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1009 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 240/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0977 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 241/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1003 - accuracy: 0.9833 - binary_accuracy: 0.98
Epoch 242/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1021 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 243/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1002 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 244/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1023 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 245/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0994 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 246/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0979 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 247/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0985 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 248/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1001 - accuracy: 0.9839 - binary_accuracy: 0.98
Epoch 249/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0954 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 250/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1023 - accuracy: 0.9834 - binary_accuracy: 0.98
Epoch 251/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1031 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 252/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0985 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 253/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0987 - accuracy: 0.9841 - binary_accuracy: 0.98
Epoch 254/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0998 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 255/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0981 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 256/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0989 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 257/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0974 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 258/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0969 - accuracy: 0.9839 - binary_accuracy: 0.98
Epoch 259/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0951 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 260/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0991 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 261/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0978 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 262/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1025 - accuracy: 0.9832 - binary_accuracy: 0.98
Epoch 263/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1015 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 264/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0998 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 265/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0987 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 266/500
596/596 [==============================] - 14s 24ms/step - loss: 0.1032 - accuracy: 0.9831 - binary_accuracy: 0.98
Epoch 267/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0994 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 268/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0971 - accuracy: 0.9839 - binary_accuracy: 0.98
Epoch 269/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0941 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 270/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0960 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 271/500
596/596 [==============================] - 16s 27ms/step - loss: 0.0991 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 272/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0987 - accuracy: 0.9835 - binary_accuracy: 0.98
Epoch 273/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0966 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 274/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0996 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 275/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0936 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 276/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0968 - accuracy: 0.9841 - binary_accuracy: 0.98
Epoch 277/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0939 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 278/500
596/596 [==============================] - 19s 31ms/step - loss: 0.0997 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 279/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0998 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 280/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0976 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 281/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0973 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 282/500
596/596 [==============================] - 19s 31ms/step - loss: 0.0970 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 283/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0985 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 284/500
596/596 [==============================] - 19s 31ms/step - loss: 0.0990 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 285/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0931 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 286/500
596/596 [==============================] - 18s 30ms/step - loss: 0.0946 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 287/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0951 - accuracy: 0.9841 - binary_accuracy: 0.98
Epoch 288/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0911 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 289/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0957 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 290/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0959 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 291/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0977 - accuracy: 0.9841 - binary_accuracy: 0.98
Epoch 292/500
596/596 [==============================] - 19s 31ms/step - loss: 0.0942 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 293/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0984 - accuracy: 0.9836 - binary_accuracy: 0.98
Epoch 294/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0929 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 295/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0985 - accuracy: 0.9839 - binary_accuracy: 0.98
Epoch 296/500
596/596 [==============================] - 18s 30ms/step - loss: 0.0967 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 297/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0904 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 298/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0936 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 299/500
596/596 [==============================] - 18s 31ms/step - loss: 0.0948 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 300/500
596/596 [==============================] - 17s 29ms/step - loss: 0.0930 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 301/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0953 - accuracy: 0.9841 - binary_accuracy: 0.98
Epoch 302/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0930 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 303/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0931 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 304/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0957 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 305/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0947 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 306/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0935 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 307/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0939 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 308/500
596/596 [==============================] - 15s 24ms/step - loss: 0.1000 - accuracy: 0.9839 - binary_accuracy: 0.98
Epoch 309/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0977 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 310/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0925 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 311/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0953 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 312/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0962 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 313/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0914 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 314/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0919 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 315/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0932 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 316/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0977 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 317/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0905 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 318/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0916 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 319/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0948 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 320/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0945 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 321/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0940 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 322/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0973 - accuracy: 0.9840 - binary_accuracy: 0.98
Epoch 323/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0942 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 324/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0934 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 325/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0897 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 326/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0920 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 327/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0930 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 328/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0931 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 329/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0972 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 330/500
596/596 [==============================] - 15s 25ms/step - loss: 0.1024 - accuracy: 0.9828 - binary_accuracy: 0.98
Epoch 331/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0941 - accuracy: 0.9843 - binary_accuracy: 0.98
Epoch 332/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0946 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 333/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0880 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 334/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0959 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 335/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0928 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 336/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0905 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 337/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0906 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 338/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0998 - accuracy: 0.9837 - binary_accuracy: 0.98
Epoch 339/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0921 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 340/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0931 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 341/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0912 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 342/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0919 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 343/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0918 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 344/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0896 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 345/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0942 - accuracy: 0.9842 - binary_accuracy: 0.98
Epoch 346/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0907 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 347/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0909 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 348/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0916 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 349/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0911 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 350/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0910 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 351/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0908 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 352/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0909 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 353/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0927 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 354/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0916 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 355/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0991 - accuracy: 0.9838 - binary_accuracy: 0.98
Epoch 356/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0911 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 357/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0922 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 358/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0903 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 359/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0932 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 360/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0906 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 361/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0910 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 362/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0919 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 363/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0882 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 364/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0920 - accuracy: 0.9844 - binary_accuracy: 0.98
Epoch 365/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0942 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 366/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0935 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 367/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0951 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 368/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0894 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 369/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0910 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 370/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0904 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 371/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0890 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 372/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0895 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 373/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0943 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 374/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0923 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 375/500
596/596 [==============================] - 15s 26ms/step - loss: 0.0904 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 376/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0895 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 377/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0891 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 378/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0897 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 379/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0906 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 380/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0918 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 381/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0886 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 382/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0947 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 383/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0910 - accuracy: 0.9848 - binary_accuracy: 0.98
Epoch 384/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0923 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 385/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0924 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 386/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0891 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 387/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0893 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 388/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0921 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 389/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0912 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 390/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0905 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 391/500
596/596 [==============================] - 14s 24ms/step - loss: 0.0883 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 392/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0884 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 393/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0885 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 394/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0928 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 395/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0873 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 396/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0913 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 397/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0894 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 398/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0896 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 399/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0893 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 400/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0900 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 401/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0907 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 402/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0886 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 403/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0921 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 404/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0930 - accuracy: 0.9845 - binary_accuracy: 0.98
Epoch 405/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0918 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 406/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0894 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 407/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0902 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 408/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0897 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 409/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0839 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 410/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0906 - accuracy: 0.9850 - binary_accuracy: 0.98
Epoch 411/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0941 - accuracy: 0.9846 - binary_accuracy: 0.98
Epoch 412/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0865 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 413/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0857 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 414/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0897 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 415/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0877 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 416/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0864 - accuracy: 0.9858 - binary_accuracy: 0.98
Epoch 417/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0932 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 418/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0898 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 419/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0901 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 420/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0860 - accuracy: 0.9857 - binary_accuracy: 0.98
Epoch 421/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0895 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 422/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0881 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 423/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0881 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 424/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0864 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 425/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0900 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 426/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0886 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 427/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0878 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 428/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0859 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 429/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0863 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 430/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0862 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 431/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0918 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 432/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0870 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 433/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0906 - accuracy: 0.9849 - binary_accuracy: 0.98
Epoch 434/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0903 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 435/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0848 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 436/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0876 - accuracy: 0.9857 - binary_accuracy: 0.98
Epoch 437/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0888 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 438/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0862 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 439/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0877 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 440/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0899 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 441/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0901 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 442/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0910 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 443/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0875 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 444/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0871 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 445/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0865 - accuracy: 0.9858 - binary_accuracy: 0.98
Epoch 446/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0908 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 447/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0917 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 448/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0894 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 449/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0863 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 450/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0865 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 451/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0851 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 452/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0876 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 453/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0898 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 454/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0869 - accuracy: 0.9857 - binary_accuracy: 0.98
Epoch 455/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0873 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 456/500
596/596 [==============================] - 16s 27ms/step - loss: 0.0884 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 457/500
596/596 [==============================] - 21s 36ms/step - loss: 0.0907 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 458/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0845 - accuracy: 0.9857 - binary_accuracy: 0.98
Epoch 459/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0881 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 460/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0838 - accuracy: 0.9861 - binary_accuracy: 0.98
Epoch 461/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0909 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 462/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0871 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 463/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0873 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 464/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0881 - accuracy: 0.9857 - binary_accuracy: 0.98
Epoch 465/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0860 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 466/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0814 - accuracy: 0.9864 - binary_accuracy: 0.98
Epoch 467/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0848 - accuracy: 0.9861 - binary_accuracy: 0.98
Epoch 468/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0930 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 469/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0869 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 470/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0912 - accuracy: 0.9847 - binary_accuracy: 0.98
Epoch 471/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0876 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 472/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0866 - accuracy: 0.9858 - binary_accuracy: 0.98
Epoch 473/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0815 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 474/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0892 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 475/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0877 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 476/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0880 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 477/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0851 - accuracy: 0.9860 - binary_accuracy: 0.98
Epoch 478/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0887 - accuracy: 0.9854 - binary_accuracy: 0.98
Epoch 479/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0874 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 480/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0841 - accuracy: 0.9858 - binary_accuracy: 0.98
Epoch 481/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0918 - accuracy: 0.9851 - binary_accuracy: 0.98
Epoch 482/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0881 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 483/500
596/596 [==============================] - 16s 26ms/step - loss: 0.0849 - accuracy: 0.9860 - binary_accuracy: 0.98
Epoch 484/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0867 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 485/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0850 - accuracy: 0.9860 - binary_accuracy: 0.98
Epoch 486/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0874 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 487/500
596/596 [==============================] - 15s 24ms/step - loss: 0.0866 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 488/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0863 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 489/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0882 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 490/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0878 - accuracy: 0.9852 - binary_accuracy: 0.98
Epoch 491/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0871 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 492/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0819 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 493/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0833 - accuracy: 0.9863 - binary_accuracy: 0.98
Epoch 494/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0924 - accuracy: 0.9853 - binary_accuracy: 0.98
Epoch 495/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0901 - accuracy: 0.9855 - binary_accuracy: 0.98
Epoch 496/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0872 - accuracy: 0.9859 - binary_accuracy: 0.98
Epoch 497/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0863 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 498/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0871 - accuracy: 0.9856 - binary_accuracy: 0.98
Epoch 499/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0822 - accuracy: 0.9863 - binary_accuracy: 0.98
Epoch 500/500
596/596 [==============================] - 15s 25ms/step - loss: 0.0863 - accuracy: 0.9854 - binary_accuracy: 0.98
It took 8134 seconds to train the model!
'''
file = open('model2.txt','wb')
pickle.dump(model,file)
pickle.dump(history,file)
file.close()
file = open('test_features2.txt','rb')
testseries_features = pickle.load(file)
file.close()
"""classification report from here"""
train_data_check = np.ravel(model.predict(timeseries_features, batch_size=5000,verbose=1)>0.96).astype(int)
print(train_data_check)
print(precision_score(timeseries_features_label, train_data_check))
print(recall_score(timeseries_features_label, train_data_check))
print(f1_score(timeseries_features_label, train_data_check))

predict_flag = (np.ravel(model.predict(testseries_features,batch_size=5000,verbose=1))>0.9999999).astype(int)
print(predict_flag)
print(sum(predict_flag)/len(predict_flag))

data_features_diff = len(test_data) - len(testseries_features)
print(data_features_diff)
data_features_diff_avg = int(data_features_diff / len(testseries_all))
print(data_features_diff_avg)

last_index = 0
predict_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(testseries_all)):
    next_index += len(testseries_all[i]) - data_features_diff_avg
    predict_new = np.concatenate((predict_new, predict_flag[last_index : next_index]))
    print(next_index)
    last_index = next_index
    if i != len(testseries_all)-1:
        predict_new = np.concatenate((predict_new,np.zeros(data_features_diff_avg)))
print(len(predict_new))
assert(len(predict_new) == len(test_data))
predict_new = predict_new.astype(int)
predict_df = pd.DataFrame({'KPI ID': test_data['KPI ID'],
                         'timestamp': test_data['timestamp'],
                         'predict': predict_new})
predict_df.to_csv('predictn.csv', index=False)
print(sum(train_data_check)/len(train_data_check))
testseries_scaled = []
for i in range(len(testseries_all)):
    testseries_scaled.append(minmax_scale(testseries_all[i]))
print(testseries_scaled)
print(len(testseries_scaled))
#print(scaler_list)
id_t1, id_t2 = np.unique(train_data['KPI ID'], return_index=True)
print(id_t1,id_t2)
id_t3, id_t4 = np.unique(test_data['KPI ID'], return_index=True)
print(id_t3,id_t4)


"""test phase"""


def get_test_features(time_series, Windows):
    data = []

    start_point = 2 * max(Windows) - 1
    start_accum = sum(time_series[0:start_point])

    # features from tsa models
    # time_series_SARIMA_residuals = get_feature_SARIMA_residuals(time_series)
    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    # features from tsa models for time series logarithm
    time_series_logs = get_feature_logs(time_series)

    for i in np.arange(start_point, len(time_series)):
        # the datum to put into the data pool
        datum = []

        # fill the datum with f01-f09
        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i + 1)

        # f01-f04: residuals
        # datum.append(time_series_SARIMA_residuals[i])
        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])
        # f05: logarithm
        datum.append(time_series_logs[i])

        # f06: diff
        datum.append(diff_plain)
        # f07: diff percentage
        datum.append(diff_plain / (time_series[i - 1] + 1e-8))  # to avoid 0, plus 1e-10
        # f08: diff of diff - derivative
        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))
        # f09: diff of accumulated mean and current value
        datum.append(time_series[i] - mean_accum)

        # fill the datum with features related to windows
        # loop over different windows size to fill the datum
        for k in Windows:
            mean_w = np.mean(time_series[i - k + 1:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k + 1:i + 1]) - mean_w) ** 2)
            # var_w = np.var(time_series[i-k:i+1])

            mean_w_and_1 = mean_w + (time_series[i - k] - time_series[i]) / k
            var_w_and_1 = np.mean((np.asarray(time_series[i - k:i]) - mean_w_and_1) ** 2)
            # mean_w_and_1 = np.mean(time_series[i-k-1:i])
            # var_w_and_1 = np.var(time_series[i-k-1:i])

            mean_2w = np.mean(time_series[i - 2 * k + 1:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k + 1:i - k + 1]) - mean_2w) ** 2)
            # var_2w = np.var(time_series[i-2*k:i-k+1])

            # diff of sliding windows
            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            # diff of jumping windows
            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            # f1
            datum.append(mean_w)  # [0:2] is [0,1]
            # f2
            datum.append(var_w)
            # f3
            datum.append(diff_mean_1)
            # f4
            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-8))
            # f5
            datum.append(diff_var_1)
            # f6
            datum.append(diff_var_1 / (var_w_and_1 + 1e-8))
            # f7
            datum.append(diff_mean_w)
            # f8
            datum.append(diff_mean_w / (mean_2w + 1e-8))
            # f9
            datum.append(diff_var_w)
            # f10
            datum.append(diff_var_w / (var_2w + 1e-8))

            # diff of sliding/jumping windows and current value
            # f11
            datum.append(time_series[i] - mean_w_and_1)
            # f12
            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))

    return data


testseries_features = None
for i in range(len(testseries_scaled)):
    print(i, len(testseries_scaled[i]))
    features_temp = get_test_features(testseries_scaled[i], W)
    features_temp = scaler_list[i].transform(features_temp)
    if i == 0:
        testseries_features = features_temp
    else:
        testseries_features = np.concatenate((testseries_features, features_temp), axis=0)

print(testseries_features.shape)
print(testseries_features)
predict_flag = (np.ravel(model.predict(testseries_features,batch_size=5000,verbose=1))>0.97).astype(int)
print(predict_flag)
print(sum(predict_flag)/len(predict_flag))

last_index = 0
predict_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(testseries_all)):
    next_index += len(testseries_all[i]) - data_features_diff_avg
    predict_new = np.concatenate((predict_new, predict_flag[last_index : next_index]))
    print(next_index)
    last_index = next_index
    if i != len(testseries_all)-1:
        predict_new = np.concatenate((predict_new,np.zeros(data_features_diff_avg)))
print(len(predict_new))
assert(len(predict_new) == len(test_data))
predict_new = predict_new.astype(int)
predict_df = pd.DataFrame({'KPI ID': test_data['KPI ID'],
                         'timestamp': test_data['timestamp'],
                         'predict': predict_new})
predict_df.to_csv('predictn2.csv', index=False)

"""do the feature engineering---windows """


def new_get_timeseries_features(time_series, time_series_label, Windows, delay):
    data = []
    data_label = []
    data_label_vital = []

    start_point = 2 * max(Windows) - 1
    start_accum = sum(time_series[0:start_point])

    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    for i in np.arange(start_point, len(time_series)):
        datum = []
        datum_label = time_series_label[i]

        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i + 1)

        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])

        datum.append(time_series[i])

        datum.append(diff_plain)

        datum.append(diff_plain / (time_series[i - 1] + 1e-8))

        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))

        datum.append(time_series[i] - mean_accum)

        for k in Windows:
            mean_w = np.mean(time_series[i - k + 1:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k + 1:i + 1]) - mean_w) ** 2)

            mean_w_and_1 = mean_w + (time_series[i - k] - time_series[i]) / k
            var_w_and_1 = np.mean((np.asarray(time_series[i - k:i]) - mean_w_and_1) ** 2)

            mean_2w = np.mean(time_series[i - 2 * k + 1:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k + 1:i - k + 1]) - mean_2w) ** 2)

            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            datum.append(mean_w)

            datum.append(var_w)

            datum.append(diff_mean_1)

            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-8))

            datum.append(diff_var_1)

            datum.append(diff_var_1 / (var_w_and_1 + 1e-8))

            datum.append(diff_mean_w)

            datum.append(diff_mean_w / (mean_2w + 1e-8))

            datum.append(diff_var_w)

            datum.append(diff_var_w / (var_2w + 1e-8))

            datum.append(time_series[i] - mean_w_and_1)

            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))
        data_label.append(np.asarray(datum_label))

        if datum_label == 1 and sum(time_series_label[i - delay:i]) < delay:
            data_label_vital.append(np.asarray(1))
        else:
            data_label_vital.append(np.asarray(0))

    return data, data_label, data_label_vital


scaler_list_new = []
timeseries_features_new = []
timeseries_features_label_new = []
timeseries_features_label_vital_new = []

for i in range(len(timeseries_all)):
    print(i, len(timeseries_all[i]), len(scaler_list_new), len(timeseries_features_new),
          len(timeseries_features_label_new),
          len(timeseries_features_label_vital_new))
    features_temp, label_temp, label_vital_temp = new_get_timeseries_features(timeseries_all[i], timeseries_label[i], W,
                                                                              delay)
    assert (len(features_temp) == len(label_temp))
    assert (len(label_temp) == len(label_vital_temp))
    scaler_temp = StandardScaler()
    features_temp = scaler_temp.fit_transform(features_temp)
    scaler_list_new.append(scaler_temp)
    if i == 0:
        timeseries_features_new = features_temp
    else:
        timeseries_features_new = np.concatenate((timeseries_features_new, features_temp), axis=0)

    timeseries_features_label_new = timeseries_features_label_new + label_temp
    timeseries_features_label_vital_new = timeseries_features_label_vital_new + label_vital_temp

file = open('train_feature.txt','wb')
pickle.dump(timeseries_features_new,file)
pickle.dump(timeseries_features_label_new,file)
pickle.dump(timeseries_features_label_vital_new,file)
pickle.dump(scaler_list_new,file)
file.close()


def new_get_test_features(time_series, Windows):
    data = []

    start_point = 2 * max(Windows) - 1
    start_accum = sum(time_series[0:start_point])

    # features from tsa models
    # time_series_SARIMA_residuals = get_feature_SARIMA_residuals(time_series)
    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    for i in np.arange(start_point, len(time_series)):
        # the datum to put into the data pool
        datum = []

        # fill the datum with f01-f09
        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i + 1)

        # f01-f04: residuals
        # datum.append(time_series_SARIMA_residuals[i])
        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])
        # f05: logarithm
        datum.append(time_series[i])

        # f06: diff
        datum.append(diff_plain)
        # f07: diff percentage
        datum.append(diff_plain / (time_series[i - 1] + 1e-8))  # to avoid 0, plus 1e-10
        # f08: diff of diff - derivative
        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))
        # f09: diff of accumulated mean and current value
        datum.append(time_series[i] - mean_accum)

        # fill the datum with features related to windows
        # loop over different windows size to fill the datum
        for k in Windows:
            mean_w = np.mean(time_series[i - k + 1:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k + 1:i + 1]) - mean_w) ** 2)
            # var_w = np.var(time_series[i-k:i+1])

            mean_w_and_1 = mean_w + (time_series[i - k] - time_series[i]) / k
            var_w_and_1 = np.mean((np.asarray(time_series[i - k:i]) - mean_w_and_1) ** 2)
            # mean_w_and_1 = np.mean(time_series[i-k-1:i])
            # var_w_and_1 = np.var(time_series[i-k-1:i])

            mean_2w = np.mean(time_series[i - 2 * k + 1:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k + 1:i - k + 1]) - mean_2w) ** 2)
            # var_2w = np.var(time_series[i-2*k:i-k+1])

            # diff of sliding windows
            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            # diff of jumping windows
            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            # f1
            datum.append(mean_w)  # [0:2] is [0,1]
            # f2
            datum.append(var_w)
            # f3
            datum.append(diff_mean_1)
            # f4
            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-8))
            # f5
            datum.append(diff_var_1)
            # f6
            datum.append(diff_var_1 / (var_w_and_1 + 1e-8))
            # f7
            datum.append(diff_mean_w)
            # f8
            datum.append(diff_mean_w / (mean_2w + 1e-8))
            # f9
            datum.append(diff_var_w)
            # f10
            datum.append(diff_var_w / (var_2w + 1e-8))

            # diff of sliding/jumping windows and current value
            # f11
            datum.append(time_series[i] - mean_w_and_1)
            # f12
            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))

    return data


testseries_features_new = []
for i in range(len(testseries_all)):
    print(i, len(testseries_all[i]), len(testseries_features_new))
    features_temp = new_get_test_features(testseries_all[i], W)
    features_temp = scaler_list_new[i].transform(features_temp)
    if i == 0:
        testseries_features_new = features_temp
    else:
        testseries_features_new = np.concatenate((testseries_features_new, features_temp), axis=0)

print(testseries_features_new.shape)
file = open('test_feature.txt','wb')
pickle.dump(testseries_features_new,file)
file.close()
timeseries_features_label_new = np.array(timeseries_features_label_new)
timeseries_features_label_vital_new = np.array(timeseries_features_label_vital_new)

ratio = round((len(timeseries_features_label_new) - sum(timeseries_features_label_new)) * 0.5 / sum(timeseries_features_label_new))
print(ratio)
non_anomaly = np.ones(len(timeseries_features_label_new)) - timeseries_features_label_new
print(non_anomaly,non_anomaly.shape)
sample_ratio = (4*ratio) * timeseries_features_label_vital_new + non_anomaly
print(sample_ratio,sum(sample_ratio))
sample_ratio = sample_ratio + ratio * timeseries_features_label_new
print(sample_ratio,sum(sample_ratio))

print('Keras: start to train DNN!')
start_time = time.time()
history = model.fit(timeseries_features_new, timeseries_features_label_new, epochs=20, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))
train_data_check = np.ravel(model.predict(timeseries_features, batch_size=5000,verbose=1)>0.95).astype(int)
print(train_data_check)
print(precision_score(timeseries_features_label, train_data_check))
print(recall_score(timeseries_features_label, train_data_check))
print(f1_score(timeseries_features_label, train_data_check))
predict_flag = (np.ravel(model.predict(testseries_features_new,batch_size=5000,verbose=1))>0.999).astype(int)
print(predict_flag)
print(sum(predict_flag)/len(predict_flag))
last_index = 0
predict_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(testseries_all)):
    next_index += len(testseries_all[i]) - data_features_diff_avg
    predict_new = np.concatenate((predict_new, predict_flag[last_index : next_index]))
    print(next_index)
    last_index = next_index
    if i != len(testseries_all)-1:
        predict_new = np.concatenate((predict_new,np.zeros(data_features_diff_avg)))
print(len(predict_new))
assert(len(predict_new) == len(test_data))
predict_new = predict_new.astype(int)
predict_df = pd.DataFrame({'KPI ID': test_data['KPI ID'],
                         'timestamp': test_data['timestamp'],
                         'predict': predict_new})
predict_df.to_csv('predictn4.csv', index=False)
ratio = round((len(timeseries_features_label_new) - sum(timeseries_features_label_new)) * 0.2 / sum(timeseries_features_label_new))
print(ratio)
non_anomaly = np.ones(len(timeseries_features_label_new)) - timeseries_features_label_new
print(non_anomaly,non_anomaly.shape)
sample_ratio = (4*ratio) * timeseries_features_label_vital_new + non_anomaly
print(sample_ratio,sum(sample_ratio))
sample_ratio = sample_ratio + ratio * timeseries_features_label_new
print(sample_ratio,sum(sample_ratio))
"""DNN training"""
print('Keras: start to train DNN!')
start_time = time.time()
history = model.fit(timeseries_features_new, timeseries_features_label_new, epochs=20, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))

train_data_check = np.ravel(model.predict(timeseries_features, batch_size=5000,verbose=1)>0.9).astype(int)
print(train_data_check)
print(precision_score(timeseries_features_label, train_data_check))
print(recall_score(timeseries_features_label, train_data_check))
print(f1_score(timeseries_features_label, train_data_check))
predict_flag = (np.ravel(model.predict(testseries_features_new,batch_size=5000,verbose=1))>0.99).astype(int)
print(predict_flag)
print(sum(predict_flag)/len(predict_flag))
last_index = 0
predict_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(testseries_all)):
    next_index += len(testseries_all[i]) - data_features_diff_avg
    predict_new = np.concatenate((predict_new, predict_flag[last_index : next_index]))
    print(next_index)
    last_index = next_index
    if i != len(testseries_all)-1:
        predict_new = np.concatenate((predict_new,np.zeros(data_features_diff_avg)))
print(len(predict_new))
assert(len(predict_new) == len(test_data))
predict_new = predict_new.astype(int)
predict_df = pd.DataFrame({'KPI ID': test_data['KPI ID'],
                         'timestamp': test_data['timestamp'],
                         'predict': predict_new})
predict_df.to_csv('predictn5.csv', index=False)

m = Sequential()
m.add(Dense(128, input_dim = 128))
# m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(Dropout(0.5))

m.add(Dense(64))
# m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(Dropout(0.5))

m.add(Dense(1))
# m.add(BatchNormalization())
m.add(Activation('sigmoid'))


m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
ratio = round((len(timeseries_features_label_new) - sum(timeseries_features_label_new)) * 0.8 / sum(timeseries_features_label_new))
print(ratio)
non_anomaly = np.ones(len(timeseries_features_label_new)) - timeseries_features_label_new
print(non_anomaly,non_anomaly.shape)
sample_ratio = (4*ratio) * timeseries_features_label_vital_new + non_anomaly
print(sample_ratio,sum(sample_ratio))
sample_ratio = sample_ratio + ratio * timeseries_features_label_new
print(sample_ratio,sum(sample_ratio))
"""train dnn"""
print('Keras: start to train DNN!')
start_time = time.time()
h = m.fit(timeseries_features_new, timeseries_features_label_new, epochs=50, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))
p = m.predict(timeseries_features_new, batch_size=5000,verbose=1)
train_data_check = np.ravel(p>0.95).astype(int)
print(train_data_check)
print(precision_score(timeseries_features_label, train_data_check))
print(recall_score(timeseries_features_label, train_data_check))
print(f1_score(timeseries_features_label, train_data_check))
print(len(train_data_check))
print(len(train_data))
last_index = 0
evaluation_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(timeseries_all)):
    next_index += len(timeseries_all[i]) - data_features_diff_avg
    evaluation_new = np.concatenate((evaluation_new, train_data_check[last_index : next_index]))
    print(len(evaluation_new),next_index)
    last_index = next_index
    if i != len(timeseries_all)-1:
        evaluation_new = np.concatenate((evaluation_new,np.zeros(data_features_diff_avg)))
print(len(evaluation_new))
assert(len(evaluation_new) == len(train_data))
evaluation_new = evaluation_new.astype(int)
evaluation_df = pd.DataFrame({'KPI ID': train_data['KPI ID'],
                         'timestamp': train_data['timestamp'],
                         'predict': evaluation_new})
evaluation_df.to_csv('evaluation.csv', index=False)
"""do this in jupyter notebook"""
# !python evaluation.py "../../input/train.csv" "evaluation.csv"
pm_t = m.predict(testseries_features_new,batch_size=5000,verbose=1)
predict_flagm = (np.ravel(pm_t)>0.9985).astype(int)
print(predict_flagm)
print(sum(predict_flagm)/len(predict_flagm))
last_index = 0
predict_new = np.zeros(data_features_diff_avg).astype(int)
next_index = 0
for i in range(len(testseries_all)):
    next_index += len(testseries_all[i]) - data_features_diff_avg
    predict_new = np.concatenate((predict_new, predict_flagm[last_index : next_index]))
    print(next_index)
    last_index = next_index
    if i != len(testseries_all)-1:
        predict_new = np.concatenate((predict_new,np.zeros(data_features_diff_avg)))
print(len(predict_new))
assert(len(predict_new) == len(test_data))
predict_new = predict_new.astype(int)
predict_df = pd.DataFrame({'KPI ID': test_data['KPI ID'],
                         'timestamp': test_data['timestamp'],
                         'predict': predict_new})
predict_df.to_csv('predictw2.csv', index=False)

"""testing"""
test = Sequential()
test.add(Dense(128, input_dim = 128, kernel_regularizer=l1(0.05)))
# test.add(BatchNormalization())
test.add(Activation('relu'))
test.add(Dropout(0.2))

test.add(Dense(128, kernel_regularizer=l1(0.05)))
# test.add(BatchNormalization())
test.add(Activation('relu'))
test.add(Dropout(0.2))

test.add(Dense(1))
# test.add(BatchNormalization())
test.add(Activation('sigmoid'))
# i tried different structure here

test.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

print('Keras: start to train DNN!')
start_time = time.time()
h_t = test.fit(timeseries_features_new, timeseries_features_label_new, epochs=100, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))

t1 = Sequential()
t1.add(Dense(128, input_dim = 128, kernel_regularizer=l1(0.1)))
# t1.add(BatchNormalization())
t1.add(Activation('relu'))
t1.add(Dropout(0.2))

t1.add(Dense(128, kernel_regularizer=l1(0.1)))
# t1.add(BatchNormalization())
t1.add(Activation('relu'))
t1.add(Dropout(0.2))

t1.add(Dense(1))
# t1.add(BatchNormalization())
t1.add(Activation('sigmoid'))


t1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print('Keras: start to train DNN!')
start_time = time.time()
h_t1 = t1.fit(timeseries_features_new, timeseries_features_label_new, epochs=100, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))
print(timeseries_features_new.shape)
print(testseries_features_new.shape)
for i in range(len(scaler_list_new)):
    print(scaler_list_new[i].mean_, scaler_list_new[i].var_)
train_feature = None
index1 = 0
index2 = 0
for i in range(len(scaler_list_new)):
    index2 += len(timeseries_all[i]) - data_features_diff_avg
    temp = timeseries_features_new[index1:index2,]
    print(temp.shape)
    temp = scaler_list_new[i].inverse_transform(temp)
    if i == 0:
        train_feature = temp
    else:
        train_feature = np.concatenate((train_feature, temp), axis = 0)
    index1 = index2
    print(train_feature.shape)
scaler_new = StandardScaler()
train_feature = scaler_new.fit_transform(train_feature)
print(train_feature)
test_feature = None
index1 = 0
index2 = 0
for i in range(len(scaler_list_new)):
    index2 += len(testseries_all[i]) - data_features_diff_avg
    temp = testseries_features_new[index1:index2,]
    print(temp.shape)
    temp = scaler_list_new[i].inverse_transform(temp)
    if i == 0:
        test_feature = temp
    else:
        test_feature = np.concatenate((test_feature, temp), axis = 0)
    index1 = index2
    print(test_feature.shape)


test_feature = scaler_new.transform(test_feature)
print(test_feature)
'''another experiment'''
m = Sequential()
m.add(Dense(512, input_dim = 128))
# m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(Dropout(0.2))

m.add(Dense(256))
# m.add(BatchNormalization())
m.add(Activation('relu'))
m.add(Dropout(0.2))

m.add(Dense(1))
# m.add(BatchNormalization())
m.add(Activation('sigmoid'))


m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

ratio = round((len(timeseries_features_label_new) - sum(timeseries_features_label_new)) * 0.8 / sum(timeseries_features_label_new))
print(ratio)
non_anomaly = np.ones(len(timeseries_features_label_new)) - timeseries_features_label_new
print(non_anomaly,non_anomaly.shape)
sample_ratio = (4*ratio) * timeseries_features_label_vital_new + non_anomaly
print(sample_ratio,sum(sample_ratio))
sample_ratio = sample_ratio + ratio * timeseries_features_label_new
print(sample_ratio,sum(sample_ratio))

print(len(train_feature),len(timeseries_features_label),len(sample_ratio))
print('Keras: start to train DNN!')
start_time = time.time()
h = m.fit(train_feature, timeseries_features_label_new, epochs=10, batch_size=5000, verbose=1,
                   sample_weight=sample_ratio)

end_time = time.time()
print('It took %d seconds to train the model!' %(end_time-start_time))

