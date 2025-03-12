# IMPORTING LIBRARIES

# General Libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")

# Preprocessing Libraries

from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Machine Learning Libraries

import sklearn
import xgboost as xgb
from sklearn import tree
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.io import arff
from sklearn.metrics import roc_curve
from imblearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from xgboost import XGBClassifier

# Defining the working directory

# IMPORTING DATA

input_path = r'C:\Users\jr721\OneDrive\文档\Xinyuan\BP Classification data\Slovak + 2013-2016\2016 RETAIL MERGED.csv'
bank_data = pd.read_csv(input_path)
bank_data.head()
bank_data.info()
print(bank_data.shape)

# Computing the descriptive statistics of our numrerical features

#bank_data.describe()
#print(bank_data.describe())

# Checking Nan presence

#bank_data.isna().sum().max()
#[print(col) for col in bank_data if bank_data[col].isna().sum() > 0]

# Checking for duplicates

#bank_data.duplicated().sum()

# The classes are heavily skewed we need to solve this issue later.

# print(bank_data['Bankrupt?'].value_counts())
# print('-'* 30)
# print('Financially stable: ', round(bank_data['Bankrupt?'].value_counts()[0]/len(bank_data) * 100,2), '% of the dataset')
# print('Financially unstable: ', round(bank_data['Bankrupt?'].value_counts()[1]/len(bank_data) * 100,2), '% of the dataset')

# Checking labels distributions

# print(bank_data['Bankrupt?'].value_counts())
# print(bank_data['Bankrupt?'].unique())


# sns.set_theme(context = 'paper')
#
# plt.figure(figsize = (10,5))
# sns.countplot(data=bank_data, x='Bankrupt?')
# plt.title('Class Distributions \n (0: Fin. Stable || 1: Fin. Unstable)', fontsize=14)
# plt.show()

# Looking at the histograms of numerical data

# axes = bank_data.hist(figsize = (30,60), bins = 50)
# for ax in axes.flatten():
#     # 设置 x 轴标签字体大小
#     ax.set_xlabel(ax.get_xlabel(), fontsize=1)
#
#     # 设置 y 轴标签字体大小
#     ax.set_ylabel(ax.get_ylabel(), fontsize=1)
#
#     # 设置 x 轴刻度字体大小
#     ax.tick_params(axis='x', labelsize=1)
#
#     # 设置 y 轴刻度字体大小
#     ax.tick_params(axis='y', labelsize=1)
#
# for ax in axes.flatten():
#     ax.set_title('')
# # plt.show()
#
# # EDA & VISUALIZATIONS
#
# # Correlation Heatmap (Spearman)
#
# f, ax = plt.subplots(figsize=(30, 25))
# mat = bank_data.corr('spearman')
# mask = np.triu(np.ones_like(mat, dtype=bool))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# ax1 = sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, #annot = True,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# ax1.set_xticklabels(ax.get_xticklabels(), fontsize=3)
# ax1.set_yticklabels(ax.get_yticklabels(), fontsize=3)
# #plt.show()
#
# # Plotting Boxplots of the numerical features
#
# plt.figure(figsize = (20,20))
# ax2 =sns.boxplot(data = bank_data, orient="h")
# ax2.set_title('Bank Data Boxplots', fontsize = 18)
# ax2.set(xscale="log")
# ax2.set_yticklabels(ax.get_yticklabels(), fontsize=3)
# ax2.set_xticklabels(ax.get_yticklabels(), fontsize=3)
# plt.show()

# Outliers removal
def outliers_removal(feature, feature_name, dataset):
    # Identify 25th & 75th quartiles

    # q25, q75 = np.percentile(feature, 25), np.percentile(feature, 75)
    # print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    # feat_iqr = q75 - q25
    # print('iqr: {}'.format(feat_iqr))
    #
    # feat_cut_off = feat_iqr * 1.5
    # feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off
    # print('Cut Off: {}'.format(feat_cut_off))
    # print(feature_name + ' Lower: {}'.format(feat_lower))
    # print(feature_name + ' Upper: {}'.format(feat_upper))
    #
    # outliers = [x for x in feature if x < feat_lower or x > feat_upper]
    # print(feature_name + ' outliers for close to bankruptcy cases: {}'.format(len(outliers)))
    # # print(feature_name + ' outliers:{}'.format(outliers))
    #
    # #dataset = dataset.drop(dataset[(dataset[feature_name] > feat_upper) | (dataset[feature_name] < feat_lower)].index)
    # #print('-' * 65)

    return dataset

for col in bank_data:
    new_df = outliers_removal(bank_data[col], str(col), bank_data)

print(new_df.shape)
# Dividing Data and Labels

labels = new_df['Bankrupt?']
new_df = new_df.drop(['Bankrupt?'], axis = 1)


def log_trans(data):
    for col in data:
        skew = data[col].skew()
        if skew > 0.5 or skew < -0.5:
            data[col] = np.log1p(data[col])
        else:
            continue

    return data


data_norm = log_trans(new_df)

# Splitting Train and Test Data

X_raw,X_test,y_raw,y_test  = train_test_split(data_norm,
                                              labels,
                                              test_size=0.3,
                                              stratify = labels,
                                              random_state = 42)

# Stratified Cross Validation Splitting

sss = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in sss.split(X_raw, y_raw):
    # print("Train:", train_index, "Test:", test_index)
    X_train_sm, X_val_sm = X_raw.iloc[train_index], X_raw.iloc[test_index]
    y_train_sm, y_val_sm = y_raw.iloc[train_index], y_raw.iloc[test_index]
X_train_sm.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_train_sm = imputer.fit_transform(scaler.fit_transform(X_train_sm))

X_train_sm = pd.DataFrame(X_train_sm)
X_train_sm = X_train_sm.reset_index(drop=True)

X_train_sm = X_train_sm.reset_index(drop=True)
y_train_sm = y_train_sm.reset_index(drop=True)

# Check the Distribution of the labels


# Turn into an array
X_train_sm = X_train_sm.values
X_val_sm = X_val_sm.values
y_train_sm = y_train_sm.values
y_val_sm = y_val_sm.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(y_train_sm, return_counts=True)
test_unique_label, test_counts_label = np.unique(y_val_sm, return_counts=True)
print('-' * 84)

print('Label Distributions: \n')
print(train_counts_label / len(y_train_sm))
print(test_counts_label / len(y_val_sm))

unique, counts = np.unique(y_train_sm, return_counts=True)
print(dict(zip(unique, counts)))

# Logistic Regression

# List to append the score and then find the average

accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

accuracy_imb = []
precision_imb = []
recall_imb = []
f1_imb = []
auc_imb = []

log_reg_sm = LogisticRegression()
# log_reg_params = {}
log_reg_params = {"penalty": ['l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'class_weight': ['balanced'],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

for train, val in sss.split(X_train_sm, y_train_sm):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                            rand_log_reg)  # SMOTE happens during Cross Validation not before..
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_log_reg.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])

    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))

for train, val in sss.split(X_train_sm, y_train_sm):
    model_reg_imb = rand_log_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg_imb = rand_log_reg.best_estimator_
    prediction_reg_imb = best_est_reg.predict(X_train_sm[val])

    accuracy_imb.append(model_reg_imb.score(X_train_sm[val], y_train_sm[val]))
    precision_imb.append(precision_score(y_train_sm[val], prediction_reg_imb))
    recall_imb.append(recall_score(y_train_sm[val], prediction_reg_imb))
    f1_imb.append(f1_score(y_train_sm[val], prediction_reg_imb))
    auc_imb.append(roc_auc_score(y_train_sm[val], prediction_reg_imb))

print('---' * 45)
print('')
print('Logistic Regression (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print("auc: {}".format(np.mean(auc_lst_reg)))
print('')
print('---' * 45)

print('---' * 45)
print('')
print('Logistic Regression results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_imb)))
print("precision: {}".format(np.mean(precision_imb)))
print("recall: {}".format(np.mean(recall_imb)))
print("f1: {}".format(np.mean(f1_imb)))
print("auc: {}".format(np.mean(auc_imb)))
print('')
print('---' * 45)

# Decision Tree

# Lists to append the score and then find the average
accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

accuracy_imb = []
precision_imb = []
recall_imb = []
f1_imb = []
auc_imb = []

# Replace Logistic Regression with Decision Tree
tree_clf = DecisionTreeClassifier()

# Define hyperparameters for the Decision Tree
tree_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

# RandomizedSearchCV with Decision Tree
rand_tree_clf = RandomizedSearchCV(DecisionTreeClassifier(), tree_params, n_iter=4)

# SMOTE + Decision Tree (with cross-validation)
for train, val in sss.split(X_train_sm, y_train_sm):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                            rand_tree_clf)
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_tree_clf.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])

    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))

# Decision Tree without SMOTE (with cross-validation)
for train, val in sss.split(X_train_sm, y_train_sm):
    model_reg_imb = rand_tree_clf.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg_imb = rand_tree_clf.best_estimator_
    prediction_reg_imb = best_est_reg_imb.predict(X_train_sm[val])

    accuracy_imb.append(model_reg_imb.score(X_train_sm[val], y_train_sm[val]))
    precision_imb.append(precision_score(y_train_sm[val], prediction_reg_imb))
    recall_imb.append(recall_score(y_train_sm[val], prediction_reg_imb))
    f1_imb.append(f1_score(y_train_sm[val], prediction_reg_imb))
    auc_imb.append(roc_auc_score(y_train_sm[val], prediction_reg_imb))

# Print results for SMOTE
print('---' * 45)
print('')
print('Decision Tree (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print("auc: {}".format(np.mean(auc_lst_reg)))
print('')
print('---' * 45)

# Print results for non-SMOTE
print('---' * 45)
print('')
print('Decision Tree results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_imb)))
print("precision: {}".format(np.mean(precision_imb)))
print("recall: {}".format(np.mean(recall_imb)))
print("f1: {}".format(np.mean(f1_imb)))
print("auc: {}".format(np.mean(auc_imb)))
print('')
print('---' * 45)

# svm

# Lists to append the score and then find the average
accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

accuracy_imb = []
precision_imb = []
recall_imb = []
f1_imb = []
auc_imb = []

# Replace Decision Tree with Support Vector Machine
svm_clf = SVC(probability=True)

# Define hyperparameters for the Support Vector Machine
svm_params = {
    'C': [0.001],
    'kernel': ['rbf'],
    'class_weight': ['balanced',None],
    'gamma': ['scale']
}

# RandomizedSearchCV with SVM
rand_svm_clf = RandomizedSearchCV(SVC(probability=True), svm_params, n_iter=4)

# SMOTE + SVM (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="SMOTE + SVM"):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                            rand_svm_clf)
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_svm_clf.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])

    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))

# SVM without SMOTE (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="SVM without SMOTE"):
    model_reg_imb = rand_svm_clf.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg_imb = rand_svm_clf.best_estimator_
    prediction_reg_imb = best_est_reg_imb.predict(X_train_sm[val])

    accuracy_imb.append(model_reg_imb.score(X_train_sm[val], y_train_sm[val]))
    precision_imb.append(precision_score(y_train_sm[val], prediction_reg_imb))
    recall_imb.append(recall_score(y_train_sm[val], prediction_reg_imb))
    f1_imb.append(f1_score(y_train_sm[val], prediction_reg_imb))
    auc_imb.append(roc_auc_score(y_train_sm[val], prediction_reg_imb))

# Print results for SMOTE
print('---' * 45)
print('')
print('SVM (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print("auc: {}".format(np.mean(auc_lst_reg)))
print('')
print('---' * 45)

# Print results for non-SMOTE
print('---' * 45)
print('')
print('SVM results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_imb)))
print("precision: {}".format(np.mean(precision_imb)))
print("recall: {}".format(np.mean(recall_imb)))
print("f1: {}".format(np.mean(f1_imb)))
print("auc: {}".format(np.mean(auc_imb)))
print('')
print('---' * 45)

#MLP

# Lists to append the score and then find the average
accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

accuracy_imb = []
precision_imb = []
recall_imb = []
f1_imb = []
auc_imb = []

# Replace previous models with MLP
mlp_clf = MLPClassifier(max_iter=1000)

# Define hyperparameters for the MLP
mlp_params = {
    'hidden_layer_sizes': [(50,50,50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['adaptive'],
}

# RandomizedSearchCV with MLP
rand_mlp_clf = RandomizedSearchCV(MLPClassifier(max_iter=1000), mlp_params, n_iter=4)

# SMOTE + MLP (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="SMOTE + MLP"):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                            rand_mlp_clf)
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_mlp_clf.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])

    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))

# MLP without SMOTE (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="MLP without SMOTE"):
    model_reg_imb = rand_mlp_clf.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg_imb = rand_mlp_clf.best_estimator_
    prediction_reg_imb = best_est_reg_imb.predict(X_train_sm[val])

    accuracy_imb.append(model_reg_imb.score(X_train_sm[val], y_train_sm[val]))
    precision_imb.append(precision_score(y_train_sm[val], prediction_reg_imb))
    recall_imb.append(recall_score(y_train_sm[val], prediction_reg_imb))
    f1_imb.append(f1_score(y_train_sm[val], prediction_reg_imb))
    auc_imb.append(roc_auc_score(y_train_sm[val], prediction_reg_imb))

# Print results for SMOTE
print('---' * 45)
print('')
print('MLP (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print("auc: {}".format(np.mean(auc_lst_reg)))
print('')
print('---' * 45)

# Print results for non-SMOTE
print('---' * 45)
print('')
print('MLP results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_imb)))
print("precision: {}".format(np.mean(precision_imb)))
print("recall: {}".format(np.mean(recall_imb)))
print("f1: {}".format(np.mean(f1_imb)))
print("auc: {}".format(np.mean(auc_imb)))
print('')
print('---' * 45)

#XGB

# Lists to append the score and then find the average
accuracy_lst_reg = []
precision_lst_reg = []
recall_lst_reg = []
f1_lst_reg = []
auc_lst_reg = []

accuracy_imb = []
precision_imb = []
recall_imb = []
f1_imb = []
auc_imb = []

# Replace SVM with XGBoost
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define hyperparameters for the XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2, 5]  # To handle class imbalance
}

# RandomizedSearchCV with XGBoost
rand_xgb_clf = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                                  xgb_params, n_iter=4)

# SMOTE + XGBoost (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="SMOTE + XGBoost"):
    pipeline_reg = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),
                                            rand_xgb_clf)
    model_reg = pipeline_reg.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg = rand_xgb_clf.best_estimator_
    prediction_reg = best_est_reg.predict(X_train_sm[val])

    accuracy_lst_reg.append(pipeline_reg.score(X_train_sm[val], y_train_sm[val]))
    precision_lst_reg.append(precision_score(y_train_sm[val], prediction_reg))
    recall_lst_reg.append(recall_score(y_train_sm[val], prediction_reg))
    f1_lst_reg.append(f1_score(y_train_sm[val], prediction_reg))
    auc_lst_reg.append(roc_auc_score(y_train_sm[val], prediction_reg))

# XGBoost without SMOTE (with cross-validation)
for train, val in tqdm(sss.split(X_train_sm, y_train_sm), desc="XGBoost without SMOTE"):
    model_reg_imb = rand_xgb_clf.fit(X_train_sm[train], y_train_sm[train])
    best_est_reg_imb = rand_xgb_clf.best_estimator_
    prediction_reg_imb = best_est_reg_imb.predict(X_train_sm[val])

    accuracy_imb.append(model_reg_imb.score(X_train_sm[val], y_train_sm[val]))
    precision_imb.append(precision_score(y_train_sm[val], prediction_reg_imb))
    recall_imb.append(recall_score(y_train_sm[val], prediction_reg_imb))
    f1_imb.append(f1_score(y_train_sm[val], prediction_reg_imb))
    auc_imb.append(roc_auc_score(y_train_sm[val], prediction_reg_imb))

# Print results for SMOTE
print('---' * 45)
print('')
print('XGBoost (SMOTE) results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_lst_reg)))
print("precision: {}".format(np.mean(precision_lst_reg)))
print("recall: {}".format(np.mean(recall_lst_reg)))
print("f1: {}".format(np.mean(f1_lst_reg)))
print("auc: {}".format(np.mean(auc_lst_reg)))
print('')
print('---' * 45)

# Print results for non-SMOTE
print('---' * 45)
print('')
print('XGBoost results:')
print('')
print("accuracy: {}".format(np.mean(accuracy_imb)))
print("precision: {}".format(np.mean(precision_imb)))
print("recall: {}".format(np.mean(recall_imb)))
print("f1: {}".format(np.mean(f1_imb)))
print("auc: {}".format(np.mean(auc_imb)))
print('')
print('---' * 45)






