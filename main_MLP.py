import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Open and read file
# Open and read file
data = pd.read_csv(r'C:\Users\jr721\OneDrive\文档\Xinyuan\BP Classification data\Datasets/Poland 2007.csv')

data = data.fillna(0) # Filling missing values

data['Bankrupt?'] = data['Bankrupt?'].astype(int) #Transfering the data type

data['Bankrupt?'].value_counts() # Count the occurrence of 'Bankrupt"

# Define positive samples and negative samples
positive_samples = data[data['Bankrupt?'] == 1]
negative_samples = data[data['Bankrupt?'] == 0]
print(len(positive_samples))
print(len(negative_samples)) #168:4043=4:100

# exit()
# Data augmentation to the positive samples (using SMOTE)
X = data.drop(columns=['Bankrupt?'])
y = data['Bankrupt?']
from imblearn.under_sampling import RandomUnderSampler

sampling_strategy = 0.5
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_new, y_new = rus.fit_resample(X, y)


# smote = SMOTE(sampling_strategy= 'auto', k_neighbors=6, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
smote = SMOTE(sampling_strategy= 0.7, k_neighbors=4, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)

X_resampled,y_resampled = smote.fit_resample(X_new,y_new)
# print('smote result:', len(X_resampled), len(y_resampled))
# print('smote bili', y_resampled)
# Merging dataframes
data = pd.concat([pd.DataFrame(X_new, columns=X_new.columns), pd.DataFrame(y_new, columns=['Bankrupt?'])], axis=1)

# # Under sampling to delete noise in the majority class
# tomek = TomekLinks()
# X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)

# Merging DataFrame after under-sampling
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Bankrupt?'])], axis=1)

# Print the new dataset value counts
print(resampled_data['Bankrupt?'].value_counts())

# Merging data after oversampling and undersampling
# resampled_data = pd.concat([pd.DataFrame(X_final, columns=X.columns), pd.DataFrame(y_final, columns=['Bankrupt?'])], axis=1)

# Shuffling data
balanced_data = resampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first few rows of the balanced_data DataFrame
balanced_data.head()

# Count the occurrences of each unique value in the 'Bankrupt?' column
balanced_data['Bankrupt?'].value_counts()

# Split the train and test set
X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop('Bankrupt?', axis=1), balanced_data['Bankrupt?'], test_size=0.3)

X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(data.drop('Bankrupt?', axis=1), data['Bankrupt?'], test_size=0.4)

# Initialize the StandardScaler, which standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)
# Apply the same transformation to the test data using the previously fitted scaler
X_test = scaler.transform(X_test)
X_train_ori = scaler.fit_transform(X_train_ori)
X_test_ori = scaler.transform(X_test_ori)

#MLP modeling
mlp_model = MLPClassifier(hidden_layer_sizes=(30,),max_iter=200, alpha=0.01, learning_rate_init=0.001, random_state=42)

# 训练模型
mlp_model.fit(X_train, y_train)

# 预测训练集
y_train_pred = mlp_model.predict(X_train)

# 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
train_auc = roc_auc_score(y_train, mlp_model.predict_proba(X_train)[:, 1])
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_confusion_matrix = confusion_matrix(y_train, y_train_pred)

print("MLP Train AUC:", train_auc)
print("MLP Train Accuracy:", train_accuracy)
print("MLP Train Precision:", train_precision)
print("MLP Train Recall:", train_recall)
print("MLP Train Confusion Matrix:\n", train_confusion_matrix)

print("*" * 50)

# 预测测试集
y_test_pred = mlp_model.predict(X_test)

# 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
test_auc = roc_auc_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

print("MLP Test AUC:", test_auc)
print("MLP Test Accuracy:", test_accuracy)
print("MLP Test Precision:", test_precision)
print("MLP Test Recall:", test_recall)
print("MLP Test Confusion Matrix:\n", test_confusion_matrix)

#MLP modeling on original data
mlp_model_ori = MLPClassifier(hidden_layer_sizes=(50,),max_iter=10, alpha=0.01, learning_rate_init=0.001, random_state=42, early_stopping=True)

# 训练模型
mlp_model_ori.fit(X_train_ori, y_train_ori)

# 预测训练集
y_train_pred_ori = mlp_model.predict(X_train_ori)

# 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
train_auc_ori = roc_auc_score(y_train_ori, y_train_pred_ori)
train_accuracy_ori = accuracy_score(y_train_ori, y_train_pred_ori)
train_precision_ori = precision_score(y_train_ori, y_train_pred_ori)
train_recall_ori = recall_score(y_train_ori, y_train_pred_ori)
train_confusion_matrix_ori = confusion_matrix(y_train_ori, y_train_pred_ori)

print("MLP Train AUC on original:", train_auc_ori)
print("MLP Train Accuracy on original:", train_accuracy_ori)
print("MLP Train Precision on original:", train_precision_ori)
print("MLP Train Recall on original:", train_recall_ori)
print("MLP Train Confusion Matrix on original:\n", train_confusion_matrix_ori)

print("*" * 50)

# 预测测试集
y_test_pred_ori = mlp_model_ori.predict(X_test_ori)

# 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
test_auc_ori = roc_auc_score(y_test_ori, y_test_pred_ori)
test_accuracy_ori = accuracy_score(y_test_ori, y_test_pred_ori)
test_precision_ori = precision_score(y_test_ori, y_test_pred_ori)
test_recall_ori = recall_score(y_test_ori, y_test_pred_ori)
test_confusion_matrix_ori = confusion_matrix(y_test_ori, y_test_pred_ori)

print("MLP Test AUC on original:", test_auc_ori)
print("MLP Test Accuracy on original:", test_accuracy_ori)
print("MLP Test Precision on original:", test_precision_ori)
print("MLP Test Recall on original:", test_recall_ori)
print("MLP Test Confusion Matrix on original:\n", test_confusion_matrix_ori)