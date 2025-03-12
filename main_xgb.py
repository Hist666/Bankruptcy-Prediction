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

sampling_strategy = 0.2
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_new, y_new = rus.fit_resample(X, y)
print(len(X_new))

# smote = SMOTE(sampling_strategy= 'auto', k_neighbors=6, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
smote = SMOTE(sampling_strategy= 0.8, k_neighbors=4, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
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
X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop('Bankrupt?', axis=1), balanced_data['Bankrupt?'], test_size=0.4)

X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(data.drop('Bankrupt?', axis=1), data['Bankrupt?'], test_size=0.4)

# Initialize the StandardScaler, which standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)
# Apply the same transformation to the test data using the previously fitted scaler
X_test = scaler.transform(X_test)
X_train_ori = scaler.fit_transform(X_train_ori)
X_test_ori = scaler.transform(X_test_ori)

# XGBoost
# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Calculate the positive samples weight
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print("scale_pos_weight:", scale_pos_weight)

# Parameters setup
param = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'aucpr',  # Using AUCPR to test the model performance
    # AUCPR is more sensitive to the imbalanced datasets, which makes it suitable for imbalanced data sets
    #'scale_pos_weight': scale_pos_weight*10  # adjust the samples weight
}

# Model training
num_round = 50  # Iteration times

# Create a monitoring list to monitor the model performance
evallist = [(dtest, 'eval'), (dtrain, 'train')]

# Using XGBoost for model training
bst = xgb.train(param, dtrain, num_round, evallist)

# train auc
train_auc = roc_auc_score(y_train, bst.predict(xgb.DMatrix(X_train)))
print("xgb Train AUC:", train_auc)

# Prediction
dtrain_ori = xgb.DMatrix(X_train, label=y_train)
dtest_ori = xgb.DMatrix(X_test, label=y_test)
y_pred = bst.predict(dtest)
# test auc
test_auc = roc_auc_score(y_test, y_pred)
test_aucpr = average_precision_score(y_test,y_pred)
print("Test AUC:", test_auc)

# test accuracy
test_accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
print("xgb Test Accuracy:", test_accuracy)

# test precision
precision = precision_score(y_test, (y_pred > 0.5).astype(int))
print("xgb test Precision:", precision)

# test recall
recall = recall_score(y_test, (y_pred > 0.5).astype(int))
print("xgb test Recall:", recall)

# test accuracy and precision, confusion metrix
test_confusion_matrix = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
print("xgb Test Confusion Matrix:\n", test_confusion_matrix)

print('*'*50)

# Prediction on the original data
# XGBoost
# Create DMatrix
dtrain_ori = xgb.DMatrix(X_train_ori, label=y_train_ori)
dtest_ori = xgb.DMatrix(X_test_ori, label=y_test_ori)

# Calculate the positive samples weight
scale_pos_weight = (len(y_train_ori) - sum(y_train_ori)) / sum(y_train_ori)
print("scale_pos_weight:", scale_pos_weight)

# Parameters setup
param = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'aucpr',  # Using AUCPR to test the model performance
    # AUCPR is more sensitive to the imbalanced datasets, which makes it suitable for imbalanced data sets
    'scale_pos_weight': scale_pos_weight*10  # adjust the samples weight
}

# Model training
num_round = 50  # Iteration times

# Create a monitoring list to monitor the model performance
evallist = [(dtest_ori, 'eval'), (dtrain_ori, 'train')]

# Using XGBoost for model training
bst = xgb.train(param, dtrain_ori, num_round, evallist)

# train auc
y_test_pred_ori = (bst.predict(xgb.DMatrix(X_test)) > 0.5).astype(int)
# Prediction
# y_test_pred_ori = bst.predict(dtest)
train_auc_ori = roc_auc_score(y_test, y_test_pred_ori)
print("xgb no smote Test AUC on original data:", train_auc_ori)

# train accuracy and precision,
train_accuracy_ori = accuracy_score(y_test, y_test_pred_ori)
train_precision_ori = precision_score(y_test, y_test_pred_ori)
train_recall_ori = recall_score(y_test, y_test_pred_ori)
print("xgb no smote Test Accuracy on original data:", train_accuracy_ori)
print("xgb no smote Test Precision on original data:", train_precision_ori)
print("xgb no smote Test Recall on original data:", train_recall_ori)

# train confusion metrix
train_confusion_matrix_ori = confusion_matrix(y_test, y_test_pred_ori)
print("xgb no smote Test Confusion Matrix on original data:\n", train_confusion_matrix_ori)
print("*"*50)

# # 创建决策树模型
# clf = DecisionTreeClassifier(max_depth=5,
#                              min_samples_split=10,
#                              min_samples_leaf=5,
#                              class_weight='balanced',
#                              random_state=42)
# params = clf.get_params()
# print(params)
# # 训练模型
# clf.fit(X_train, y_train)
#
# # 预测训练集
# y_train_pred = clf.predict(X_train)
#
# # 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
# # train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
# # train_auc1 = average_precision_score(y_train, clf.predict_proba(X_train)[:, 1])
# # train_accuracy = accuracy_score(y_train, y_train_pred)
# # train_precision = precision_score(y_train, y_train_pred)
# # train_recall = recall_score(y_train, y_train_pred)
# # train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# # print("Decision Tree Train AUC:", train_auc)
# # print("Decision Tree Train Accuracy:", train_accuracy)
# # print("Decision Tree Train Precision:", train_precision)
# # print("Decision Tree Train Recall:", train_recall)
# # print("Decision Tree Train Confusion Matrix:\n", train_confusion_matrix)
#
# print("*"*50)
#
# # 预测测试集
# y_test_pred = clf.predict(X_test)
#
# # 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
# test_auc = roc_auc_score(y_test, y_test_pred) # clf.predict_proba(X_test_ori)[:, 1])
# test_auc1 = average_precision_score(y_test, y_test_pred, average='weighted') # clf.predict_proba(X_test)[:, 1])
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# print("Decision Tree Test AUC:", test_auc1)
# print("Decision Tree Test Accuracy:", test_accuracy)
# print("Decision Tree Test Precision:", test_precision)
# print("Decision Tree Test Recall:", test_recall)
# print("Decision Tree Test Confusion Matrix:\n", test_confusion_matrix)
#
# # 对no smote原始数据进行预测
# clf_nosmote = DecisionTreeClassifier(max_depth=5,
#                              min_samples_split=10,
#                              min_samples_leaf=5,
#                              # class_weight='balanced',
#                              random_state=42)
# # 训练模型
# clf_nosmote.fit(X_train_ori, y_train_ori)
# threshold = 0.2
#
# y_test_pred = clf_nosmote.predict(X_test) #clf.predict_proba(scaler.transform(X))[:, 1] > threshold).astype(int)
# test_auc = roc_auc_score(y_test, y_test_pred)
# test_auc_ori = average_precision_score(y_test, y_test_pred, average='weighted') #weighted
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# print("Decision Tree No smote AUC:", test_auc_ori)
# print("Decision Tree No smote  Accuracy:", test_accuracy)
# print("Decision Tree No smote  Precision:", test_precision)
# print("Decision Tree No smote  Recall:", test_recall)
# print("Decision Tree No smote  Confusion Matrix:\n", test_confusion_matrix)
#
# print('*'*50)
#SVM
# Initialize the SVM model
# svm_model = SVC(kernel='linear', probability=True, random_state=42)
#
# # Train the SVM model
# svm_model.fit(X_train, y_train)
#
# # Evaluate on the training set
# y_train_pred = svm_model.predict(X_train)
# train_auc = roc_auc_score(y_train, svm_model.predict_proba(X_train)[:, 1])
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# print("SVM Train AUC:", train_auc)
# print("SVM Train Accuracy:", train_accuracy)
# print("SVM Train Precision:", train_precision)
# print("SVM Train Recall:", train_recall)
# print("SVM Train Confusion Matrix:\n", train_confusion_matrix)
#
# print("*" * 50)
#
# # Predict on the test set
# y_pred = svm_model.predict(X_test)
# y_pred_proba = svm_model.predict_proba(X_test)[:, 1]
#
# # Evaluate on the test set
# test_auc = roc_auc_score(y_test, y_pred_proba)
# test_accuracy = accuracy_score(y_test, y_pred)
# test_precision = precision_score(y_test, y_pred)
# test_recall = recall_score(y_test, y_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_pred)
#
# print("SVM Test AUC:", test_auc)
# print("SVM Test Accuracy:", test_accuracy)
# print("SVM Test Precision:", test_precision)
# print("SVM Test Recall:", test_recall)
# print("SVM Test Confusion Matrix:\n", test_confusion_matrix)
#
# # Evaluate on the original data (before resampling)
# y_pred_ori = svm_model.predict(scaler.transform(X))
#
# test_auc_ori = roc_auc_score(y, y_pred_ori)
# test_accuracy_ori = accuracy_score(y, y_pred_ori)
# test_precision_ori = precision_score(y, y_pred_ori)
# test_recall_ori = recall_score(y, y_pred_ori)
# test_confusion_matrix_ori = confusion_matrix(y, y_pred_ori)
#
# print("No SMOTE SVM Test AUC:", test_auc_ori)
# print("No SMOTE SVM Test Accuracy:", test_accuracy_ori)
# print("No SMOTE SVM Test Precision:", test_precision_ori)
# print("No SMOTE SVM Test Recall:", test_recall_ori)
# print("No SMOTE SVM Test Confusion Matrix:\n", test_confusion_matrix_ori)
#
# #Logistic model
# # Logistic Regression Model
# logistic_model = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', random_state=42)
#
# # 训练模型
# logistic_model.fit(X_train, y_train)
#
# # 预测训练集
# y_train_pred = logistic_model.predict(X_train)
#
# # 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
# train_auc = roc_auc_score(y_train, logistic_model.predict_proba(X_train)[:, 1])
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# print("Logistic Regression Train AUC:", train_auc)
# print("Logistic Regression Train Accuracy:", train_accuracy)
# print("Logistic Regression Train Precision:", train_precision)
# print("Logistic Regression Train Recall:", train_recall)
# print("Logistic Regression Train Confusion Matrix:\n", train_confusion_matrix)
#
# print("*" * 50)
#
# # 预测测试集
# y_test_pred = logistic_model.predict(X_test)
#
# # 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
# test_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1])
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# print("Logistic Regression Test AUC:", test_auc)
# print("Logistic Regression Test Accuracy:", test_accuracy)
# print("Logistic Regression Test Precision:", test_precision)
# print("Logistic Regression Test Recall:", test_recall)
# print("Logistic Regression Test Confusion Matrix:\n", test_confusion_matrix)
#
# #原始数据集上的表现
# # 预测原始数据集
# y_pred_ori_logistic = logistic_model.predict(scaler.transform(X))
#
# # 计算原始数据集上的AUC、准确率、精确率、召回率和混淆矩阵
# ori_test_auc_logistic = roc_auc_score(y, logistic_model.predict_proba(scaler.transform(X))[:, 1])
# ori_test_accuracy_logistic = accuracy_score(y, y_pred_ori_logistic)
# ori_test_precision_logistic = precision_score(y, y_pred_ori_logistic)
# ori_test_recall_logistic = recall_score(y, y_pred_ori_logistic)
# ori_test_confusion_matrix_logistic = confusion_matrix(y, y_pred_ori_logistic)
#
# print("Logistic Regression No SMOTE Test AUC:", ori_test_auc_logistic)
# print("Logistic Regression No SMOTE Test Accuracy:", ori_test_accuracy_logistic)
# print("Logistic Regression No SMOTE Test Precision:", ori_test_precision_logistic)
# print("Logistic Regression No SMOTE Test Recall:", ori_test_recall_logistic)
# print("Logistic Regression No SMOTE Test Confusion Matrix:\n", ori_test_confusion_matrix_logistic)
#
# # Multi-layer Perceptron (MLP) Model
# mlp_model = MLPClassifier(hidden_layer_sizes=(50,),max_iter=10, alpha=0.01, learning_rate_init=0.001, random_state=42, early_stopping=True)
#
# # 训练模型
# mlp_model.fit(X_train, y_train)
#
# # 预测训练集
# y_train_pred = mlp_model.predict(X_train)
#
# # 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
# train_auc = roc_auc_score(y_train, mlp_model.predict_proba(X_train)[:, 1])
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# print("MLP Train AUC:", train_auc)
# print("MLP Train Accuracy:", train_accuracy)
# print("MLP Train Precision:", train_precision)
# print("MLP Train Recall:", train_recall)
# print("MLP Train Confusion Matrix:\n", train_confusion_matrix)
#
# print("*" * 50)
#
# # 预测测试集
# y_test_pred = mlp_model.predict(X_test)
#
# # 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
# test_auc = roc_auc_score(y_test, mlp_model.predict_proba(X_test)[:, 1])
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# print("MLP Test AUC:", test_auc)
# print("MLP Test Accuracy:", test_accuracy)
# print("MLP Test Precision:", test_precision)
# print("MLP Test Recall:", test_recall)
# print("MLP Test Confusion Matrix:\n", test_confusion_matrix)
#
# # 预测原始数据集
# y_pred_ori_mlp = mlp_model.predict(scaler.transform(X))
#
# # 计算原始数据集上的AUC、准确率、精确率、召回率和混淆矩阵
# ori_test_auc_mlp = roc_auc_score(y, mlp_model.predict_proba(scaler.transform(X))[:, 1])
# ori_test_accuracy_mlp = accuracy_score(y, y_pred_ori_mlp)
# ori_test_precision_mlp = precision_score(y, y_pred_ori_mlp)
# ori_test_recall_mlp = recall_score(y, y_pred_ori_mlp)
# ori_test_confusion_matrix_mlp = confusion_matrix(y, y_pred_ori_mlp)
#
# print("MLP No SMOTE Test AUC:", ori_test_auc_mlp)
# print("MLP No SMOTE Test Accuracy:", ori_test_accuracy_mlp)
# print("MLP No SMOTE Test Precision:", ori_test_precision_mlp)
# print("MLP No SMOTE Test Recall:", ori_test_recall_mlp)
# print("MLP No SMOTE Test Confusion Matrix:\n", ori_test_confusion_matrix_mlp)
#
# # Check and handle NaN values in the dataset by replacing them with 0
# X_train = np.nan_to_num(X_train)
# X_test = np.nan_to_num(X_test)
# X = np.nan_to_num(X)
# y_train = np.nan_to_num(y_train)
# y_test = np.nan_to_num(y_test)
# y = np.nan_to_num(y)
#
# # Standardize the features (mean = 0, std = 1) for better model performance
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_all = scaler.transform(X)
#
# # Add a constant column (intercept) to the feature matrices for the Probit model
# X_train_const = sm.add_constant(X_train)
# X_test_const = sm.add_constant(X_test)
# X_all_const = sm.add_constant(X_all)
#
# # Initialize and fit the Probit model with extended iterations to ensure convergence
# probit_model = sm.Probit(y_train, X_train_const)
# probit_result = probit_model.fit(maxiter=1000, method='bfgs')
#
# ### 1. Training Set Evaluation ###
# # Predict probabilities on the training set
# y_train_pred_prob = probit_result.predict(X_train_const)
# # Convert probabilities to binary predictions with a threshold of 0.5
# y_train_pred = (y_train_pred_prob > 0.5).astype(int)
#
# # Calculate AUC, accuracy, precision, recall, and confusion matrix on the training set
# train_auc = roc_auc_score(y_train, y_train_pred_prob)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# # Print training evaluation metrics
# print("Probit Train AUC:", train_auc)
# print("Probit Train Accuracy:", train_accuracy)
# print("Probit Train Precision:", train_precision)
# print("Probit Train Recall:", train_recall)
# print("Probit Train Confusion Matrix:\n", train_confusion_matrix)
# print("*" * 50)
#
# ### 2. Test Set Evaluation ###
# # Predict probabilities on the test set
# y_test_pred_prob = probit_result.predict(X_test_const)
# # Convert probabilities to binary predictions with a threshold of 0.5
# y_test_pred = (y_test_pred_prob > 0.5).astype(int)
#
# # Calculate AUC, accuracy, precision, recall, and confusion matrix on the test set
# test_auc = roc_auc_score(y_test, y_test_pred_prob)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# # Print test evaluation metrics
# print("Probit Test AUC:", test_auc)
# print("Probit Test Accuracy:", test_accuracy)
# print("Probit Test Precision:", test_precision)
# print("Probit Test Recall:", test_recall)
# print("Probit Test Confusion Matrix:\n", test_confusion_matrix)
# print("*" * 50)
#
# ### 3. Original Dataset Evaluation ###
# # Predict probabilities on the original dataset (without SMOTE or modifications)
# # Use X_all_const, which already has the constant column
# X_all_const = np.column_stack((np.ones(X_all.shape[0]), X_all))
# y_pred_ori_mlp = probit_result.predict(X_all_const)
# # Convert probabilities to binary predictions with a threshold of 0.5
# y_pred_ori_mlp_class = (y_pred_ori_mlp > 0.5).astype(int)
#
# # Calculate AUC, accuracy, precision, recall, and confusion matrix on the original dataset
# ori_test_auc_pb = roc_auc_score(y, y_pred_ori_mlp)
# ori_test_accuracy_pb = accuracy_score(y, y_pred_ori_mlp_class)
# ori_test_precision_pb = precision_score(y, y_pred_ori_mlp_class)
# ori_test_recall_pb = recall_score(y, y_pred_ori_mlp_class)
# ori_test_confusion_matrix_pb = confusion_matrix(y, y_pred_ori_mlp_class)
#
# # Print original dataset evaluation metrics
# print("Probit Original Data AUC:", ori_test_auc_pb)
# print("Probit Original Data Accuracy:", ori_test_accuracy_pb)
# print("Probit Original Data Precision:", ori_test_precision_pb)
# print("Probit Original Data Recall:", ori_test_recall_pb)
# print("Probit Original Data Confusion Matrix:\n", ori_test_confusion_matrix_pb)
#
#
# #MDA multilinear analysis
# # 检查并处理 NaN 值
# X_train = np.nan_to_num(X_train)
# X_test = np.nan_to_num(X_test)
# X = np.nan_to_num(X)
# y_train = np.nan_to_num(y_train)
# y_test = np.nan_to_num(y_test)
# y = np.nan_to_num(y)
#
# # 标准化数据
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_all = scaler.transform(X)
#
# # 初始化并拟合 LDA 模型
# lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
# lda_model.fit(X_train, y_train)
#
# ### 1. 训练集评估 ###
# # 预测训练集概率
# y_train_pred_prob = lda_model.predict_proba(X_train)[:, 1]  # 取第二列为类别 1 的概率
# y_train_pred = lda_model.predict(X_train)  # 获取分类标签
#
# # 计算训练集上的AUC、准确率、精确率、召回率和混淆矩阵
# train_auc = roc_auc_score(y_train, y_train_pred_prob)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_precision = precision_score(y_train, y_train_pred)
# train_recall = recall_score(y_train, y_train_pred)
# train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
#
# print("LDA Train AUC:", train_auc)
# print("LDA Train Accuracy:", train_accuracy)
# print("LDA Train Precision:", train_precision)
# print("LDA Train Recall:", train_recall)
# print("LDA Train Confusion Matrix:\n", train_confusion_matrix)
#
# print("*" * 50)
#
# ### 2. 测试集评估 ###
# # 预测测试集概率
# y_test_pred_prob = lda_model.predict_proba(X_test)[:, 1]
# y_test_pred = lda_model.predict(X_test)
#
# # 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
# test_auc = roc_auc_score(y_test, y_test_pred_prob)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_precision = precision_score(y_test, y_test_pred)
# test_recall = recall_score(y_test, y_test_pred)
# test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
#
# print("LDA Test AUC:", test_auc)
# print("LDA Test Accuracy:", test_accuracy)
# print("LDA Test Precision:", test_precision)
# print("LDA Test Recall:", test_recall)
# print("LDA Test Confusion Matrix:\n", test_confusion_matrix)
#
# print("*" * 50)
#
# ### 3. 原始数据集评估 ###
# # 预测原始数据集
# y_pred_ori_lda = lda_model.predict_proba(scaler.fit_transform(X))[:, 1]
# y_pred_ori_lda_class = lda_model.predict(scaler.fit_transform(X))
#
# # 计算原始数据集上的AUC、准确率、精确率、召回率和混淆矩阵
# ori_test_auc_lda = roc_auc_score(y, y_pred_ori_lda)
# ori_test_accuracy_lda = accuracy_score(y, y_pred_ori_lda_class)
# ori_test_precision_lda = precision_score(y, y_pred_ori_lda_class)
# ori_test_recall_lda = recall_score(y, y_pred_ori_lda_class)
# ori_test_confusion_matrix_lda = confusion_matrix(y, y_pred_ori_lda_class)
#
# print("LDA No SMOTE Test AUC:", ori_test_auc_lda)
# print("LDA No SMOTE Test Accuracy:", ori_test_accuracy_lda)
# print("LDA No SMOTE Test Precision:", ori_test_precision_lda)
# print("LDA No SMOTE Test Recall:", ori_test_recall_lda)
# print("LDA No SMOTE Test Confusion Matrix:\n", ori_test_confusion_matrix_lda)
#
#
