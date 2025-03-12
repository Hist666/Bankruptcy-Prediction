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

sampling_strategy = 0.4
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_new, y_new = rus.fit_resample(X, y)


# smote = SMOTE(sampling_strategy= 'auto', k_neighbors=6, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
smote = SMOTE(sampling_strategy= 0.6, k_neighbors=4, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)

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

# svm
# Initialize svm model
svm_model = SVC(kernel= 'rbf', probability= True, random_state=42)

#Train the svm model
svm_model.fit(X_train,y_train)

# Evaluate on the training set
y_train_pred = svm_model.predict(X_train)
train_auc = roc_auc_score(y_train, svm_model.predict_proba(X_train)[:, 1])
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_confusion_matrix = confusion_matrix(y_train, y_train_pred)

print("SVM Train AUC:", train_auc)
print("SVM Train Accuracy:", train_accuracy)
print("SVM Train Precision:", train_precision)
print("SVM Train Recall:", train_recall)
print("SVM Train Confusion Matrix:\n", train_confusion_matrix)

print("*" * 50)

# Predict on the test set
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# Evaluate on the test set
test_auc = roc_auc_score(y_test, y_pred_proba)
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_confusion_matrix = confusion_matrix(y_test, y_pred)

print("SVM Test AUC:", test_auc)
print("SVM Test Accuracy:", test_accuracy)
print("SVM Test Precision:", test_precision)
print("SVM Test Recall:", test_recall)
print("SVM Test Confusion Matrix:\n", test_confusion_matrix)

# original data
svm_model_ori = SVC(kernel= 'rbf', probability= True, random_state=42)

#train on the original data
svm_model_ori.fit(X_train_ori,y_train_ori)

# Evaluate on the training set
y_train_pred_ori = svm_model.predict(X_train_ori)
train_auc_ori = roc_auc_score(y_train_ori, svm_model.predict_proba(X_train_ori)[:, 1])
train_accuracy_ori = accuracy_score(y_train_ori, y_train_pred_ori)
train_precision_ori = precision_score(y_train_ori, y_train_pred_ori)
train_recall_ori = recall_score(y_train_ori, y_train_pred_ori)
train_confusion_matrix_ori = confusion_matrix(y_train_ori, y_train_pred_ori)

print("SVM Train AUC original:", train_auc_ori)
print("SVM Train Accuracy original:", train_accuracy_ori)
print("SVM Train Precision original:", train_precision_ori)
print("SVM Train Recall original:", train_recall_ori)
print("SVM Train Confusion Matrix original:\n", train_confusion_matrix_ori)

print("*" * 50)

# Predict on the test set
y_pred_ori = svm_model.predict(X_test_ori)
y_pred_proba_ori = svm_model.predict_proba(X_test_ori)[:, 1]

# Evaluate on the test set
test_auc_ori = roc_auc_score(y_test_ori, y_pred_proba_ori)
test_accuracy_ori = accuracy_score(y_test_ori, y_pred_ori)
test_precision_ori = precision_score(y_test_ori, y_pred_ori)
test_recall_ori = recall_score(y_test_ori, y_pred_ori)
test_confusion_matrix_ori = confusion_matrix(y_test_ori, y_pred_ori)

print("SVM Test AUC original:", test_auc_ori)
print("SVM Test Accuracy original:", test_accuracy_ori)
print("SVM Test Precision original:", test_precision_ori)
print("SVM Test Recall original:", test_recall_ori)
print("SVM Test Confusion Matrix original:\n", test_confusion_matrix_ori)