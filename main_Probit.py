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

sampling_strategy = 0.06
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_new, y_new = rus.fit_resample(X, y)


# smote = SMOTE(sampling_strategy= 'auto', k_neighbors=6, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
smote = SMOTE(sampling_strategy= 0.1, k_neighbors=4, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)

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
X_all = scaler.transform(X_new)

# Add a constant column (intercept) to the feature matrices for the Probit model
X_train_const = sm.add_constant(X_train)
X_train_const_ori = sm.add_constant(X_train_ori)
X_test_const = sm.add_constant(X_test)
X_test_const_ori = sm.add_constant(X_test_ori)
X_all_const = sm.add_constant(X_all)

# Initialize and fit the Probit model with extended iterations to ensure convergence
probit_model = sm.Probit(y_train, X_train_const)
probit_result = probit_model.fit(maxiter=1000, method='bfgs')

### 1. Training Set Evaluation ###
# Predict probabilities on the training set
y_train_pred_prob = probit_result.predict(X_train_const)
# Convert probabilities to binary predictions with a threshold of 0.5
y_train_pred = (y_train_pred_prob > 0.5).astype(int)

# Calculate AUC, accuracy, precision, recall, and confusion matrix on the training set
train_auc = roc_auc_score(y_train, y_train_pred_prob)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_confusion_matrix = confusion_matrix(y_train, y_train_pred)

# Print training evaluation metrics
print("Probit Train AUC:", train_auc)
print("Probit Train Accuracy:", train_accuracy)
print("Probit Train Precision:", train_precision)
print("Probit Train Recall:", train_recall)
print("Probit Train Confusion Matrix:\n", train_confusion_matrix)
print("*" * 50)

### 2. Test Set Evaluation ###
# Predict probabilities on the test set
y_test_pred_prob = probit_result.predict(X_test_const)
# Convert probabilities to binary predictions with a threshold of 0.5
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# Calculate AUC, accuracy, precision, recall, and confusion matrix on the test set
test_auc = roc_auc_score(y_test, y_test_pred_prob)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

# Print test evaluation metrics
print("Probit Test AUC:", test_auc)
print("Probit Test Accuracy:", test_accuracy)
print("Probit Test Precision:", test_precision)
print("Probit Test Recall:", test_recall)
print("Probit Test Confusion Matrix:\n", test_confusion_matrix)
print("*" * 50)

#Probit on the original data
# Initialize and fit the Probit model with extended iterations to ensure convergence
probit_model_ori = sm.Probit(y_train_ori, X_train_const_ori)
probit_result_ori = probit_model_ori.fit(maxiter=1000, method='bfgs')

### 1. Training Set Evaluation ###
# Predict probabilities on the training set
y_train_pred_prob_ori = probit_result.predict(X_train_const_ori)
# Convert probabilities to binary predictions with a threshold of 0.5
y_train_pred_ori = (y_train_pred_prob_ori > 0.5).astype(int)

# Calculate AUC, accuracy, precision, recall, and confusion matrix on the training set
train_auc_ori = roc_auc_score(y_train_ori, y_train_pred_prob_ori)
train_accuracy_ori = accuracy_score(y_train_ori, y_train_pred_ori)
train_precision_ori = precision_score(y_train_ori, y_train_pred_ori)
train_recall_ori = recall_score(y_train_ori, y_train_pred_ori)
train_confusion_matrix_ori = confusion_matrix(y_train_ori, y_train_pred_ori)

# Print training evaluation metrics
print("Probit Train AUC:", train_auc_ori)
print("Probit Train Accuracy:", train_accuracy_ori)
print("Probit Train Precision:", train_precision_ori)
print("Probit Train Recall:", train_recall_ori)
print("Probit Train Confusion Matrix:\n", train_confusion_matrix_ori)
print("*" * 50)

### 2. Test Set Evaluation ###
# Predict probabilities on the test set
y_test_pred_prob_ori = probit_result.predict(X_test_const_ori)
# Convert probabilities to binary predictions with a threshold of 0.5
y_test_pred_ori = (y_test_pred_prob_ori > 0.5).astype(int)

# Calculate AUC, accuracy, precision, recall, and confusion matrix on the test set
test_auc_ori = roc_auc_score(y_test_ori, y_test_pred_prob_ori)
test_accuracy_ori = accuracy_score(y_test_ori, y_test_pred_ori)
test_precision_ori = precision_score(y_test_ori, y_test_pred_ori)
test_recall_ori = recall_score(y_test_ori, y_test_pred_ori)
test_confusion_matrix_ori = confusion_matrix(y_test_ori, y_test_pred_ori)

# Print test evaluation metrics
print("Probit Test AUC:", test_auc_ori)
print("Probit Test Accuracy:", test_accuracy_ori)
print("Probit Test Precision:", test_precision_ori)
print("Probit Test Recall:", test_recall_ori)
print("Probit Test Confusion Matrix:\n", test_confusion_matrix_ori)
print("*" * 50)