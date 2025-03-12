import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Open and read file
data = pd.read_csv(r'C:\Users\jr721\OneDrive\文档\Xinyuan\BP Classification data\Datasets/Brazil 2020.csv')

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

sampling_strategy = 0.3
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_new, y_new = rus.fit_resample(X, y)
print(len(X_new))


# smote = SMOTE(sampling_strategy= 'auto', k_neighbors=6, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)
smote = SMOTE(sampling_strategy= 0.8, k_neighbors=4, random_state = 42) # (let k_neighbors = 2 since bankruptcy datasets are usually extremely imbalanced)

X_resampled,y_resampled = smote.fit_resample(X_new,y_new)
data = pd.concat([pd.DataFrame(X_new, columns=X_new.columns), pd.DataFrame(y_new, columns=['Bankrupt?'])], axis=1)

# Merging DataFrame after under-sampling
resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Bankrupt?'])], axis=1)

# Print the new dataset value counts
print(resampled_data['Bankrupt?'].value_counts())

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

# DT Modeling
clf = DecisionTreeClassifier(max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             class_weight='balanced',
                             random_state=42)
# 训练模型
clf.fit(X_train, y_train)

# 预测训练集
y_train_pred = clf.predict(X_train)

print("*"*50)

# 预测测试集
y_test_pred = clf.predict(X_test)

# 计算测试集上的AUC、准确率、精确率、召回率和混淆矩阵
test_auc = roc_auc_score(y_test, y_test_pred) # clf.predict_proba(X_test_ori)[:, 1])
test_auc1 = average_precision_score(y_test, y_test_pred, average='weighted') # clf.predict_proba(X_test)[:, 1])
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

print("Decision Tree Test AUC:", test_auc1)
print("Decision Tree Test Accuracy:", test_accuracy)
print("Decision Tree Test Precision:", test_precision)
print("Decision Tree Test Recall:", test_recall)
print("Decision Tree Test Confusion Matrix:\n", test_confusion_matrix)

# 对no smote原始数据进行预测
clf_nosmote = DecisionTreeClassifier(max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             # class_weight='balanced',
                             random_state=42)
# 训练模型
clf_nosmote.fit(X_train_ori, y_train_ori)
threshold = 0.2

y_test_pred = clf_nosmote.predict(X_test) #clf.predict_proba(scaler.transform(X))[:, 1] > threshold).astype(int)
test_auc = roc_auc_score(y_test, y_test_pred)
test_auc_ori = average_precision_score(y_test, y_test_pred, average='weighted') #weighted
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

print("Decision Tree No smote AUC:", test_auc_ori)
print("Decision Tree No smote  Accuracy:", test_accuracy)
print("Decision Tree No smote  Precision:", test_precision)
print("Decision Tree No smote  Recall:", test_recall)
print("Decision Tree No smote  Confusion Matrix:\n", test_confusion_matrix)

print('*'*50)