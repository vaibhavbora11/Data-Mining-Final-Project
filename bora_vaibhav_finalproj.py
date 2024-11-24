#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all the necessary libraries for the code to run smoothly
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import time

# Using the time library to calculate and output the total time the code took loading

# Starting time
start_time = time.time()

# Loading the bank dataset
Bank_Dataset = pd.read_csv("bank.csv", delimiter=';')

# Mentioning the different columns from the dataset
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    le = LabelEncoder()
    Bank_Dataset[col] = le.fit_transform(Bank_Dataset[col].astype(str))

# Target variable 'y' is the output and it is in a binary classification of either 'yes' or 'no'
# Converting the target variable from categorical strings to binary format (1 for 'yes' and 0 for 'no')
# This helps in the usage of mathematical calculations in the code further down
Bank_Dataset['y'] = Bank_Dataset['y'].apply(lambda x: 1 if x == 'yes' else 0)

# The below code is fundamental step for data prep and is splitting the dataset into features and target variable
X = Bank_Dataset.drop('y', axis=1)
y = Bank_Dataset['y']

# Standardizing sets
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # This reshaped one is for the LSTM algorithm

# Setting up for the k-fold cross-validation making sure that there is balanced sampling
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Creating an empty list for all three algorithms to store the output of each fold
rf_metrics = []
svm_metrics = []
lstm_metrics = []

# Defining the LSTM model using input layer alongside input shape
def create_lstm_model(shape_input):
    model = Sequential()
    model.add(Input(shape=shape_input)) # Defining the shape
    model.add(LSTM(50, activation='relu', return_sequences=True)) # Setting return sequence to True
    model.add(Dropout(0.2)) # Drouput helps out with regularization
    model.add(LSTM(20, activation='relu')) # Another layer
    model.add(Dense(1, activation='sigmoid')) # Helping with the binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compilation
    return model

# Cross-validation loop for each of the three algorithms
for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index] # Splitting into training and testing sets
    X_train_reshaped, X_test_reshaped = X_reshaped[train_index], X_reshaped[test_index] # LSTM exclusive
    y_train, y_test = y[train_index], y[test_index]

    # Random Forest Classifier training and prediction
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_rf_proba = rf.predict_proba(X_test)[:, 1]

    # SVM Classifier training and prediction
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_pred_svm_proba = svm.predict_proba(X_test)[:, 1]

    # LSTM Classifier training and prediction
    lstm_model = create_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
    lstm_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64, verbose=0)
    y_pred_lstm = (lstm_model.predict(X_test_reshaped) > 0.5).astype(int)
    y_pred_lstm_proba = lstm_model.predict(X_test_reshaped).flatten()
    
    # Now we will be inputting the confusion matrix and formulas for each algorithm
    # Below we are solving the confusion matrix in order to calculate accuracy variables i.e. TP, TN, FP, FN for Random Forest
    conf_matr_rf = confusion_matrix(y_test, y_pred_rf)
    TN = conf_matr_rf[0, 0]
    FP = conf_matr_rf[0, 1]
    FN = conf_matr_rf[1, 0]
    TP = conf_matr_rf[1, 1]
    
    # Jotting down all the important formulas needed for the output calculations for Random Forest
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    TPR = TP/(TP+FN)  # Sensitivity
    TNR = TN/(TN+FP)  # Specificity
    Precision = TP/(TP+FP)
    F1_Score = 2*(Precision*TPR)/(Precision+TPR)
    Error_Rate = (FP+FN)/(TP+TN+FP+FN)
    BACC = (TPR+TNR)/2
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    TSS = TPR-FPR
    BS = brier_score_loss(y_test, y_pred_rf_proba)
    BSS = 1-BS/np.var(y_test)
    AUC = roc_auc_score(y_test, y_pred_rf_proba)
    HSS = 2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    
    # Below we are solving the confusion matrix in order to calculate accuracy variables i.e. TP, TN, FP, FN for SVM
    conf_matr_svm = confusion_matrix(y_test, y_pred_svm)
    TN = conf_matr_svm[0, 0]
    FP = conf_matr_svm[0, 1]
    FN = conf_matr_svm[1, 0]
    TP = conf_matr_svm[1, 1]
    
    # Jotting down all the important formulas needed for the output calculations for SVM
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    TPR = TP/(TP+FN)  # Sensitivity
    TNR = TN/(TN+FP)  # Specificity
    Precision = TP/(TP+FP)
    F1_Score = 2*(Precision*TPR)/(Precision+TPR)
    Error_Rate = (FP+FN)/(TP+TN+FP+FN)
    BACC = (TPR+TNR)/2
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    TSS = TPR-FPR
    BS = brier_score_loss(y_test, y_pred_svm_proba)
    BSS = 1-BS/np.var(y_test)
    AUC = roc_auc_score(y_test, y_pred_svm_proba)
    HSS = 2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    
    # Below we are solving the confusion matrix in order to calculate accuracy variables i.e. TP, TN, FP, FN for LSTM
    conf_matr_lstm = confusion_matrix(y_test, y_pred_lstm) # LSTM Confusion Matrix
    TN = conf_matr_lstm[0, 0]
    FP = conf_matr_lstm[0, 1]
    FN = conf_matr_lstm[1, 0]
    TP = conf_matr_lstm[1, 1]
    
    # Jotting down all the important formulas needed for the output calculations for LSTM
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    TPR = TP/(TP+FN)  # Sensitivity
    TNR = TN/(TN+FP)  # Specificity
    Precision = TP/(TP+FP) if(TP+FP)!=0 else 0
    F1_Score = 2*(Precision*TPR)/(Precision+TPR) if(Precision+TPR)>0 else 0 # Here I set this if statement because sometimes it was giving an error for diving by 0
    Error_Rate = (FP+FN)/(TP+TN+FP+FN)
    BACC = (TPR+TNR)/2
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    TSS = TPR-FPR
    BS = brier_score_loss(y_test, y_pred_lstm_proba)
    BSS = 1-BS/np.var(y_test)
    AUC = roc_auc_score(y_test, y_pred_lstm_proba)
    HSS = 2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    
    # Using the append feature to add the variables to the list with the values of Random Forest
    rf_metrics.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': Accuracy,
        'Sensitivity (TPR)': TPR,
        'Specificity (TNR)': TNR,
        'Precision': Precision,
        'F1 Score': F1_Score,
        'Error Rate': Error_Rate,
        'Balanced Accuracy (BACC)': BACC,
        'FPR': FPR,
        'FNR': FNR,
        'Brier Score (BS)': BS,
        'Brier Skill Score (BSS)': BSS,
        'AUC': AUC,
        'HSS': HSS,
        'TSS': TSS
    })
    # Using the append feature to add the variables to the list with the values of SVM
    svm_metrics.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': Accuracy,
        'Sensitivity (TPR)': TPR,
        'Specificity (TNR)': TNR,
        'Precision': Precision,
        'F1 Score': F1_Score,
        'Error Rate': Error_Rate,
        'Balanced Accuracy (BACC)': BACC,
        'FPR': FPR,
        'FNR': FNR,
        'Brier Score (BS)': BS,
        'Brier Skill Score (BSS)': BSS,
        'AUC': AUC,
        'HSS': HSS,
        'TSS': TSS
    })
    # Using the append feature to add the variables to the list with the values of LSTM
    lstm_metrics.append({
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': Accuracy,
        'Sensitivity (TPR)': TPR,
        'Specificity (TNR)': TNR,
        'Precision': Precision,
        'F1 Score': F1_Score,
        'Error Rate': Error_Rate,
        'Balanced Accuracy (BACC)': BACC,
        'FPR': FPR,
        'FNR': FNR,
        'Brier Score (BS)': BS,
        'Brier Skill Score (BSS)': BSS,
        'AUC': AUC,
        'HSS': HSS,
        'TSS': TSS
    })
#------------------------RF--------------------------------------
# Finding the average of the variables among all folds of Random Forest
avg_rf = {key: np.mean([metrics[key] for metrics in rf_metrics]) for key in rf_metrics[0]}

# With the previously calculated values from the folds, we create an empty dataframe and fill it with the metrics
rf_metrics_df = pd.DataFrame(rf_metrics)
rf_metrics_df.index = [f"Fold {i+1}" for i in range(len(rf_metrics))]

# Creating an empty dataframe for the average value dataset
avg_rf_df = pd.DataFrame([avg_rf], index=["Average"])

# Combining both the dataframes into a single one
combined_rf_df = pd.concat([rf_metrics_df, avg_rf_df])

# Finding the mean metrics for a seperate table below
avg_rf_metrics = rf_metrics_df.mean()

# Output the combined Random Forest dataset
print("Random Forest Classifier Metrics:")
print(combined_rf_df)
print("\nAverage Values Among All Folds:")
print(avg_rf_metrics)
print("\n\n")

#-----------------------SVM--------------------------------------
# Finding the average of the variables among all folds of SVM
avg_svm = {key: np.mean([metrics[key] for metrics in svm_metrics]) for key in svm_metrics[0]}

# With the previously calculated values from the folds, we create an empty dataframe and fill it with the metrics
svm_metrics_df = pd.DataFrame(svm_metrics)
svm_metrics_df.index = [f"Fold {i+1}" for i in range(len(svm_metrics))]

# Creating an empty dataframe for the average value dataset
avg_svm_df = pd.DataFrame([avg_svm], index=["Average"])

# Combining both the dataframes into a single one
combined_svm_df = pd.concat([svm_metrics_df, avg_svm_df])

# Finding the mean metrics for a seperate table below
avg_svm_metrics = svm_metrics_df.mean()

# Output the combined SVM dataset
print("SVM Metrics:")
print(combined_svm_df)
print("\nAverage Values Among All Folds:")
print(avg_svm_metrics)
print("\n\n")

#------------------------LSTM-------------------------------------
# Finding the average of the variables among all folds of LSTM
avg_lstm = {key: np.mean([metrics[key] for metrics in lstm_metrics]) for key in lstm_metrics[0]}

# With the previously calculated values from the folds, we create an empty dataframe and fill it with the metrics
lstm_metrics_df = pd.DataFrame(lstm_metrics)
lstm_metrics_df.index = [f"Fold {i+1}" for i in range(len(lstm_metrics))]

# Creating an empty dataframe for the average value dataset
avg_lstm_df = pd.DataFrame([avg_lstm], index=["Average"])

# Combining both the dataframes into a single one
combined_lstm_df = pd.concat([lstm_metrics_df, avg_lstm_df])

# Finding the mean metrics for a seperate table below
avg_lstm_metrics = lstm_metrics_df.mean()

#Output the combined LSTM dataset
print("LSTM Metrics:")
print(combined_lstm_df)
print("\nAverage Values Among All Folds:")
print(avg_lstm_metrics)
print("\n")

# Ending time
end_time = time.time()
total_duration = end_time-start_time # Calculating and determining the total duration it took to output the code
print(f"Time taken to output: {total_duration:.2f} seconds") # Outputting the total time for complete execution


# In[ ]:




