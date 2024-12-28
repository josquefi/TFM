#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

for choose_dataset in range(1, 7):

    if choose_dataset == 1:
        # Dataset 1
        data = pd.read_csv('data/dataset_v1.csv')
    elif choose_dataset == 2:
        # Dataset 2
        data = pd.read_csv('data/dataset_v1_wo.csv')
    elif choose_dataset == 3:
        # Dataset 3
        data = pd.read_csv('data/dataset_v2.csv')
    elif choose_dataset == 4:
        # Dataset 4
        data = pd.read_csv('data/dataset_v2_wo.csv')
    elif choose_dataset == 5:    
        #Dataset 5
        data = pd.read_csv('data/dataset_v1.csv')
        # Selecció de variables basada en el resultat de Boruta
        data =  data[['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 
                      'has_children', 'length_of_residence', 'home_owner', 
                      'college_degree', 'home_market_value', 'Churn']]
    elif choose_dataset == 6:
        #Dataset 6
        data = pd.read_csv('data/dataset_v1_wo.csv')
        # Selecció de variables basada en el resultat de Boruta
        data =  data[['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 
                      'has_children', 'length_of_residence', 'home_owner', 
                      'college_degree', 'home_market_value', 'Churn']]
    
    
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apliquem SMOTE a conjunt d'entrenament
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Estadaritzem les variables
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)
    
    # Entrenem el model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Prediccions
    y_pred = model.predict(X_test)
    
    # Evaluació del model
    
    # Calculem AUC
    y_pred_proba = model.predict_proba(X_test)[::,1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print('-------------')
    print(f"Resultats dataset {choose_dataset}")
    print('-------------')
    print(f"auc: {auc:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"sensitivity: {sensitivity:.4f}")
    print(f"specificity: {specificity:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"kappa: {kappa:.4f}")
    
    coefficients = model.coef_[0]
    feature_names = X.columns
    coeff_dict = dict(zip(feature_names, coefficients))
    
    print("Coeficients:")
    for feature, coeff in coeff_dict.items():
        print(f"{feature}: {coeff:.4f}")
