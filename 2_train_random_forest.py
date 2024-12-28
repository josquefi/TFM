#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE 
import warnings
warnings.filterwarnings("ignore")

# Utilitzem els paràmetres obtinguts del Random Search
best_params = {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 20, 'bootstrap': False}

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
        
    data = data.sample(frac=0.6)

    X = data.drop(columns=['Churn'])
    y = data['Churn']
    
    # Dividim en conjunts d'entrenament i test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apliquem SMOTE a conjunt d'entrenament
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Entrenem el model
    rf_model = RandomForestClassifier(random_state=42, **best_params)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # Fem les prediccions
    y_pred = rf_model.predict(X_test)
    
    # Evaluació del model
    
    # Calculem AUC
    y_pred_proba = rf_model.predict_proba(X_test)[::,1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    print('--------')
    print(f"Resultats dataset {choose_dataset}")
    print('--------')
    print(f"auc: {auc:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"sensitivity: {sensitivity:.4f}")
    print(f"specificity: {specificity:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"kappa: {kappa:.4f}")

