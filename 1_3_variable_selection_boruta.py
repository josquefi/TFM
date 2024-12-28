#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/dataset_v2_wo.csv')

X = data.drop(columns=['Churn'])
y = data['Churn']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)

boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)

boruta_selector.fit(X_train.values, y_train)

selected_features = X.columns[boruta_selector.support_].tolist()
rejected_features = X.columns[~boruta_selector.support_].tolist()

print("Selected Features:")
print(selected_features)

print("\nRejected Features:")
print(rejected_features)

# Obtenim la importància de cada variable

feature_ranking = boruta_selector.ranking_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': feature_ranking
}).sort_values(by='Ranking')
print(feature_importance_df)

