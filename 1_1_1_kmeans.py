#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:05:43 2024

@author: josepqueraltfibla
"""

from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv('data/dataset.csv')

#---------------------------------------------------
# K-Means
#---------------------------------------------------

city_churn = data.groupby('city')['Churn'].mean().reset_index()
kmeans = KMeans(n_clusters=10, random_state=42)
city_churn['city_cluster'] = kmeans.fit_predict(city_churn[['Churn']])
data = data.merge(city_churn[['city', 'city_cluster']], on='city', how='left')

#---------------------------------------------------
# One hot encoding
#---------------------------------------------------

data1 = pd.get_dummies(data, columns=['city_cluster'], drop_first=True)

del data1['city']

data1.to_csv('data/dataset_v1.csv', index=False)

#---------------------------------------------------
# Target mean
#---------------------------------------------------

city_cluster_means = data.groupby('city_cluster')['Churn'].mean()

data['city_cluster_encoded'] = data['city_cluster'].map(city_cluster_means)

del data['city']
del data['city_cluster']

data.to_csv('data/dataset_v2.csv', index=False)



