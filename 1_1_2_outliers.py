#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:29:09 2024

@author: josepqueraltfibla
"""

import pandas as pd

def remove_outliers(df, columns):

    for column in columns:
        if column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        else:
            print(f"Warning: '{column}' no existeix.")
    return df

numerical_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'home_market_value', 'length_of_residence']

#---------------------------------------------------
# 1. dataset_v1
#---------------------------------------------------

df_v1 = pd.read_csv('data/dataset_v1.csv')

df_v1 = remove_outliers(df_v1, numerical_cols)

df_v1.to_csv('data/dataset_v1_wo.csv', index=False)

#---------------------------------------------------
# 2. dataset_v2
#---------------------------------------------------

df_v2 = pd.read_csv('data/dataset_v2.csv')

df_v2 = remove_outliers(df_v2, numerical_cols)

df_v2.to_csv('data/dataset_v2_wo.csv', index=False)


