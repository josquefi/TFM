#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:13:41 2024

@author: josepqueraltfibla
"""

import pandas as pd
import random

# Carregar el joc de dades
df = pd.read_csv('data/autoinsurance_churn.csv')


#---------------------------------------------------
# Descripció variables
#---------------------------------------------------

# Fem una primera inspecció de les dades i les classifiquem segons el seu tipus.
df.dtypes

numerical_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'home_market_value', 'length_of_residence']
categorical_nominal_cols = ['city', 'state', 'county']
categorical_binary_cols = ['marital_status', 'home_owner', 'college_degree', 'good_credit', 'has_children']
date_cols = ['date_of_birth', 'cust_orig_date', 'acct_suspd_date']

#---------------------------------------------------
# Eliminen duplicats si existiren
#---------------------------------------------------

df.drop_duplicates(inplace=True)

#---------------------------------------------------
# Comprovem que tots els valors que contenen les columnes es corresponen amb el seu tipus
#---------------------------------------------------

def check_column_data_types(df, numerical_cols, categorical_nominal_cols, categorical_binary_cols, date_cols):
    issues = {}

    # Columnes numèriques
    for col in numerical_cols:
        if col in df.columns:
            non_numeric = df[~df[col].apply(lambda x: isinstance(x, (int, float, type(None))))][col]
            if not non_numeric.empty:
                issues[col] = {
                    "tipus": "Numèric",
                    "valors_invalids": non_numeric.unique().tolist()
                }
    
    # Columnes categòriques
    for col in categorical_nominal_cols:
        if col in df.columns:
            non_string = df[~df[col].apply(lambda x: isinstance(x, str) or pd.isnull(x))][col]
            if not non_string.empty:
                issues[col] = {
                    "tipus": "Categòriques (Nominals)",
                    "valors_invalids": non_string.unique().tolist()
                }

    # Columnes categòriques binàries
    for col in categorical_binary_cols:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 2:
                issues[col] = {
                    "tipus": "Categòriques (Binàries)",
                    "valors_invalids": unique_values.tolist()
                }
    
    # Columnes datetime
    for col in date_cols:
        if col in df.columns:
            invalid_dates_mask = ~pd.to_datetime(df[col], errors='coerce').notnull() & df[col].notnull()
            non_dates = df.loc[invalid_dates_mask, col]
            if not non_dates.empty:
                issues[col] = {
                    "tipus": "Data",
                    "valors_invalids": non_dates.unique().tolist()
                }
    return issues


issues = check_column_data_types(df, numerical_cols, categorical_nominal_cols, categorical_binary_cols, date_cols)

if issues:
    for col, details in issues.items():
        print(f"Columna: {col}")
        print(f"Tipus: {details['tipus']}")
        print(f"Valors: {details['valors_invalids']}")
else:
    print("No es detecten errors.")

#---------------------------------------------------    
# home_market_value
#---------------------------------------------------
    
# Com podem veure en l'anàlisi anterior tenim alguns valor en la columna home_market_value
# que es corresponenamb un rang de valors. He pres la decisió de substituir aquest rang pel
# pel seu punt mitja.

def home_market_value_a_numeric(range_str):
    try:
        if 'Plus' in range_str:
            return 1000000
        lower, upper = map(int, range_str.split(' - '))
        return random.randint(lower, upper)
    except Exception as e:
        print(f"Error: {range_str}, {e}")
        return None

df['home_market_value'] = df['home_market_value'].apply(lambda x: home_market_value_a_numeric(x) if isinstance(x, str) else x)

#---------------------------------------------------
# Eliminació de valors potencialment erronis
#---------------------------------------------------

# Eliminem edats de més de 100 anys perquè lo més probable és que siguin deguts a errors
df = df[df.age_in_years < 100]

# Eliminem el 75% de les files dels clients que tenen 55 anys per l'excés que representen sobre el total del conjunt de dades
rows_to_check = df[df['age_in_years'] == 55]

if not rows_to_check.empty:
    num_to_delete = int(len(rows_to_check) * 0.75)
    rows_to_delete = rows_to_check.sample(n=num_to_delete).index
else:
    rows_to_delete = []
df = df.drop(index=rows_to_delete)

# Eliminem valors negatius de curr_ann_amt
df = df[df.curr_ann_amt > 0]

#---------------------------------------------------
# Eliminació de features
#---------------------------------------------------

# Eliminem la columna perquè no aporta informació per a la predicció

del df['acct_suspd_date']
del df['individual_id']
del df['address_id']
del df['state']
del df['county']
del df['date_of_birth']
del df['cust_orig_date']

#---------------------------------------------------
# Missing values
#---------------------------------------------------

# Comprovació null values
df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)

# Eliminem les colunes que contenen massa valors null
del df['latitude']
del df['longitude']

# Imputem la moda a les columnes categòriques que tenen un percentatge petit de missing values
df['city'] = df['city'].fillna(df['city'].mode()[0])

# Imputem la mediana a home_market_value perquè és una variables que segueix una distribució lognormal
df['home_market_value'] = df['home_market_value'].fillna(df['home_market_value'].median())

# Verifiquem si queden valors null
df.isnull().sum()

#---------------------------------------------------
# Variables categòriques binàries
#---------------------------------------------------

mapping = {'Married': 1, 'Single': 0}
df['marital_status'] = df['marital_status'].map(mapping)

df['has_children'] = df['has_children'].astype(int)
df['home_owner'] = df['home_owner'].astype(int)
df['college_degree'] = df['college_degree'].astype(int)
df['good_credit'] = df['good_credit'].astype(int)

#---------------------------------------------------
# Guardem el joc de dades després del preprocessament
#---------------------------------------------------

df.to_csv('data/dataset.csv', index=False)
