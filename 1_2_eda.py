#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import skew, kurtosis, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/dataset.csv')
    
def categorical_features_hist(df, categorical_cols):
    for col in categorical_cols:
        print(f"\nColumna: {col}")
        print(df[col].value_counts(normalize=True) * 100)
        df[col].value_counts().plot(kind='bar', title=f'Histograma {col}')
        plt.show()
        
def numerical_features_statistics(df, numerical_cols):
    for col in numerical_cols:
        print(f"\nColumna: {col}")
        print(f"Mean: {df[col].mean()}, Median: {df[col].median()}, Std: {df[col].std()}")
        print(f"Max: {df[col].max()}, Min: {df[col].min()}")
        print(f"Skew: {skew(df[col].dropna())}, Kurtosis: {kurtosis(df[col].dropna())}")

def numerical_features_hist(df, numerical_cols):
    for col in numerical_cols:
        sns.kdeplot(df[col], label=f"{col}", fill=True, alpha=0.5)
        plt.title(f"Histograma {col}")
        plt.show()

def numerical_vs_target_hist(df, numerical_columns, target_column):
    for column in numerical_columns:
        plt.figure(figsize=(8, 5))
        for value in df[target_column].unique():
            subset = df[df[target_column] == value]
            sns.kdeplot(subset[column], label=f"{column} = {value}", fill=True, alpha=0.5)
        plt.title(f"Histograma {column} vs {target_column}")
        plt.xlabel(column)
        plt.ylabel("Freqüència")
        plt.legend()
        plt.show()

def chi_square_test(df, categorical_columns, target_column):
    results = {}
    for column in categorical_columns:
        contingency_table = pd.crosstab(df[column], df[target_column])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        results[column] = {"Chi2": chi2, "p-value": p}
    results_df = pd.DataFrame(results).T
    print("\n Resultats Chi-Square:")
    print(results_df)

def numerical_vs_categorical_hist(df, numerical_columns, categorical_columns):
    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            plt.figure(figsize=(8, 5))
            for value in df[cat_col].unique():
                subset = df[df[cat_col] == value]
                sns.kdeplot(subset[num_col], label=f"{cat_col} = {value}", fill=True, alpha=0.5)
            plt.title(f"Histograma {num_col} per {cat_col}")
            plt.xlabel(num_col)
            plt.ylabel("Densitat")
            plt.legend(title=cat_col)
            plt.show()

def numerical_correlation(df, numerical_columns):
    correlation_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Matriu de correlació")
    plt.show()

        
numerical_cols = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'home_market_value', 'length_of_residence']
categorical_nominal_cols = ['city']
categorical_binary_cols = ['marital_status', 'home_owner', 'college_degree', 'good_credit', 'has_children']

categorical_features_hist(data, categorical_nominal_cols)
categorical_features_hist(data, categorical_binary_cols)
numerical_features_hist(data, numerical_cols)
numerical_features_statistics(data, numerical_cols)

numerical_vs_target_hist(data, numerical_cols, 'Churn')
chi_square_test(data, categorical_nominal_cols, 'Churn')
chi_square_test(data, categorical_binary_cols, 'Churn')

numerical_vs_categorical_hist(data, numerical_cols, categorical_binary_cols)
numerical_correlation(data, numerical_cols)



