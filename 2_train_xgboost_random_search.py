
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data/dataset_v1.csv')

data =  data[['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 
              'has_children', 'length_of_residence', 'home_owner', 
              'college_degree', 'Churn']]

X = data.drop(columns=['Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]
}

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, 
                                    n_iter=50, scoring='recall', cv=3, verbose=1, n_jobs=-1, random_state=42)

random_search.fit(X_train_resampled, y_train_resampled)

best_params = random_search.best_params_
print(f"Millors paràmetres: {best_params}")

