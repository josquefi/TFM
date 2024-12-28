# Models de Machine Learning per a la Predicció d'Abandonament de Clients en el Sector Assegurador

Aquest repositori conté el codi del Treball Final de Màster (TFM) sobre la predicció d'abandonament de clients en el sector assegurador utilitzant tècniques de machine learning.

## Estructura del Projecte

El projecte està estructurat en tres parts principals:

### 1. Anàlisi Exploratòria de les Dades

- `1_2_eda.py`: Realitza l'anàlisi exploratòria de les dades.

### 2. Enginyeria de Variables

- `1_1_1_kmeans.py`: Implementa l'algoritme K-means per a la generació de noves variables.
- `1_1_2_outliers.py`: Detecta i gestiona els valors atípics en les dades.
- `1_1_preprocessing.py`: Realitza el preprocessament de les dades.
- `1_3_variable_selection_boruta.py`: Utilitza l'algoritme Boruta per a la selecció de variables.

### 3. Entrenament i Avaluació dels Models

- `2_train_logit.py`: Entrena un model de regressió logística.
- `2_train_random_forest.py`: Implementa un model Random Forest.
- `2_train_random_forest_random_search.py`: Optimitza els hiperparàmetres del Random Forest mitjançant Random Search.
- `2_train_xgboost.py`: Entrena un model XGBoost.
- `2_train_xgboost_random_search.py`: Optimitza els hiperparàmetres del XGBoost mitjançant Random Search.
