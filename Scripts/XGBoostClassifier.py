## Importing libraries
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import joblib
import os

# Reading the data
file_path = 'Data/train(1).csv'
df = pd.read_csv(file_path)

# Converting the predictor variable to binary values (1 for 'yes', 0 for 'no')
df['default_oct'] = df['default_oct'].apply(lambda x: 1 if x == 'yes' else 0)

# Splitting the data into features (X) and target variable (y)
X = df.drop('default_oct', axis=1)
y = df['default_oct']

########### Define the preprocessing pipeline ###########
# This pipeline includes KNN imputation for missing values and standard scaling for feature normalization
X.drop('customer_id', axis=1, inplace=True)  # Drop 'customer_id' column if present

preprocessor = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Apply the preprocessing pipeline to the data
X_preprocessed = preprocessor.fit_transform(X)

########### Initializing model and Parameters for RandomSearchCV ###########
# Calculate scale_pos_weight to handle class imbalance
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

# Define the initial XGBoost model
xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)

# Define the parameter distribution for hyperparameter optimization using RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.03, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 7, 9],
    'colsample_bytree': [0.3, 0.7],
    'subsample': [0.8, 1.0]
}

# Use stratified k-fold cross-validation for more robust evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Define the scoring metric as F1 score
f1_scorer = make_scorer(f1_score)

# Perform randomized search for hyperparameter optimization
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, scoring=f1_scorer, cv=skf, n_iter=10, random_state=7, verbose=1, n_jobs=-1)
random_search.fit(X_preprocessed, y)

# Extract the best parameters from the random search
best_params = random_search.best_params_

########### Training the Final Model ###########
# Train the final model with the best hyperparameters on the entire dataset
final_model = XGBClassifier(**best_params, objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
final_model.fit(X_preprocessed, y)

########### Save the Final Model ###########
# Including the preprocessing in the final pipeline. This ensures that the same preprocessing steps are applied during prediction
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', final_model)
])

# Save the final pipeline to a file
pipeline_path = 'Model/XGBoostClassifier_pipeline.joblib'  # Replace with the desired path to save the pipeline
joblib.dump(final_pipeline, pipeline_path)