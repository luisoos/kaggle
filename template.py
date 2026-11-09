import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

# Algorithms we'll combine for our ensemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# 1. Setup & Data Loading
# ==========================================
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
ID_COL = 'id'
TARGET_COL = 'target_column_name'

# Load the datasets
X = pd.read_csv(TRAIN_PATH, index_col=ID_COL)
X_test = pd.read_csv(TEST_PATH, index_col=ID_COL)

# Drop any rows where we are missing our target value
X.dropna(axis=0, subset=[TARGET_COL], inplace=True)

# Grab the target variable (convert text to numbers if needed, e.g., 'Yes'/'No' -> 1/0)
# y = X[TARGET_COL].map({'Yes': 1, 'No': 0})
y = X[TARGET_COL] 
X.drop([TARGET_COL], axis=1, inplace=True)

# ==========================================
# 2. Feature Engineering (Optional)
# ==========================================
def add_features(df):
    df = df.copy()
    # Drop any custom feature engineering here if you want to test new ideas
    # Example: df['New_Feature'] = df['Col1'] + df['Col2']
    return df

X = add_features(X)
X_test = add_features(X_test)

# ==========================================
# 3. Preprocessing Pipeline
# ==========================================
# Automatically figure out which columns are text/categorical vs numbers
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()

# Handle missing numbers by filling them with the median
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Optional: helpful for some models, but tree models usually don't care
])

# Handle missing categories and encode text into numbers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine everything into one tidy preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ==========================================
# 4. Handle Imbalanced Data (Class Weight)
# ==========================================
# If your dataset has way more 'No's than 'Yes's, this helps the models pay attention to the minority class
num_negative = (y == 0).sum()
num_positive = (y == 1).sum()
scale_weight = num_negative / num_positive
print(f"Class distribution -> Negatives: {num_negative}, Positives: {num_positive}")
print(f"Calculated scale_pos_weight: {scale_weight:.4f}\n")

# ==========================================
# 5. Define Individual Models
# ==========================================
# Feel free to plug in your Optuna-tuned parameters here later!
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ))
])

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    ))
])

cat_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0,
        thread_count=-1
    ))
])

# ==========================================
# 6. Build & Evaluate the Ensemble
# ==========================================
# Combine our three models into a voting classifier (soft voting averages their probabilities)
final_ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_pipeline),
        ('lgb', lgbm_pipeline),
        ('cat', cat_pipeline)
    ],
    voting='soft',
    weights=[1.0, 1.0, 1.0] # Adjust weights if one model performs significantly better
)

print("Running cross-validation to check local performance...")
cv_scores = cross_val_score(final_ensemble, X, y, cv=3, scoring='accuracy', n_jobs=-1)

print("\n--- Final Model Statistics ---")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}\n")

# ==========================================
# 7. Final Training & Submission Export
# ==========================================
print("Training the final ensemble on the full dataset...")
final_ensemble.fit(X, y)

print("Generating predictions in chunks to keep memory usage safe...")
chunks = np.array_split(X_test, 5)
predictions_list = []

for chunk in chunks:
    predictions_list.append(final_ensemble.predict(chunk))

test_predictions = np.concatenate(predictions_list)

# Put everything together into a clean submission file
submission = pd.DataFrame({
    ID_COL: X_test.index,
    TARGET_COL: test_predictions
})

submission.to_csv('submission_ensemble_template.csv', index=False)
print("All done! Saved your submission as 'submission_ensemble_template.csv'.")
```
