import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

print("Step 1: Loading Dataset...")
df = pd.read_csv('churn.csv')

# 1. Data Cleaning: Handle missing values and remove duplicates
df = df.dropna()
df = df.drop_duplicates()

if 'CustomerID' in df.columns:
    df = df.drop(columns=['CustomerID'])

# 2. Outlier Detection & Capping (IQR method for numerical columns)
num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# 3. Feature Engineering
df['Spend_Per_Tenure'] = df['Total Spend'] / (df['Tenure'] + 1)
df['Support_Delay_Ratio'] = df['Support Calls'] / (df['Payment Delay'] + 1)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Stratified sampling for robust & efficient training
df_sample = df.sample(n=60000, random_state=42)
X_s = df_sample.drop(columns=['Churn'])
y_s = df_sample['Churn']

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

print("\nStep 2: Training 4 Machine Learning Models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_score = 0
best_model_name = ""
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"-> {name} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    if f1 > best_score:
        best_score = f1
        best_model_name = name
        best_pipeline = pipeline

print(f"\nBest Initial Model Selected: {best_model_name}")

print("\nStep 3: Performing Hyperparameter Tuning (GridSearchCV)...")
if best_model_name == "Logistic Regression":
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0]
    }
elif best_model_name == "Decision Tree":
    param_grid = {
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif best_model_name == "Random Forest":
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 20, None]
    }
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1]
    }
else:
    param_grid = {'classifier__max_depth': [10, 20]}

grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test)
y_prob_final = final_model.predict_proba(X_test)[:, 1]

print("\n--- Final Tuned Model Evaluation ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_final):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_final):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))

print("\nStep 4: Saving Model using Joblib...")
joblib.dump(final_model, 'churn_model.pkl')
print("Model successfully saved as 'churn_model.pkl'!")