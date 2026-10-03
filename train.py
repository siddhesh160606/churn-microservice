import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Mock Dataset (Simulating customer subscription data)
data = pd.DataFrame(
    {
        'tenure': [1, 12, 24, 5, 36, 2, 48, 3, 18, 60],
        'monthly_charges': [
            70.5,
            50.0,
            85.2,
            65.0,
            20.0,
            90.0,
            110.0,
            30.0,
            75.0,
            25.0,
        ],
        'contract': [
            'month-to-month',
            'one-year',
            'two-year',
            'month-to-month',
            'two-year',
            'month-to-month',
            'two-year',
            'month-to-month',
            'one-year',
            'two-year',
        ],
        'churn': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    }
)

X = data.drop('churn', axis=1)
y = data['churn']

# 2. Define Preprocessing Pipeline for Numerical and Categorical Features
num_cols = ['tenure', 'monthly_charges']
cat_cols = ['contract']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ]
)

# 3. Combine Preprocessor and Model into a Single Pipeline
model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'classifier',
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
    ]
)

# 4. Train the Pipeline
model_pipeline.fit(X, y)

# 5. Export the Trained Pipeline as a .pkl file
joblib.dump(model_pipeline, 'model_pipeline.pkl')
print('✅ Success: Model pipeline trained and saved as model_pipeline.pkl!')