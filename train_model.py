# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load your training data
df = pd.read_csv('loan_approval_dataset.csv')  # Update path to your actual data file

# Clean up any extra spaces in column names
df.columns = df.columns.str.strip()

# Clean up spaces in the target variable (loan_status)
df['loan_status'] = df['loan_status'].str.strip()

# Handle NaN values in the target variable 'loan_status'
df['loan_status'] = df['loan_status'].fillna(0)  # Replace NaN with 0 (Rejected)
# Alternatively, if you want to drop rows with NaN in 'loan_status':
# df = df.dropna(subset=['loan_status'])

# If loan_status is categorical, map ' Approved' to 1 and ' Rejected' to 0
df['loan_status'] = df['loan_status'].map({' Approved': 1, ' Rejected': 0})

# Handle NaN values in feature columns if needed
df.fillna(0, inplace=True)  # Replace NaNs in all columns with 0

# Check if the target column exists
if 'loan_status' not in df.columns:
    raise ValueError("The target column 'loan_status' is not found in the dataset.")

# Separate features (X) and target (y)
X_train = df.drop('loan_status', axis=1)  # Features
y_train = df['loan_status']  # Target variable

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                                   'residential_assets_value', 'commercial_assets_value',
                                   'luxury_assets_value', 'bank_asset_value']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['education', 'self_employed', 'no_of_dependents'])
    ]
)

# Define the Random Forest model
model = RandomForestClassifier()

# Create a pipeline that first preprocesses and then applies the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Save the entire pipeline (model + preprocessing steps) to a .pkl file
joblib.dump(pipeline, 'loan_model_pipeline.pkl')

print("Model trained and pipeline saved successfully!")

