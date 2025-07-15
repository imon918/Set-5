# housing_price_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset with validation
try:
    df = pd.read_csv("housing.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: housing.csv file not found.")
    exit()

# Check for null values
print("\n🔍 Missing values in each column:\n")
print(df.isnull().sum())

# Step 2: Split data into features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=["float64"]).columns.tolist()
categorical_features = ["ocean_proximity"]

# Step 3: Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Combine preprocessor with Linear Regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Step 4: Split the data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model.fit(X_train, y_train)
print("\n✅ Model training complete.")

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n📈 R-squared Score: {r2:.4f}")
print(f"📉 Mean Squared Error: {mse:.2f}")
