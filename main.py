
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Step 1: Load Breast Cancer Dataset
# ==============================
DATA_PATH = "Cancer Detection dataset.csv"
df = pd.read_csv(DATA_PATH)

print("First 5 Records:")
print(df.head())

# ==============================
# Step 2: Data Cleaning
# ==============================

# Remove ID column if it exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Remove unnamed column if present
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# ==============================
# Step 3: Feature Selection
# ==============================

features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean'
]

X = df[features]

# ==============================
# Step 4: Label Encoding
# ==============================

# B = 0, M = 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

y = df['diagnosis']

# ==============================
# Step 5: Split Dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==============================
# Step 6: Train Model
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==============================
# Step 7: Make Predictions
# ==============================

y_pred = model.predict(X_test)

# ==============================
# Step 8: Evaluate Model
# ==============================

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))