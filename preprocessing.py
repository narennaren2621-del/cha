import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("signals_features.csv")

# Remove duplicates
df = df.drop_duplicates()

# Convert boolean columns to integer
df["is_male"] = df["is_male"].astype(int)
df["chagas"] = df["chagas"].astype(int)

# Drop ID column
df = df.drop(columns=["exam_id"])

# Separate features and target
X = df.drop("chagas", axis=1)
y = df["chagas"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)