import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Read data from CSV file
data_path = "car_data.csv" # Replace with the actual path to your file

# Specify the correct target column name (replace with the actual name in your data)
target_column_name = "target" # Replace with the actual name

df = pd.read_csv(data_path, usecols=["buying", "price", "doors", "persons", "lug_boot", "safety", target_column_name])

# Handle missing values (if any)
# Replace the following with your preferred missing value handling strategy
# (e.g., impute with mean/median, remove rows/columns with missing values)
df.dropna(inplace=True)

# Identify categorical features
categorical_features = [
  "buying", "price", "doors", "persons", "lug_boot", "safety" # Assuming these are categorical
]

# Encode categorical features using OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = pd.DataFrame(encoder.fit_transform(df[categorical_features]))

# Concatenate the target variable with the encoded features DataFrame
encoded_df = pd.concat([df[target_column_name], encoded_features], axis=1)

# Separate features and target variable
features = encoded_df.iloc[:, 1:].values
target = encoded_df[target_column_name].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=0)

# Train decision tree model
model = DecisionTreeClassifier(criterion="entropy") # Entropy criterion for information gain calculation
model.fit(X_train, y_train)

# Make predictions and evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)