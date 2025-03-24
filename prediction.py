pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("skin_disease_data.csv")

# Preprocessing the data
# Convert categorical data to numerical values
data['Skin_Type'] = data['Skin_Type'].map({'Oily': 0, 'Dry': 1, 'Normal': 2})
data['Symptom_1'] = data['Symptom_1'].map({'Redness': 0, 'Scaling': 1, 'Blisters': 2})
data['Symptom_2'] = data['Symptom_2'].map({'Itching': 0, 'Pain': 1})

# Features and target
X = data[['Age', 'Skin_Type', 'Symptom_1', 'Symptom_2']]
y = data['Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example of making a prediction
sample_input = [[30, 1, 1, 0]]  # Example: Age = 30, Skin_Type = Dry, Symptom_1 = Scaling, Symptom_2 = Itching
prediction = model.predict(sample_input)
print(f"Predicted Disease: {prediction[0]}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("skin_disease_data.csv")

# Preprocessing the data
data['Skin_Type'] = data['Skin_Type'].map({'Oily': 0, 'Dry': 1, 'Normal': 2})
data['Symptom_1'] = data['Symptom_1'].map({'Redness': 0, 'Scaling': 1, 'Blisters': 2})
data['Symptom_2'] = data['Symptom_2'].map({'Itching': 0, 'Pain': 1})

# Features and target
X = data[['Age', 'Skin_Type', 'Symptom_1', 'Symptom_2']]
y = data['Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Plot feature importance
feature_importance = model.feature_importances_

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.show()
disease_counts = data['Disease'].value_counts()

plt.pie(disease_counts, labels=disease_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Distribution of Skin Diseases in Dataset')
plt.show()
import seaborn as sns

# Create correlation matrix
correlation_matrix = data[['Age', 'Skin_Type', 'Symptom_1', 'Symptom_2']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
import seaborn as sns

# Pairplot to visualize relationships between features
sns.pairplot(data[['Age', 'Skin_Type', 'Symptom_1', 'Symptom_2', 'Disease']], hue='Disease')
plt.title('Pairplot of Features for Skin Disease Prediction')
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming you have trained your model and have predictions
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix for Skin Disease Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
