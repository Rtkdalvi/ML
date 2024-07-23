import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create the dataset
data = {
'Temp': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
'Humidity': [85, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91],
'Wind Speed': [12, 9, 4, 3, 5, 20, 2, 12, 5, 2, 3, 4, 5, 15],
'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Step 2: Encode the target variable
df['Play'] = df['Play'].map({'No': 0, 'Yes': 1})

# Step 3: Split the data into features and target
X = df.drop('Play', axis=1)
y = df['Play']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Step 5: Implement the KNN algorithm
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 6: Predict and evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the test set with predictions
test_set_with_predictions = X_test.copy()
test_set_with_predictions['Actual'] = y_test
test_set_with_predictions['Predicted'] = y_pred
print(test_set_with_predictions)
