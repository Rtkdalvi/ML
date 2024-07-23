import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Training data
data = [
['Youth', 'High', 'No', 'Fair', 'No'],
['Youth', 'High', 'No', 'Excellent', 'No'],
['Middle-aged', 'High', 'No', 'Fair', 'Yes'],
['Senior', 'Medium', 'No', 'Fair', 'Yes'],
['Senior', 'Low', 'Yes', 'Fair', 'Yes'],
['Senior', 'Low', 'Yes', 'Excellent', 'No'],
['Middle-aged', 'Low', 'Yes', 'Excellent', 'Yes'],
['Youth', 'Medium', 'No', 'Fair', 'No'],
['Youth', 'Low', 'Yes', 'Fair', 'Yes'],
['Senior', 'Medium', 'Yes', 'Fair', 'Yes'],
['Youth', 'Medium', 'Yes', 'Excellent', 'Yes'],
['Middle-aged', 'Medium', 'No', 'Excellent', 'Yes'],
['Middle-aged', 'High', 'Yes', 'Fair', 'Yes'],
['Senior', 'Medium', 'No', 'Excellent', 'No']
]

# Separate the features and the target variable
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

# Encode categorical variables into numerical variables
label_encoders = []
for i in range(len(X[0])):
le = LabelEncoder()
column = [x[i] for x in X]
le.fit(column)
X = [[le.transform([x[i]])[0] if i==j else x[j] for j in range(len(x))] for x in X]
label_encoders.append(le)

# Encode the target variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)
# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Test data
test_data = ['Youth', 'Low', 'No', 'Fair']

# Encode the test data using the same label encoders
test_data_encoded = []
for i in range(len(test_data)):
test_data_encoded.append(label_encoders[i].transform([test_data[i]])[0])

# Make a prediction
prediction = model.predict([test_data_encoded])

# Decode the prediction
prediction_decoded = le_y.inverse_transform(prediction)
print(f"The prediction for the test data {test_data} is: {prediction_decoded[0]}")