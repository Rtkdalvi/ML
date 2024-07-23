import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_text 
from sklearn import tree 
import matplotlib.pyplot as plt
# Sample dataset 
data = { 
 'age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Middle', 'Senior', 'Youth', 'Youth', 'Senior', 'Youth',  'Middle', 'Middle', 'Senior'], 
 'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium',  'medium', 'high', 'medium'], 
 'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'], 
 'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'], 
 'Buys_Computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'] } 
# Create DataFrame 
df = pd.DataFrame(data) 
# Convert categorical variables to numeric 
df['age'] = df['age'].astype('category').cat.codes 
df['income'] = df['income'].astype('category').cat.codes 
df['student'] = df['student'].astype('category').cat.codes 
df['credit_rating'] = df['credit_rating'].astype('category').cat.codes 
df['Buys_Computer'] = df['Buys_Computer'].astype('category').cat.codes 
# Features and target variable 
X = df.drop('Buys_Computer', axis=1) 
y = df['Buys_Computer'] 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
# Create Decision Tree classifier 
clf = DecisionTreeClassifier(criterion='entropy') 
# Train the classifier 
clf = clf.fit(X_train, y_train) 
# Predict the response for the test dataset 
y_pred = clf.predict(X_test) 
# Evaluate the model 
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)) 
print("\nDecision Tree:\n", export_text(clf, feature_names=list(X.columns))) 
#VISUALIZE THE DECISION TREE 
plt.figure(figsize=(20,10)) 
tree.plot_tree(clf,feature_names=X.columns, class_names=['No', 'Yes'], filled=True) plt.show()
