import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt

def convert_categorical(df, column, conditions, categories):
    df[column] = np.select(conditions, categories)

def print_seaborn_matrix(y_test, predictions):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Define class labels
    class_names = ['Not Overdrawn', 'Overdrawn']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def predict_overdrawn(query_data):
    prediction = dtree.predict(query_data)
    return "Yes" if prediction == [1] else "No"

over_df = pd.read_csv('overdrawn.csv')

# Convert categorical data to int representations of unique categories
for col in over_df.columns:
    labels, uniques = pd.factorize(over_df[col])
    over_df[col] = labels

# Change DaysDrink into categorical data
conditions = [
    (over_df['DaysDrink'] < 7),
    (over_df['DaysDrink'] >= 14),
    (over_df['DaysDrink'] >= 7) & (over_df['DaysDrink'] < 14)
]
categories = [0, 2, 1]
convert_categorical(over_df, 'DaysDrink', conditions, categories)
X = over_df.drop(columns='Overdrawn')
y = over_df['Overdrawn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
dot_data = export_graphviz(dtree, out_file=None,
                           feature_names=('Age', 'Sex', 'DaysDrink'),
                           class_names=('0', '1'),
                           filled=True, rounded=True, precision=0,
                           )

graph = graphviz.Source(dot_data, format="png")
graph.render('overdrawn_dt', view=True)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

queries = [
    {'Age': [20], 'Sex': [0], 'DaysDrink': [1]},
    {'Age': [25], 'Sex': [1], 'DaysDrink': [0]},
    {'Age': [19], 'Sex': [0], 'DaysDrink': [2]},
    {'Age': [22], 'Sex': [1], 'DaysDrink': [2]},
    {'Age': [21], 'Sex': [0], 'DaysDrink': [2]}
]

for i, query in enumerate(queries, start=1):
    prediction_text = predict_overdrawn(pd.DataFrame(query))
    print(f"Prediction {i}: Will the student overdraw a checking account? {prediction_text}")

print_seaborn_matrix(y_test, predictions)
