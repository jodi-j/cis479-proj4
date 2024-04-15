import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

over_df = pd.read_csv('overdrawn.csv')

# convert categorical data to int representations of unique categories
for col in over_df.columns:
    labels, uniques = pd.factorize(over_df[col])
    over_df[col] = labels
    
X = over_df.drop(columns='Overdrawn')
y = over_df['Overdrawn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Change DaysDrink into categorical data
conditions = [
(over_df['DaysDrink'] < 7),
(over_df['DaysDrink'] >= 14),
(over_df['DaysDrink'] >= 7) & (over_df['DaysDrink'] < 14)
]
categories = [0, 2, 1]
# Apply the conditions to create the categorical data
over_df['DaysDrink'] = np.select(conditions, categories)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=('Age', 'Sex', 'DaysDrink'),
                                class_names=('0','1'),
                                filled=True)
graph = graphviz.Source(dot_data, format="png")
graph.render('overdrawn_dt', view=True)