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

#Query 1: Predict whether a 20-year-old male student who has drunk alcohol for 10 days in the past 30 
#         days will overdraw a checking account.
query_1_data = pd.DataFrame({'Age': [20],
                             'Sex': [0], 
                             'DaysDrink': [1]}) 
prediction_1 = dtree.predict(query_1_data)
if (prediction_1 == [1]):
    pred_1_text = "Yes"
elif (prediction_1 == [0]):
    pred_1_text = "No"
print("Prediction 1: Will the student overdraw a checking account?", pred_1_text)

#Query 2:Predict whether a 25-year-old female student who has drunk alcohol for 5 days in the past 30 
#         days will overdraw a checking account.
query_2_data = pd.DataFrame({'Age': [25],
                             'Sex': [1],
                             'DaysDrink': [0]})
prediction_2 = dtree.predict(query_2_data)
if (prediction_2 == [1]):
    pred_2_text = "Yes"
elif (prediction_2 == [0]):
    pred_2_text = "No"
print("Prediction 2: Will the student overdraw a checking account?", pred_2_text)

#Query 3: Predict whether a 19-year-old male student who has drunk alcohol for 20 days in the past 30 
#         days will overdraw a checking account.
query_3_data = pd.DataFrame({'Age': [19],
                             'Sex': [0],
                             'DaysDrink': [2]})
prediction_3 = dtree.predict(query_3_data)
if (prediction_3 == [1]):
    pred_3_text = "Yes"
elif (prediction_3 == [0]):
    pred_3_text = "No"
print("Prediction 3: Will the student overdraw a checking account?", pred_3_text)

#Query 4: Predict whether a 22-year-old female student who has drunk alcohol for 15 days in the past 30 
#         days will overdraw a checking account.
query_4_data = pd.DataFrame({'Age': [22],
                             'Sex': [1],
                             'DaysDrink': [2]})
prediction_4 = dtree.predict(query_4_data)
if (prediction_4 == [1]):
    pred_4_text = "Yes"
elif (prediction_4 == [0]):
    pred_4_text = "No"
print("Prediction 4: Will the student overdraw a checking account?", pred_4_text)

#Query 5: Predict whether a 21-year-old male student who has drunk alcohol for 20 days in the past 30 
#         days will overdraw a checking account.
query_5_data = pd.DataFrame({'Age': [21],
                             'Sex': [0],
                             'DaysDrink': [2]})
prediction_5 = dtree.predict(query_5_data)
if (prediction_5 == [1]):
    pred_5_text = "Yes"
elif (prediction_5 == [0]):
    pred_5_text = "No"
print("Prediction 5: Will the student overdraw a checking account?", pred_5_text)