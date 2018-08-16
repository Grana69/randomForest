# Pandas is used for data manipulation
import pandas as pd
from datetime import datetime
# Read in data and display first 8 rows
headers = ['haveBusy','busyStart','busyEnd','day','timeSent','timeAnswered','answered']
dtypes = [bool,datetime,datetime,int,datetime,datetime,bool]
features = pd.read_csv('noti.csv')
#print(features.head(8))

#print('The shape of our features is:', features.shape)

# Descriptive statistics for each column
#print(features.describe(include='all'))


features = pd.get_dummies(features,prefix_sep="_",columns=headers)
print(features)
# # Display the first 4 rows of the last 8 columns
# print(features.iloc[:, 4:].head(8))

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
#labels = np.array(features['busyStart'], dtype='M8[h]')
#print(labels)
# # Remove the labels from the features
# # axis 1 refers to the columns
# features = features.drop('timeSent', axis=1)

# # Saving feature names for later use
# feature_list = list(features.columns)
# # Convert to numpy array
# features = np.array(features)


# # Using Skicit-learn to split data into training and testing sets
# from sklearn.model_selection import train_test_split

# # Split the data into training and testing sets
# train_features, test_features, train_labels, test_labels = train_test_split(
#     features, labels, test_size=0.25, random_state=42)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)


# # The baseline predictions are the historical averages
# baseline_preds = test_features[:, feature_list.index('timeAnswered')]

# # Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_labels)

# print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# # Import the model we are using
# from sklearn.ensemble import RandomForestRegressor

# # Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# # Train the model on training data
# rf.fit(train_features, train_labels)

# from sklearn.externals import joblib
# filename = 'finalized_model.sav'
# joblib.dump(rf, filename)

# # Use the forest's predict method on the test data
# predictions = rf.predict(test_features)
# print(rf.predict([[1,23,4]]))

# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')

# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Import tools needed for visualization
# from sklearn.tree import export_graphviz
# import pydot
# # Pull out one tree from the forest
# tree = rf.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file='tree.dot',
#                 feature_names=feature_list, rounded=True, precision=1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png')


# # Limit depth of tree to 3 levels
# rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
# rf_small.fit(train_features, train_labels)
# # Extract the small tree
# tree_small = rf_small.estimators_[5]
# # Save the tree as a png image
# export_graphviz(tree_small, out_file='small_tree.dot',
#                 feature_names=feature_list, rounded=True, precision=1)
# (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('small_tree_small.png')


