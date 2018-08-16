import pandas as pd
from pandas import DataFrame
import numpy
series = pd.read_csv('nostis.csv')
#print(series['busyStart'])
labels = numpy.array(series['timeSent'])
print(labels)

series = series.drop('timeSent', axis=1)
feature_list = list(series.columns)
# # Convert to numpy array
series = numpy.array(series)
print(series)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    series, labels, test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('timeAnswered')]
print(baseline_preds)

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(numpy.mean(baseline_errors), 2))


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)

from sklearn.externals import joblib
filename = 'noti.sav'
joblib.dump(rf, filename)