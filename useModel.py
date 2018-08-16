from sklearn.externals import joblib

filename = 'noti.sav'
rf = joblib.load(filename)
print(rf.predict([[0,0,0,6,1601,1]]))