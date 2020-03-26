import pandas as pd
df = pd.read_csv('Cleaned-Data.csv')

import numpy as np
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size=int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train, test = data_split(df, 0.2)

x_train = train[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion',
 'Runny-Nose', 'Diarrhea', 'Gender_Female', 'Gender_Male', 'Gender_Transgender', 'Severity_Mild', 'Severity_Moderate',
  'Severity_Severe', 'Contact_Dont-Know', 'Contact_No', 'Contact_Yes']].to_numpy()
#print(x_train)

x_test = test[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion',
 'Runny-Nose', 'Diarrhea', 'Gender_Female', 'Gender_Male', 'Gender_Transgender', 'Severity_Mild', 'Severity_Moderate',
  'Severity_Severe', 'Contact_Dont-Know', 'Contact_No', 'Contact_Yes']].to_numpy()
#print(x_test)

y_train = train[['Severity_Severe']].to_numpy().reshape(253440 ,)
#print(y_train)  

y_test = test[['Severity_Severe']].to_numpy().reshape(63360 ,)
#print(y_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
print(clf.fit(x_train, y_train))

inputfeatures = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
infprob = clf.predict_proba([inputfeatures])[0][1]
print(infprob)

