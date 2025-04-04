import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

dataset = pd.read_csv("datasets/Training1.csv")

X = dataset.drop('diseases', axis=1)
y = dataset['diseases']

# ecoding prognonsis
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)
    
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
print(X_train.shape)

svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
ypred = svc.predict(X_test)
print(accuracy_score(y_test,ypred))

# save svc
import pickle
pickle.dump(svc,open('svc1.pkl','wb'))

# load model
svc = pickle.load(open('svc1.pkl','rb'))
 
print("hi")