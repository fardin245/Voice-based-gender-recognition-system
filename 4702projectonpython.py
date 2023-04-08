import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

voice = pd.read_csv('/voice.csv')

label_encode = LabelEncoder()
voice['label'] = label_encode.fit_transform(voice['label'])

#plt.subplots(4,5,figsize=(30,30))
#for i in range(1,21):
#    plt.subplot(4,5,i)
#    plt.title(voice.columns[i-1])
#    sns.kdeplot(voice.loc[voice['label'] == 0, voice.columns[i-1]], color= 'red', label='female')
#    sns.kdeplot(voice.loc[voice['label'] == 1, voice.columns[i-1]], color= 'blue', label='male')

selected_features = ['sd','IQR','Q25','meanfun']
voice_X = voice[selected_features]
#print(voice_X)
voice_y = voice.label
trainX, testX, trainy, testy = train_test_split(voice_X, voice_y,test_size=0.3,random_state=14)

suppvec = svm.SVC(kernel='linear', gamma=1)
suppvec.fit(trainX, trainy)
suppvec.score(trainX, trainy)
y_pred_svm= suppvec.predict(testX)
print('SVM stats:')
print('accuracy:', accuracy_score(testy,y_pred_svm))
print('f1-score:', f1_score(testy,y_pred_svm))
print("SVM Model Classification Report")
print(classification_report(testy, y_pred_svm))

logreg = LogisticRegression()
logreg.fit(trainX, trainy)
logreg.score(trainX, trainy)
y_pred_logr = logreg.predict(testX)
print('LR stats:')
print('accuracy:', accuracy_score(testy,y_pred_logr))
print('f1-score:', f1_score(testy,y_pred_logr))
print("LR Model Classification Report")
print(classification_report(testy, y_pred_logr))

forest = RandomForestClassifier(criterion='gini', n_estimators=12, random_state=1)
forest.fit(trainX, trainy)
forest.score(trainX, trainy)
y_pred_forest = logreg.predict(testX)
print('RF stats:')
print('accuracy:', accuracy_score(testy,y_pred_forest))
print('f1-score:', f1_score(testy,y_pred_forest))
print("RF Model Classification Report")
print(classification_report(testy, y_pred_forest))

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainX, trainy)
knn.score(trainX, trainy)
y_pred_knn = knn.predict(testX)
print('KNN stats:')
print('accuracy:', accuracy_score(testy,y_pred_knn))
print('f1-score:', f1_score(testy,y_pred_knn))
print("KNN Model Classification Report")
print(classification_report(testy, y_pred_knn))

vuic = pd.read_csv('/testvoice.csv')
vuic_X = vuic[selected_features]

y_svm = suppvec.predict(vuic_X)
y_logreg = logreg.predict(vuic_X)
y_forest = forest.predict(vuic_X)
y_knn = knn.predict(vuic_X)
y_forest
result = label_encode.inverse_transform(y_forest)
#print(vuic['label'])
print(result)
