from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
# The dataset used here is iris.csv stored in the anaconda installed folder
dataset=datasets.load_iris()
model=MultinomialNB()
model.fit(dataset.data,dataset.target)
expected=dataset.target
predected=model.predict(dataset.data)
print(metrics.accuracy_score(expected,predected))
print(metrics.confusion_matrix(expected,predected))    
