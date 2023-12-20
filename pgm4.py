
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree 
from graphviz import Digraph
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
# The dataset used here is iris.csv stored in the anaconda installed folder
iris=datasets.load_iris()
X=iris.data
Y=iris.target
print(iris.target_names,iris.feature_names)
clf=DecisionTreeClassifier(criterion="entropy")
model=clf.fit(X,Y)
dot_data=tree.export_graphviz(clf,out_file=None,class_names=iris.target_names)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.jpg")
img=pltimg.imread('tree.jpg')
imgplot = plt.imshow(img)
plt.show()
