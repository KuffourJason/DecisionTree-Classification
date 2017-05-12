from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris

import graphviz

#loads the dataset
data = load_iris()

#Splits dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)

#Trains a decision tree with pure  leaves
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

print( "The training set accuracy, with pure leaves, is " + str(tree.score(x_train,y_train)) )
print( "The training set accuracy, with pure leaves, is " + str(tree.score(x_test,y_test)) )

treeD = DecisionTreeClassifier(max_depth=3, random_state=4)
treeD.fit(x_train,y_train)

print( "The training set accuracy, with non-pure leaves, is " + str(treeD.score(x_train,y_train)) )
print( "The training set accuracy, with non-pure leaves, is " + str(treeD.score(x_test,y_test)) )

export_graphviz(tree, out_file="tree.dot", class_names=['setosa', 'versicolor', 'virginica'],
 feature_names=data.feature_names, impurity=False, filled=True)

with open("tree.dot") as f:
 dot_graph = f.read()

graphviz.Source(dot_graph)

export_graphviz(treeD, out_file="treeD.dot", class_names=['setosa', 'versicolor', 'virginica'],
 feature_names=data.feature_names, impurity=False, filled=True)

with open("treeD.dot") as f:
 dot_graph1 = f.read()
 
graphviz.Source(dot_graph1)