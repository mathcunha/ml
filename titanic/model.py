import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import f1_score

np.random.seed(42)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

num_attribs = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Sex", "Embarked"]

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(["Sex"])),
        ('label_binarizer', LabelBinarizer()),
    ])

cat_pipeline_emb = Pipeline([
        ('selector', DataFrameSelector(["Embarked"])),
        ('label_binarizer', LabelBinarizer()),
    ])
	
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        #("cat_pipeline_emb", cat_pipeline_emb),
        ("cat_pipeline", cat_pipeline),
    ])


def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target, y_pred), sum(target == y_pred) / float(len(y_pred))

#train = pd.read_csv("train.csv")
train = pd.read_csv("train.csv")
labels = train["Survived"]

train.drop(["Name","Ticket","Survived","Cabin","PassengerId"], axis = 1, inplace = True)

train_prepared = full_pipeline.fit_transform(train)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
scores = cross_val_score(sgd_clf, train_prepared, labels, scoring="f1", cv=5)
print("SGDClassifier")
display_scores(scores)

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(forest_clf, train_prepared, labels, scoring="f1", cv=5)
print("RandomForestClassifier")
display_scores(scores)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
scores = cross_val_score(log_reg, train_prepared, labels, scoring="f1", cv=5)
print("LogisticRegression")
display_scores(scores)

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(seed = 42)
scores = cross_val_score(xgb_clf, train_prepared, labels, scoring="f1", cv=5)
print("XGBClassifier")
display_scores(scores)

from sklearn.grid_search import GridSearchCV

# TODO: Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }  

parameters1 = { 'learning_rate' : [0.001, 0.01, 0.3, 0.35],
                'n_estimators' : [40, 117],
                'max_depth':[4,5,6],
                'min_child_weight':[4,5,6],
               'gamma':[0.4],
               'subsample':[i/10.0 for i in range(8,10)],
                'colsample_bytree':[i/10.0 for i in range(2,6)],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
             } 


parameters1 = { 'learning_rate' : [0.3],
                'n_estimators' : [117],
                'max_depth':[4,5,6],
                'min_child_weight':[4,5,6],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             } 

parameters1 = { 'learning_rate' : [0.3],
                'n_estimators' : [117],
                'max_depth':[4],
                'min_child_weight':[4],
               'gamma':[0.4],
               'subsample':[i/10.0 for i in range(8,10)],
                'colsample_bytree':[i/10.0 for i in range(2,6)],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             } 

parameters1 = { 'learning_rate' : [0.3],
                'n_estimators' : [117],
                'max_depth':[4],
                'min_child_weight':[4],
               'gamma':[0.4],
               'subsample':[0.9],
                'colsample_bytree':[0.5],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-2]
             } 

grid_search = GridSearchCV(xgb_clf, parameters1, cv=5,scoring='f1')
grid_search.fit(train_prepared, labels)
clf = grid_search.best_estimator_
clf

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, train_prepared, labels)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    

clf = xgb_clf
test = pd.read_csv("test.csv")
test.drop("Cabin", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
passengers = test["PassengerId"]
test.drop("PassengerId", axis = 1, inplace = True)
test_prepared = full_pipeline.fit_transform(test)
out = clf.predict(test_prepared)

for idx, val in enumerate(passengers):
    print(val, out[idx], sep=",")


train = pd.read_csv("train_complete.csv", index_col=0)
passengers = train["PassengerId"]

out = sgd_clf.predict(train_prepared)

from sklearn.metrics import accuracy_score
accuracy_score(labels, out)

for idx, val in enumerate(passengers):
    print(val, out[idx], sep=",")
