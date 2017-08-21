import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(42)

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
        ("cat_pipeline_emb", cat_pipeline_emb),
        ("cat_pipeline", cat_pipeline),
    ])


#train = pd.read_csv("train.csv")
train = pd.read_csv("train_complete.csv", index_col=0)
labels = train["Survived"]


train.drop("PassengerId", axis = 1, inplace = True)
train.drop("Cabin", axis = 1, inplace = True)
train.drop("Survived", axis = 1, inplace = True)
train.drop("Ticket", axis = 1, inplace = True)
train.drop("Name", axis = 1, inplace = True)

train_prepared = full_pipeline.fit_transform(train)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(train_prepared, labels)

test = pd.read_csv("test.csv")
test.drop("Cabin", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
passengers = test["PassengerId"]
test.drop("PassengerId", axis = 1, inplace = True)
test_prepared = full_pipeline.fit_transform(test)
out = sgd_clf.predict(test_prepared)

for idx, val in enumerate(passengers):
    print(val, out[idx], sep=",")


train = pd.read_csv("train_complete.csv", index_col=0)
passengers = train["PassengerId"]

out = sgd_clf.predict(train_prepared)

for idx, val in enumerate(passengers):
    print(val, out[idx], sep=",")