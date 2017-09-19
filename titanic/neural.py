import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

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
train = pd.read_csv("train.csv", index_col = "PassengerId")

train_set, test_set, train_label, test_label = train_test_split(train, train["Survived"], random_state=42, train_size=0.8)

train_prepared = full_pipeline.fit_transform(train_set)
test_prepared = full_pipeline.fit_transform(test_set)

from sklearn.linear_model import LogisticRegression

param_grid ={   
                'max_iter': range(5, 100, 10),
                'C' : [0.2, 0.4, 0.6, 0.8, 1.0],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag']
            }


log_clf = LogisticRegression(random_state=42)
log_clf.fit(train_prepared, train_label)
f1_score(log_clf.predict(train_prepared), train_label)
f1_score(log_clf.predict(test_prepared), test_label)

neu_clf = MLPClassifier(random_state=42)
neu_clf.fit(train_prepared, train_label)
f1_score(neu_clf.predict(train_prepared), train_label)
f1_score(neu_clf.predict(test_prepared), test_label)

parameters={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [x for x in itertools.product((10,20,30,40,50,100),repeat=3)],
'activation': ["logistic", "relu", "Tanh"]
}

parameters={
'alpha': 10.0 ** -np.arange(1, 7),
'max_iter' : range(100, 5000, 100)
}

grid_search = GridSearchCV(MLPClassifier(random_state=42), parameters, cv=5, scoring='f1')
grid_search.fit(train_prepared, train_label)
neu_clf = grid_search.best_estimator_
neu_clf
f1_score(neu_clf.predict(train_prepared), train_label)
f1_score(neu_clf.predict(test_prepared), test_label)

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

neu_clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
       
neu_clf.fit(full_pipeline.fit_transform(train), train["Survived"])

test = pd.read_csv("test.csv", index_col = "PassengerId")
test["Survived"] = neu_clf.predict(full_pipeline.fit_transform(test))
test['Survived'].to_csv("result_2.csv")

parameters={
'alpha': [0.00008, 0.0001, 0.00012],
'max_iter' : range(100, 1000, 100),
'hidden_layer_sizes': [x for x in itertools.product((10,20,30,40,50,100),repeat=3)]
}

grid_search = GridSearchCV(MLPClassifier(random_state=42), parameters, cv=5, scoring='f1')
grid_search.fit(train_prepared, train_label)
neu_clf = grid_search.best_estimator_
neu_clf
f1_score(neu_clf.predict(train_prepared), train_label)
f1_score(neu_clf.predict(test_prepared), test_label)