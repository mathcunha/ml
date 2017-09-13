import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split, cross_val_predict, cross_val_score
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

grid_search = GridSearchCV(log_clf, param_grid, cv=5, scoring='f1')
grid_search.fit(train_prepared, train_label)
log_clf = grid_search.best_estimator_
log_clf
f1_score(log_clf.predict(train_prepared), train_label)
f1_score(log_clf.predict(test_prepared), test_label)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(train_prepared)
X_Test_poly = poly_features.fit_transform(test_prepared)
log_clf = LogisticRegression(random_state=42)
log_clf.fit(X_poly, train_label)
f1_score(log_clf.predict(X_poly), train_label)
f1_score(log_clf.predict(X_Test_poly), test_label)


poly_features = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features.fit_transform(train_prepared)
X_Test_poly = poly_features.fit_transform(test_prepared)
log_clf = LogisticRegression(random_state=42)
log_clf.fit(X_poly, train_label)
f1_score(log_clf.predict(X_poly), train_label)
f1_score(log_clf.predict(X_Test_poly), test_label)

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_poly, train_label)
log_clf = grid_search.best_estimator_
log_clf
f1_score(log_clf.predict(X_poly), train_label)
f1_score(log_clf.predict(X_Test_poly), test_label)

test = pd.read_csv("test.csv", index_col = "PassengerId")

best = LogisticRegression(C=0.2, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=40, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=42, solver='newton-cg', tol=0.0001,
          verbose=0, warm_start=False)
best.fit(poly_features.fit_transform(full_pipeline.fit_transform(train)), train["Survived"])

test["Survived"] = grid_search.best_estimator_.predict(poly_features.fit_transform(full_pipeline.fit_transform(test)))
test['Survived'].to_csv("result.csv")