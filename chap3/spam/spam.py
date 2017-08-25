import pandas as pd
import numpy as np
import codecs
import array

from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.linear_model import SGDClassifier

from os import listdir
from nltk.tokenize import word_tokenize

columns = ['name','type', 'spam','path', 'tokens','array']

dictionary = list()

def loadTokens(filePath):
    with codecs.open(filePath, 'r',encoding='utf-8', errors='ignore') as infile:
        tokens =  word_tokenize(infile.read())
        set_tokens = set(tokens)#unique items
        for item in list(set_tokens):#cleaning tokens
            if item.isalpha() == False: 
                set_tokens.remove(item)
            elif len(item) == 1:
                set_tokens.remove(item)
        tokens = list(set_tokens)
        dictionary.extend(tokens)
        return tokens

def loadEmailData(directory, spam=1):
    email = pd.DataFrame(columns=columns)
    for f in listdir('dataset/'+directory):
        filePath = 'dataset/'+directory+'/'+f
        email = pd.concat([pd.DataFrame([[f, directory, spam, filePath , loadTokens(filePath) , None]], columns=columns), email])
    return email

data = pd.concat([loadEmailData('easy_ham', False), loadEmailData('easy_ham_2', False), loadEmailData('hard_ham', False), loadEmailData('spam', True), loadEmailData('spam_2', True)])
#data = pd.concat([loadEmailData('xingar', 1), loadEmailData('nomes', 0)])
data['id'] = range(1, len(data) + 1)
data.set_index(['id'], inplace=True)
data.reset_index()

dictionary = list(set(dictionary))

def createTokenArray(tokens):
    item = list()
    for word in dictionary:
        item.append(tokens.count(word))
    return item

data['tok_array'] = data['tokens'].apply(createTokenArray)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["type"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

def type_proportions(data):
    return data["type"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall": type_proportions(data),
    "Stratified": type_proportions(strat_train_set),
    "Stratified-test": type_proportions(strat_test_set),
}).sort_index()
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props["Strat. test %error"] = 100 * compare_props["Stratified-test"] / compare_props["Overall"] - 100

compare_props


mapper = DataFrameMapper([
    ("tok_array", None),
], df_out = False)

pipeline = Pipeline([
    ("mapper", mapper)
])

labels = strat_train_set["spam"]
train_prepared = pipeline.fit_transform(strat_train_set)

sgd_clf = SGDClassifier(random_state=42)
y_scores = cross_val_predict(sgd_clf, train_prepared, labels, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(labels, y_scores)