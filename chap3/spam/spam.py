import pandas as pd
import numpy as np
import codecs
import array
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.linear_model import SGDClassifier

from os import listdir
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

columns = ['name','type', 'spam','path', 'tokens']
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

dictionary = list()

def save_fig(fig_id, tight_layout=True):
    path =  fig_id + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def loadTokens(filePath):
    with codecs.open(filePath, 'r',encoding='utf-8', errors='ignore') as infile:
        tokens =  tknzr.tokenize(infile.read())
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
        email = pd.concat([pd.DataFrame([[f, directory, spam, filePath , loadTokens(filePath)]], columns=columns), email])
    return email

print('loading email files')
data = pd.concat([loadEmailData('easy_ham', 0), loadEmailData('easy_ham_2', 0), loadEmailData('hard_ham', 0), loadEmailData('spam', 1), loadEmailData('spam_2', 0)])
#data = pd.concat([loadEmailData('xingar', 1), loadEmailData('nomes', 0)])
data['id'] = range(0, len(data), 1)
data.set_index(['id'], inplace=True)
data.reset_index()

dictionary = list(set(dictionary))

#https://stackoverflow.com/question74310s/39745807/typeerror-expected-sequence-or-array-like-got-estimator

random.seed(42)
random.shuffle(dictionary)

bkp_dictionary = list(dictionary)
dictionary = bkp_dictionary[40000:]#just for my laptop

def createTokenArray(tokens):
    item = list()
    for word in dictionary:
        item.append(tokens.count(word))
    #item = np.array(item)
    return item

#for word in dictionary:
#    def countTokens(tokens):
#        return tokens.count(word)
#    data[word] = data['tokens'].apply(countTokens)


#data.drop("tokens", axis = 1, inplace = True)
print('counting tokens by file')
data['tok_array'] = data['tokens'].apply(createTokenArray)

print('saving data')
data.to_csv('data.csv', sep=',', encoding='utf-8')

data.drop("tokens", axis = 1, inplace = True)

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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(dictionary)),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("cat_pipeline", cat_pipeline),
    ])

labels = strat_train_set["spam"]
#train_prepared = full_pipeline.fit_transform(strat_train_set)
train_prepared = np.array(strat_train_set['tok_array'].tolist())

sgd_clf = SGDClassifier(random_state=42)

print('training model')
y_scores = cross_val_predict(sgd_clf, train_prepared, labels, cv=3, method="decision_function")

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper right", fontsize=16)
    plt.ylim([0, 1])

precisions, recalls, thresholds = precision_recall_curve(labels, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

save_fig('modelo')

exit()