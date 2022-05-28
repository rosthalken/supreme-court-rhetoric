import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.inspection import permutation_importance
import pickle
import numpy as mp
from sklearn.linear_model import LogisticRegression


'''
filename = 'lr_rhetoric.sav'
lr = pickle.load(open(filename, 'rb'))
filename = 'vectorizer.sav'
vectorizer = pickle.load(open(filename, 'rb'))
'''
annotations_df = pd.read_csv(r'annotations_rhetoric.csv')


lr = LogisticRegression()
y = annotations_df["label_num"]

vectorizer = TfidfVectorizer(
    encoding='utf-8',
    min_df=2, 
    max_df=0.8, 
    binary=False,
    norm='l2',
    use_idf=True 
)

X = vectorizer.fit_transform(annotations_df.text)
#X = pd.DataFrame(tfidf_m.toarray(), index=y, columns=vectorizer.get_feature_names())

lr.fit(X, y)

# Coefficient importance
coef_dict = {}
for coef, feat in zip(lr.coef_[0],vectorizer.get_feature_names()):
    coef_dict[feat] = coef

#keys = coef_dict.keys()
#values = coef_dict.values()

small_dict = {k:v for (k,v) in coef_dict.items() if v > 1.5}
keys = small_dict.keys()
values = small_dict.values()


dict_df = pd.DataFrame.from_dict(small_dict, orient='index')
dict_df['feature'] = dict_df.index


# Plot
import seaborn as sns
ax = sns.barplot(x=0, y="feature", data=dict_df)



# Permutation and Feature Importance
# Also too many features? There currently are 7430
r = permutation_importance(lr, X, y, n_repeats=10, random_state=0)
perm_sorted_idx = r.importances_mean.argsort()

import pickle
filename = 'permuations.sav'
pickle.dump(r, open(filename, 'wb'))

# import w/ pickle
r = pickle.load(open(filename, 'rb'))

pickle_file = open("permuations.sav","rb")
r = pickle.load(pickle_file)



