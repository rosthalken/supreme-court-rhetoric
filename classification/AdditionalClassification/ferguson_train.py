import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GroupKFold
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.inspection import permutation_importance

file_name = "Annotations/monologic_annotations.json"

annotations = []
for line in open(file_name, 'r'):
    annotations.append(json.loads(line))
annotations_df = pd.DataFrame(annotations)


annotations_df["label_num"] = annotations_df["answer"].map({'accept': 1, 'reject': 0, 'ignore': 0})
annotations_df.answer.value_counts()
annotations_df.categority.value_counts()

#annotations_df = annotations_df[annotations_df.categority == "majority"]

# Rhetoric probability is higher for monologic labeled text
grouped = annotations_df.groupby(["answer"])
grouped.prob_1.mean()



stop_words=frozenset(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'as', 'do', 'at', 'this', 'but', 'by', 'from'])

vectorizer = TfidfVectorizer(
    encoding='utf-8',
    min_df=2, 
    stop_words=stop_words,
    binary=False,
    norm='l2',
    use_idf=False 
)

vectorizer.fit(annotations_df.text)

# Classification
lr = LogisticRegression()
x_values = []
y_values = []
scores_l = []

sample_sizes = range(50, len(annotations_df.label_num), 10)

for i in sample_sizes:
    sample_df = annotations_df.sample(n=i)
    y_opinions = sample_df["label_num"]
    #tfidf_m = vectorizer.transform(sample_df.text)
    X_opinions = vectorizer.fit_transform(sample_df.text)
    #X_opinions = pd.DataFrame(tfidf_m.toarray(), index=y_opinions, columns=vectorizer.get_feature_names())
    # Cross validation
    scores = cross_validate(lr, X_opinions, y_opinions, cv=10, scoring=['accuracy', 'f1', 'f1_macro', 'f1_micro'])
    scores_l.append(scores)
    x = len(y_opinions)
    x_values.append(x)
    y = np.mean(scores.get("test_f1"))
    y_values.append(y)

plt.plot(x_values, y_values);

# Total documents
y_opinions = annotations_df["label_num"]
X_opinions = vectorizer.fit_transform(annotations_df.text)

# Baseline scores
baseline_accuracy = sum(y_opinions)/len(y_opinions)
baseline_precision = baseline_accuracy
baseline_recall = 1.0
baseline_f1 = 2*baseline_precision*baseline_recall/(baseline_precision+baseline_recall)
print("Baseline accuracy:", round(baseline_accuracy, 3))
print("Baseline F1:", round(baseline_f1, 3))

lr = LogisticRegression()
scores = cross_validate(lr, X_opinions, y_opinions, cv=10, scoring=['accuracy', 'f1', 'f1_macro', 'f1_micro'])
np.mean(scores.get("test_f1"))
np.mean(scores.get("test_accuracy"))

# Fit the model
lr.fit(X_opinions, y_opinions)

# Coefficient importance
coef_dict = {}
for coef, feat in zip(lr.coef_[0],vectorizer.get_feature_names()):
    coef_dict[feat] = coef
large_dict_df = pd.DataFrame.from_dict(coef_dict, orient='index')


print(coef_dict.get("we"))
print(coef_dict.get("i"))

#keys = coef_dict.keys()
#values = coef_dict.values()

small_dict = {k:v for (k,v) in coef_dict.items() if v > .8}
keys = small_dict.keys()
values = small_dict.values()


dict_df = pd.DataFrame.from_dict(small_dict, orient='index')
dict_df['feature'] = dict_df.index


# Plot
import seaborn as sns
ax = sns.barplot(x=0, y="feature", data=dict_df)

# Analyzing Output: Probability and Errors

#https://datascience.stackexchange.com/questions/22762/understanding-predict-proba-from-multioutputclassifier
probability_scores = lr.predict_proba(X_opinions)[:,1]

y_best = lr.predict(X_opinions)
# Need to add cross validation
def pull_errors(labels, gold_labels, corpus, n=3):
    labeled_opinions = pd.DataFrame(
        {
            'gold':gold_labels,
            'computed': labels,
            'text':corpus
        }
    )
    errors = labeled_opinions.loc[labeled_opinions.gold != labeled_opinions.computed]
    with pd.option_context('display.max_colwidth', None):
        display(errors.sample(n))

pull_errors(y_best, y_opinions, annotations_df.text, n=20)



# Confusion Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn import metrics

confusion_m = metrics.confusion_matrix(y_opinions, y_best)
sns.heatmap(confusion_m, annot=True)

# Permutation
r = permutation_importance(lr, X_opinions.toarray(), y_opinions, n_repeats=10, random_state=0)
perm_sorted_idx = r.importances_mean.argsort()

print('Permutation importance scores', r.importances)

perm_dict = {}
for perm, feat in zip(r.importances_mean,vectorizer.get_feature_names()):
    perm_dict[feat] = perm

# Save perm_dict
import pickle
pickle.dump(perm_dict, open("mon_perm_dict.p", "wb" ))


# Or load
perm_dict = pickle.load(open("mon_perm_dict.p", "rb" ))