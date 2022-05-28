import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk

input_dir = "/Users/rosamondthalken/Documents/Projects/SupremeCourt/SupremeCourtPython/DataX/scdb_court_merged.csv"
sc_opinion_df = pd.read_csv(input_dir)

metadata_df = sc_opinion_df[["Unnamed: 0", "usCite", "caseName", "precedentAlteration", "issue", "author_name", "category", "per_curiam", "year_filed", "scdb_decision_direction", "scdb_votes_majority", "scdb_votes_minority", "text"]].copy()



# SPLIT VOTES
# ADD 1 OR 0 BASED ON WHETHER THE VOTE IS SPLIT
metadata_df["vote_split"] = np.where(metadata_df["scdb_votes_majority"] <= 5, 1, 0)





# PRECEDENT ALTERATION
metadata_df["precedentAlteration"]




# UNANIMOUS AND PER CURIAM VS SPLIT
def unanimous_vs_split(row):
    if row["category"] == "per_curiam":
        return 1
    elif row["scdb_votes_minority"] == 0:
        return 1
    elif row["vote_split"] == 1:
        return 0
    else:
        return "NA"
metadata_df["unan_v_split"] = metadata_df.apply(lambda row: unanimous_vs_split(row), axis=1)
unanimous_and_split_df = metadata_df[metadata_df["unan_v_split"] != "NA"]





# UNANIMOUS VS PER CURIAM
def percuriam_vs_unan(row):
    if row["category"] == "per_curiam":
        return 1
    elif (row["category"] != "per_curiam") and (row["scdb_votes_minority"] == 0):
        return 0
    else:
        return "NA"

metadata_df["unan_v_percuriam"] = metadata_df.apply(lambda row: percuriam_vs_unan(row), axis=1)
unanimous_and_perc_df = metadata_df[metadata_df["unan_v_percuriam"] != "NA"]






##############################################################
# CLASSIFICATION
# Set up the vectorizer

def tokenize_words(text, numbers = 'remove'):
    if numbers == 'remove':
        text = re.sub('[0-9]+', ' ', text)
    elif numbers == 'replace':
        text = re.sub('[0-9]+', 'NUM', text)
    text = nltk.word_tokenize(text)
    return text

vectorizer = TfidfVectorizer(
    encoding='utf-8',
    min_df=.1,
    max_df=.8, 
    tokenizer=tokenize_words,
    binary=False,
    norm='l2',
    use_idf=True 
)

# MOST Xs
X = vectorizer.fit_transform(metadata_df["text"])
y = metadata_df["vote_split"]

# PRECEDENT ALTERATION
#y = metadata_df["precedentAlteration"]


# UNANIMOUS/PER CURIAM VS SPLIT
#X = vectorizer.fit_transform(unanimous_and_split_df["text"])
#y = unanimous_and_split_df["unan_v_split"].astype(int)


# UNANIMOUS VS PER CURIAM
#X = vectorizer.fit_transform(unanimous_and_perc_df["text"])
#y = unanimous_and_perc_df["unan_v_percuriam"].astype(int)



lr = LogisticRegression()
scores = cross_validate(lr, X, y, cv=10, scoring=['accuracy', 'f1', 'f1_macro', 'f1_micro'])
print(np.mean(scores.get("test_f1")))
print(np.mean(scores.get("test_accuracy")))
lr.fit(X, y)


# Coefficient importance
coef_dict = {}
for coef, feat in zip(lr.coef_[0], vectorizer.get_feature_names()):
    coef_dict[feat] = coef

top_feat_dict = {k:v for (k,v) in coef_dict.items() if v > 1}
dict_df = pd.DataFrame.from_dict(top_feat_dict, orient='index')
dict_df['feature'] = dict_df.index
dict_df.sort_values(by=[0], ascending=False).head(20)


# Plot
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.barplot(x=0, y="feature", data=dict_df)






# Confusion Matrix
y_best = lr.predict(X)
confusion_m = metrics.confusion_matrix(y, y_best)
sns.set(font_scale=2)
fig, ax = plt.subplots(figsize=(10,6))
group_names = ["True Neg","False Pos","False Neg","True Pos"]
# Below percentages show percentage of total (not percentage of pos or neg)
group_percentages = ["{0:.2%}".format(value) for value in
                     confusion_m.flatten()/np.sum(confusion_m)]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(ax = ax, data = confusion_m, annot=labels, fmt="", cmap="Blues");




# Baseline scores --> Fix these
baseline_accuracy = 1-sum(y)/len(y)
baseline_precision = baseline_accuracy
baseline_recall = 1.0
baseline_f1 = 2*baseline_precision*baseline_recall/(baseline_precision+baseline_recall)
print("Baseline F1:", round(baseline_f1, 3))
print("Baseline accuracy:", round(baseline_accuracy, 3))

