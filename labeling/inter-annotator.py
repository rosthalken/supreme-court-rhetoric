import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import pandas as pd

file_name1 = "Annotations/monologic_annotations.json"
file_name2 = "Annotations/annotations_aditi.json"
file_name3 = "Annotations/annotations_hana.json"

# GOLD ANNOTATIONS
gold_annotations = []
for line in open(file_name1, 'r'):
    gold_annotations.append(json.loads(line))
gold_annotations_df = pd.DataFrame(gold_annotations)
print(gold_annotations_df.shape)
gold_annotations_df.head(10)

gold_annotations_df["label_num"] = gold_annotations_df["answer"].map({'accept': 1, 'reject': 0, 'ignore': 0})
print(gold_annotations_df.label_num.value_counts())

subset = gold_annotations_df.head(500)

# SECOND ANNOTATIONS
second_annotations = []
for line in open(file_name2, 'r'):
    second_annotations.append(json.loads(line))
second_annotations_df = pd.DataFrame(second_annotations)
print(second_annotations_df.shape)
second_annotations_df.head(10)

second_annotations_df["label_num2"] = second_annotations_df["answer"].map({'accept': 1, 'reject': 0, 'ignore': 0})
print(second_annotations_df.label_num2.value_counts())

second_annotations_df = second_annotations_df.head(500)

# THIRD ANNOTATIONS
third_annotations = []
for line in open(file_name3, 'r'):
    third_annotations.append(json.loads(line))
third_annotations_df = pd.DataFrame(third_annotations)
print(third_annotations_df.shape)
third_annotations_df.head(10)

third_annotations_df["label_num3"] = third_annotations_df["answer"].map({'accept': 1, 'reject': 0, 'ignore': 0})
print(third_annotations_df.label_num3.value_counts())



# Annotation reshapint
annotations_1 = gold_annotations_df[["label_num"]]
annotations_1["annotator"] = 1
annotations_1["sentence"] = annotations_1.index
annotations_1 = annotations_1.head(500)
annotations_1 = annotations_1.rename(columns={"label_num":"label"})

annotations_2 = second_annotations_df[["label_num2"]]
annotations_2["annotator"] = 2
annotations_2["sentence"] = annotations_2.index
annotations_2 = annotations_2.rename(columns={"label_num2":"label"})

annotations_3 = third_annotations_df[["label_num3"]]
annotations_3["annotator"] = 3
annotations_3["sentence"] = annotations_3.index
annotations_3 = annotations_3.rename(columns={"label_num3":"label"})


pd_list = [annotations_1, annotations_2, annotations_3]
labels_df = pd.concat(pd_list)


# Krippendorff's alpha
import simpledorff

simpledorff.calculate_krippendorffs_alpha_for_df(labels_df,experiment_col='sentence',
                                                 annotator_col='annotator',
                                                 class_col='label')






# OR Cohen's kappa for each

# MERGE
large_df = subset.merge(second_annotations_df, left_index=True, right_index=True)
all_df = large_df.merge(third_annotations_df, left_index=True, right_index=True)
double_check = all_df.sample(20)


# Looks good!

gold_answers = all_df["label_num"].tail(200)
second_annotator = all_df["label_num2"].tail(200)
third_annotator = all_df["label_num3"].tail(200)



from collections import Counter
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(gold_answers, third_annotator)

cohen_kappa_score(gold_answers, second_annotator)

cohen_kappa_score(second_annotator, third_annotator)
