import os
import time
from collections import defaultdict
import random
import pickle
import pandas as pd
import numpy as np
import random
import json
import re

# For machine learning tools and evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

start_time = time.time()

# Choose the BERT model that we want to use (make sure to keep the cased/uncased consistent)
model_name = 'distilbert-base-cased'  

# Choose the GPU we want to process this script
device_name = 'cuda'       

# This is the maximum number of tokens in any document sent to BERT
max_length = 512      

scotus_dir = '/share/luxlab/roz/scotus'

annotation_path = os.path.join(scotus_dir, 'monologic_annotations.json')
output_path = os.path.join(scotus_dir, 'output')
model_output_path = os.path.join(output_path, 'models')


print("BEGIN PROCESSING SENTENCES")

# helpful functions
def clean_regex(df, column):
  
  df["text"] = df["text"].str.replace('\n', ' ')
  df["text"] = df["text"].replace('\s+', ' ', regex = True)
  df["text"] = df["text"].replace(r'\[','', regex=True) 
  df["text"] = df["text"].replace(r'\]','', regex=True)
  df["text"] = df["text"].replace(r'\- ','', regex=True)
  df["text"] = df["text"].replace(r'\xad','', regex=True)
  df["text"] = df["text"].replace(r'\'','', regex=True)
  df["text"] = df["text"].replace(r'\x97',',', regex=True)

  return df["text"]

# Keep only sentences above certain threshold of alphanumeric characters
def percent_text(text):
    char_dict = dict()
    char_dict["alpha_count"] = 0
    char_dict["total_count"] = 0

    for char in text:
        char_dict["total_count"] += 1
        if char.isalpha():
            char_dict["alpha_count"] += 1
    
    percent_letter = float(char_dict["alpha_count"]) / float(char_dict["total_count"]) * 100

    return percent_letter

def header_eraser(text):
    spaces = re.search(r'[ \t]{2,}', text)
    opinion = re.search(r'Opinion of', text)
    if spaces and opinion:
        # delete text between first space and opinion of + 20 char 
        result = re.sub('[ \t]{2,}.*?Opinion of[\s\S]{15}', '', text)
    else:
        result = text
    return result


print("LOADING ANNOTATIONS")
annotations = []
for line in open(annotation_path, 'r'):
    annotations.append(json.loads(line))
df = pd.DataFrame(annotations)
df["label_num"] = df["answer"].map({'accept': 1, 'reject': 0, 'ignore': 0})
df = df[df["answer"]!= "ignore"]

print("CLEANING SENTENCES")
df["text"] = clean_regex(df, "text")

# Get sentences with more letters
df["percent_letter"] = df["text"].apply(percent_text)
df = df[df["percent_letter"] > 50]

# Remove header
df["text"] = df["text"].apply(header_eraser)


print("SPLIT INTO TRAINING, TEST TEXTS")
train_texts, test_texts, train_labels, test_labels = train_test_split(df["text"].to_list(), df["label_num"].to_list(), test_size=.3)

print("LOAD TOKENIZER")
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# PASS TO TOKENIZER, ADD PADDING AND TRUNCATE
train_encodings = tokenizer(train_texts,  truncation=True, padding=True)
test_encodings = tokenizer(test_texts,  truncation=True, padding=True)


# MAKE DATASET OBJECTS

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SCDataset(train_encodings, train_labels)
test_dataset = SCDataset(test_encodings, test_labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    learning_rate=5e-5,              # initial learning rate for Adam optimizer
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy='steps',
)

model = DistilBertForSequenceClassification.from_pretrained(model_name)#.to(device_name)

# Custom evaluation function 
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics      # custom evaluation function
)

print("BEGIN FINE-TUNING")
trainer.train()

print("RUN EVALUATION")
trainer.evaluate()

print("SAVE MODEL")
trainer.save_model(os.path.join(model_output_path))


# More evaluation
predicted_labels = trainer.predict(test_dataset)
actual_predicted_labels = predicted_labels.predictions.argmax(-1)

print("SAVING CLASSIFICATION REPORT")
class_report = classification_report(predicted_labels.label_ids.flatten(), actual_predicted_labels.flatten(), output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv(os.path.join(output_path, 'mono_classification_report.csv'))

print("SAVING ERRORS")
val_results = list(zip(predicted_labels.label_ids, actual_predicted_labels, test_texts))
val_results = pd.DataFrame(val_results, columns=['Predicted', 'Actual','Text'])
errors = val_results[val_results["Predicted"]!=val_results["Actual"]]
errors.to_csv(os.path.join(output_path, 'mono_errors.csv'))


# total end time
end_time = time.time()
print(f"FINISHED PROCESSING IN {end_time - start_time} SECONDS.")
