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
device = 'cuda'       

# This is the maximum number of tokens in any document sent to BERT
max_length = 512      

scotus_dir = '/share/luxlab/roz/scotus'

dataset_path = os.path.join(scotus_dir, 'scotus_final_sents.json')
output_path = os.path.join(scotus_dir, 'output')
model_path = os.path.join(output_path, 'models')

raw_output_path = os.path.join(output_path, 'raw_output')
combined_output_path = os.path.join(output_path, 'combined_output')
combined_csv_path = os.path.join(output_path,'combined_output.csv')

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

def get_clean_data(dataset_path):
  opinion_js = json.load(open(dataset_path))
  df = pd.DataFrame.from_dict(opinion_js)

  # Clean sentences
  df["text"] = clean_regex(df, "text")

  # Get sentences with more letters
  df["percent_letter"] = df["text"].apply(percent_text)
  df = df[df["percent_letter"] > 50]

  # Remove header
  df["text"] = df["text"].apply(header_eraser)

  # Normalize dissenting category
  df.loc[(df.category == 'second_dissenting'),'category']='dissenting'

  # Prepare sentences for tokenization
  all_sentences = df["text"].to_list()

  return df, all_sentences


print("CLEANING SENTENCE DATA")
df, worklist = get_clean_data(dataset_path)
batchsize = 32
predictions = []

print("LOAD MODELS")
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

print("RUN PREDICTIONS")
for i in range(0, len(worklist), batchsize):
  batch = worklist[i:i+batchsize] # extract batch from worklist
  test_encodings = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device) # tokenize the posts
  output = model(**test_encodings) # make predictions with model on our test_encodings for this batch
  batch_predictions = torch.softmax(output.logits, dim=1).tolist() # get the predictions result
  predictions.append(batch_predictions)
  if i % 1000 == 0:
      print(f"{round((i/len(worklist))*100, 2)}% of the way finished")

print("SAVE RAW PREDICTIONS")
pickle.dump(predictions, open(raw_output_path, "wb"))


print("ADD PREDICTIONS TO MAIN DF")
flat_list = [item for sublist in predictions for item in sublist]
df["predictions"] = flat_list
df[['prob_0','prob_1']] = pd.DataFrame(df["predictions"].tolist(), index=df.index)
df['monologic_prediction'] = np.where(df['prob_1'] > .50, 1, 0) # this is the column we're interested in, since this is a binary label


print("CLEAN UP BAD DATA")
# Rename categories
df.loc[(df.category == 'majority'),'category']='Majority'
df.loc[(df.category == 'dissenting'),'category']='Dissenting'
df.loc[(df.category == 'concurring'),'category']='Concurring'
df.loc[(df.category == 'per_curiam'),'category']='Per Curiam'

# Remove bad names
wrong_names = ["Justice And", "Justice O2122", "Justice Or", "Justice Connor", "Justice Holmes", "Justice Fuller", "Justice Waite", "Justice Woods", "Justice McReynolds", "Justice Stone"]
df = df[~df['author'].isin(wrong_names)]

# Remove Justice White errors
df[df["author"] == "Justice White"].year.max()
index_names = df[(df['author'] == "Justice White") & (df['year'] == 2010)].index
df.drop(index_names, inplace = True)
index_names = df[(df['author'] == "Justice White") & (df['year'] == 2005)].index
df.drop(index_names, inplace = True)

# Add Chief Justice
conditions = [
    (df['year'] <= 1953),
    (df['year'] > 1953) & (df['year'] <= 1969),
    (df['year'] > 1969) & (df['year'] <= 1986),
    (df['year'] > 1986) & (df['year'] <= 2004),
    (df['year'] > 2004)
    ]

values = ["Vinson", "Warren", "Burger", "Rehnquist", "Roberts"]
df['chief_justice'] = np.select(conditions, values)

print("SAVE FULL DF WITH PREDICTIONS")
df.to_pickle(combined_output_path, protocol = 4)
df.to_csv(combined_csv_path)

# total end time
end_time = time.time()
print(f"ALL PREDICTIONS FINISHED PROCESSING IN {end_time - start_time} SECONDS.")

