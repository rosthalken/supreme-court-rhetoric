'''
Turn the corpus into a data frame split by one sentence per row
Remove opinions with less than 100 words
'''
from asyncio import open_unix_connection
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# CLEAN DATA
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalnum()]
    return words

def percent_text(text):
    char_dict = dict()
    char_dict["alpha_count"] = 0
    char_dict["total_count"] = 0

    for char in text:
        if not char.isspace():
            char_dict["total_count"] += 1
        if char.isalpha():
            char_dict["alpha_count"] += 1
    
    percent_letter = float(char_dict["alpha_count"]) / float(char_dict["total_count"]) * 100

    return percent_letter


# Remove middle of cases
def remove_middles(df):

    if len(word_tokenize(df["text"])) < 2000:
        new_text = df["text"]
    else:
        part_1 = []
        part_2 = []
        first_words = 0
        last_words = 0

        sentences = sent_tokenize(df["text"])
        for sentence in sentences:
            text_percent = percent_text(sentence)
            if text_percent < 85:
                continue
            else:
                num_words = len(word_tokenize(sentence))
                part_1.append(sentence)
                first_words += num_words
                if first_words > 1000:
                    break

        for sentence in reversed(sentences):
            text_percent = percent_text(sentence)
            if text_percent < 85:
                continue
            else:
                num_words = len(word_tokenize(sentence))
                part_2.append(sentence)
                last_words += num_words
                if last_words > 1000:
                    break 
        
        part_2.reverse()
        part_1 = ' '.join(part_1)
        part_2 = ' '.join(part_2)

        new_text = part_1 + part_2

    return new_text


# Split each case into sentences
def split_df_by_sentences(df, output_path):
    sentences = []
    for row in df.itertuples():
        for sentence in sent_tokenize(row.text):
            if len(sentence)>10:
                sentences.append((row.opinion_num, row.category, row.author_name, row.case_name, row.year_filed, row.token_count, sentence))

    sentences_df = pd.DataFrame(sentences, columns=['opinion_num','category', 'author', 'case', 'year', 'token_count','text'])

    # Add sentence index
    sentences_df["sent_index"] = sentences_df.groupby('opinion_num').cumcount()

    # Total # of sentences in each case
    counted_sents = Counter(sentences_df.opinion_num)
    counted_sents_df = pd.DataFrame.from_dict(counted_sents, orient="index")
    counted_sents_df = counted_sents_df.rename(columns={0:"length"})

    # Change column name to prep for merge of two
    counted_sents_df['opinion_num'] = counted_sents_df.index

    # Merge
    sentences_df = pd.merge(sentences_df, counted_sents_df, on='opinion_num')

    # Find sentence location (normalized by length)
    sentences_df["sent_location"] = sentences_df["sent_index"]/sentences_df["length"]

    sentences_df.to_json(output_path)

    return sentences_df



# LOAD DATA

# - Main collection of opinions
full_df = pd.read_csv(r'DataX/all_opinions.csv')
modern_df = full_df[full_df["year_filed"] > 1945]

# - Manually collected, missing per curiam opinions
per_cur_df = pd.read_csv(r'DataX/per_curiam.csv').iloc[:, 1:-2]
per_cur_df["per_curiam"] = "True"
per_cur_df["scdb_votes_minority"] = "NaN"

# - and recent opinions, manually collected
new_df = pd.read_csv(r'DataX/recent_opinions_3.csv').iloc[:, 1:]

new_df["tokens"] = new_df.apply(lambda row: clean_text(row['text']), axis=1)
new_df["token_count"] = new_df.apply(lambda row: len(row["tokens"]), axis = 1)
new_df["group"] = "recent"

modern_df["tokens"] = modern_df.apply(lambda row: clean_text(row['text']), axis=1)
modern_df["token_count"] = modern_df.apply(lambda row: len(row["tokens"]), axis = 1)
modern_df["group"] = "modern"

per_cur_df["tokens"] = per_cur_df.apply(lambda row: clean_text(row['text']), axis=1)
per_cur_df["token_count"] = per_cur_df.apply(lambda row: len(row["tokens"]), axis = 1)
per_cur_df["group"] = "per_cur"

# Merge the diff dataframes 
frames = [new_df, per_cur_df, modern_df]
full_df = pd.concat(frames)

full_df = full_df.drop_duplicates(subset=['category','author_name', 'case_name', 'year_filed'])
full_df[full_df["token_count"]<500]["token_count"].hist()
full_df["token_count"].describe()

# Remove extremely short cases
outlier_removed_df = full_df[full_df["token_count"] < 100]
outlier_removed_df["token_count"].hist(bins = 100)

full_df = full_df[full_df["token_count"] > 100]
full_df['opinion_num'] = np.arange(len(full_df))

# full_df["text"] = full_df.apply(remove_middles, axis = 1)   
sentences_df = split_df_by_sentences(full_df, "DataX/scotus_final_sents.json")
