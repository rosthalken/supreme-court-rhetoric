# supreme-court-rhetoric
This repository includes code for the project "Judicial Self-Fashioning": A Computational Analysis of Rhetoric in Supreme Court Opinions." 

The main code for this project resides in the "DistilBERT" directory, and includes the pipeline from training (distilbert_train.ipynb), to predictions and sentence cleaning (distilbert_predict.ipynb), to analysis (distilbert_analysis.ipynb). 

The necessary data can be found in "data/annotations" and includes hand-labeled sentences for each of Ferguson's rhetorical types: the monologic voice, interrogative mode, declarative tone, and rhetoric of innevitability (see ["The Judicial Opinion as Literary Genre"](https://digitalcommons.law.yale.edu/yjlh/vol2/iss1/15/)). "Judicial Self Fashioning" only studies the monologic voice, but any of the rhetorical types should run through the pipeline, from start to finish. 

distilbert_predict.ipynb and distilbert_analysis.ipynb files rely on the full corpus, which is available upon request. 
