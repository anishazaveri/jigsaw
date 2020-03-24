# Jigsaw Unintended Bias in Toxicity Classification
**Introduction**: This is code used for the Kaggle competition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). The approach described acheived a [top 6% ranking on Kaggle](https://www.kaggle.com/anneblythe1)

**Goal**: In the original Toxic Comments Classification Challenge models learnt to predict toxicity based on comments. However, they found that models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities.

**Approach**
This competition allowed kernel only submission to Kaggle i.e. your code had to run within 9 hours. You were allowed to upload pretrained models but predictions had to be done within the final kernel.

TLDR: [BERT-base](https://github.com/google-research/bert)  was modified to accept sample weighting. Comments which contained identities and were non-toxic were given greater weight. This modified BERT was run with several parameters and finally blended using Logistic Regression along with a Bi-directional LSTM model.

1. `Introduction_and_EDA`: This notebook explains the goal of the problem, the data provided and performs some basic exploratory data analysis on the problem
2. `Make_train_test_splits`: Splits the data into training and validation. Also splits the training data into two halves to enable training of BERT, on Kaggle. Makes data into a form usable by BERT
3. `Add weights`: Adds sample weights. This code was modified using different parameters in the weighting function.
