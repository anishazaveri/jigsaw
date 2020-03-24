# Jigsaw Unintended Bias in Toxicity Classification
**Introduction**: This is code used for the Kaggle competition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). The approach described acheived a [top 6% ranking on Kaggle](https://www.kaggle.com/anneblythe1)

**Goal**: In the original Toxic Comments Classification Challenge models learnt to predict toxicity based on comments. However, they found that models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.

In this competition, you're challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. The data consists of ~2 million comments. 

**Approach**
This competition allowed kernel only submission to Kaggle i.e. your code had to run within 9 hours. You were allowed to upload pretrained models but predictions had to be done within the final kernel.

TLDR: [BERT-base](https://github.com/google-research/bert)  was modified to accept sample weighting. Comments which contained identities and were non-toxic were given greater weight. This modified BERT was run with several parameters and finally blended using Logistic Regression along with a Bi-directional LSTM model.

1. `Introduction_and_EDA`: This notebook explains the goal of the problem, the data provided and performs some basic exploratory data analysis on the problem
2. `Make_train_test_splits`: Splits the data into training and validation. Also splits the training data into two halves to enable training of BERT, on Kaggle. Makes data into a form usable by BERT
3. `Add weights`: Adds sample weights. This code was modified using different parameters in the weighting function.
4. `modified_bert\run_classifier_float_y_weights`: Modified BERT-base (uncased) to accept floats as targets and allow samples weighting. Weighted cross entropy loss was used. 
5. BERT was used with various configurations for 2 epochs, with 4 checkpoints saved.
    1. Default BERT-base cased and uncased
    2. BERT-base cased and uncased - modified to accept floats as targets
    3. BERT-base cased and uncased - modified to accept floats as targets and sample reweighting using 5 different reweighting parameters.
    4. Pretraining BERT for 20K, 36K, 72K examples followed by predictions using methods in b and c
    5. Trying variants of BERT-base with no dropouts
6. `BiLSTM`: A Bidirectional LSTM model was also used with GloVe and fastText embeddings. This was trained for different epochs with checkpoints saved and using various dropout parameters.
7. `Ensembling`: All BERT and LSTM predictions were subjected to Logistic Regression with lasso regularization - this allowed minimizing the number of ensembled models such that the script would run in the provided 9 h window.  The models that worked best in the ensembling were:
	1. test_results_float_p2_e2 : BERT run with the entire training set for 2 epochs, with targets as float and unweighted cross entropy loss
	2. test_results_pretrained_72K_float_p1_e2_weighted-m3: BERT pretrained on 72K of the examples and subsequently run for 1.5 epochs, with targets as float and weighted cross entropy loss with weights assigned by method3
	3. test_results_pretrained_72K_float_p2_weighted-m3: BERT pretrained on 72K of the examples and subsequently run for 1 epoch, with targets as float and weighted cross entropy loss with weights assigned by method3
	4. test_results_float_p1_e2: BERT run with the entire training set for 1.5 epochs, with targets as float and unweighted cross entropy loss
	5. epoch0_lstm_glove_ft_0.05_nc-m3: LSTM with 1 epoch using glove and fasttext embeddings. Dropout was 0.05. No characters were removed from the comment_text. Loss was weighted cross entropy with weights assigned as method 3
	6. test_results_float_p1_cased: Cased BERT run for 0.5 epochs with targets as float and unweighted cross entropy loss
