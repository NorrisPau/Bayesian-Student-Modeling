# PFA Manually
# See http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
# Code found in: https://github.com/theophilee/learner-performance-prediction, train_lr.py

# TODO: Question -> How do I import functions from other script? This gives an error
# from PerformanceFactorAnalysis_BayesianKnowledgeTracing.py import compute_metrics

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss


def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    mse = brier_score_loss(y, y_pred)
    return acc, auc, nll, mse

# Load dataset of bridge_algebra06
train_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_train.csv", sep="\t")
test_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_test.csv", sep="\t")
full_data = pd.read_csv(r"data/bridge_algebra06/preprocessed_data.csv", sep="\t")
print(test_df)

#Tryout for 1 student:?
# #Question: Or how to group students together?
# train_student = train_df[(train_df["user_id"] == "0")]

#Formula:
#Question: Do I need S and K individually or is "correct" covering both as a binary variable? And is beta the intercept included automatically?
#S_k = number of correctly solved tasks
S_train = train_df[(train_df["correct"] == "1")]
S_test = test_df[(train_df["correct"] == "1")]
#F_sk = number of incorrectly solved tasks
F_train = train_df[(train_df["correct"] == "0")]
F_test = test_df[(train_df["correct"] == "0")]

#m_train =

#feature_cols = ['S_train', 'F_train']
#X_train = train_df[feature_cols] # Features
#X_test = test_df[feature_cols]
#y = train_df. ???? # Target variable: Correct 
#Target variable = Accumulated learning? Question: We don't know the target, how to compute it then? What do we subset?
#Simulate it?








"""

# X = data (correct/incorrect)
#X_train = train_df["correct"]
#X_test = test_df["correct"]
#X_full = full_data["correct"]

#Student-wise train-test split (error in toarray().flatten())
user_ids = full_data["user_id"]
users_train = train_df["user_id"].unique()
users_test = test_df["user_id"].unique()
train = X_train[np.where(np.isin(user_ids, users_train))]
test = X_test[np.where(np.isin(user_ids, users_test))]

# Train
model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(test_df, train_df)
"""