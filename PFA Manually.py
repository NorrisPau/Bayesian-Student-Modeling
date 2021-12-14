# PFA Manually
# See http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
# Code found in: https://github.com/theophilee/learner-performance-prediction, train_lr.py

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

# Load dataset of bridge_algebra06
# train_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_train.csv", sep="\t")
# test_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_test.csv", sep="\t")
full_data = pd.read_csv(
    "https://raw.githubusercontent.com/theophilee/learner-performance-prediction/master/data/bridge_algebra06/preprocessed_data.csv",
    sep="\t")
print(full_data)

# 1. Include time per student and skill
processed_df = pd.DataFrame(full_data)
processed_df = processed_df.sort_values(["user_id", "skill_id", "timestamp"], ascending=(True))
counts = processed_df.groupby(["user_id", "skill_id"]).count()

time = []
for i in counts['timestamp']:
    for j in range(i):
        time.append(j)

processed_df["time"] = time

# 2.Loop over whole dataset to include prior successes and failures per student and per skill in groups
prior_successes_count = 0
prior_successes = []
prior_failures = []
prior_failures_count = 0

for i in processed_df.index:
    if processed_df.time[i] == 0:
        prior_successes.append(None)
        prior_failures.append(None)
        prior_successes_count = 0
        prior_failures_count = 0
    else:
        if processed_df.loc[i, "correct"] == 1:
            prior_successes_count += 1
            prior_successes.append(prior_successes_count)
            prior_failures.append(prior_failures_count)
        else:
            prior_successes.append(prior_successes_count)
            prior_failures_count += 1
            prior_failures.append(prior_failures_count)

processed_df["prior_successes"] = prior_successes
processed_df["prior_failures"] = prior_failures

print(processed_df)


####Implement Logistic Regression for PFA
# Based on prior successes and prior failures we want to predict if correct = 1/0 in current row

# Output = correct, Input = prior success/failures
df_small = processed_df[["user_id", "correct", "skill_id", "time", "prior_successes", "prior_failures"]]
df_small = df_small.fillna(0)

# TODO: Question - How to run a logistic regression per skill?
# each skill has one logistic regression (for all students)
# filter data on skill
# do code on filtered dataframe for each skill

# for i in range(skills)
# filtered_df = df[df["skill"]=i]
# skill_models = {}
# skill_models["skill_1"]=sklearnModel
# key = skill_id, value = model

skill_models = {} #save each logistic regression model per skill in a dictionary
    #key: skill name
    #value:

x_filtered = []
y_filtered = []
model = pd.DataFrame()

for i in df_small.skill_id:
    filtered_df_skill = df_small[df_small["skill_id"] == i]
    y_filtered.append(filtered_df_skill.correct)
    x_filtered.append(filtered_df_skill.prior_sucesses)
    x_filtered.append(filtered_df_skill.prior_failures)
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

model = LogisticRegression()
model.fit(x_filtered, y_filtered)




x = df_small[["prior_successes", "prior_failures"]]
y = df_small["correct"]
x = np.array(x)
# x = x.reshape(-1, 1)
y = np.array(y)

# Train/Test-Split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size=0.3)

# Build Model
model = LogisticRegression()
model.fit(x_training_data, y_training_data)

# Making predictions
predictions = model.predict(x_test_data)
classification_report(y_test_data, predictions)
print(confusion_matrix(y_test_data, predictions))





#NOTES

#
# 100days code challenge python (codeacademy)
# automate the boring
"""
#Student-wise train-test split (error in toarray().flatten())
user_ids = full_data["user_id"]
users_train = train_df["user_id"].unique()
users_test = test_df["user_id"].unique()
train = X_train[np.where(np.isin(user_ids, users_train))]
test = X_test[np.where(np.isin(user_ids, users_test))]

# Train
model = LogisticRegression(solver="lbfgs", max_iter=1000)
model.fit(test_df, train_df)

# S_k = number of correctly solved tasks
# S_train = train_df[(train_df["correct"] == "1")]
# F_sk = number of incorrectly solved tasks
# F_train = train_df[(train_df["correct"] == "0")]
# m_train =
# feature_cols = ['S_train', 'F_train']
# X_train = train_df[feature_cols] # Features
# X_test = test_df[feature_cols]
# y = train_df. ???? # Target variable: Correct
"""
