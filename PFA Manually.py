# PFA Manually
# See http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
# Code found in: https://github.com/theophilee/learner-performance-prediction, train_lr.py

# TODO: Question -> How do I import functions from other script? This gives an error
from PerformanceFactorAnalysis_BayesianKnowledgeTracing import compute_metrics

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss


# Load dataset of bridge_algebra06
train_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_train.csv", sep="\t")
test_df = pd.read_csv(r"data/bridge_algebra06/preprocessed_data_test.csv", sep="\t")
#full_data = pd.read_csv(r"data/bridge_algebra06/preprocessed_data.csv", sep="\t")
print(train_df)

#Tryout for 1 student (student_id = 0) and 1 skill (191)
# #Question: Or how to group students together?
df_student = train_df[train_df["user_id"] == 0]
df_student.describe(include="all")
f = df_student.groupby('skill_id').agg({'timestamp': 'count', 'correct': 'count'})
f.timestamp.max() #skill_id 191 has most practices (74)
df_student_skill = df_student[df_student["skill_id"] == 191]
df_student_skill = df_student_skill.drop(columns = ["user_id", "skill_id"])
df_student_skill.groupby("timestamp"). agg({"item_id": "count", "correct": "count"})

#Make column for time
df_student_skill = df_student_skill.reset_index()
sort = df_student_skill.sort_values(by = "timestamp")
print(sort)
df_student_skill["time"] = range(1, len(df_student_skill)+1)


#Calculate S sequentially (prior successes) and F (prior failures)
df_student_skill["prior_successes"] = 0
df_student_skill["prior_failures"] = 0

"""
for index, row in df_student_skill.iterrows():
    print("number of iteration is", index)
    if row["correct"] == 1:
        row["prior_successes"] = row["prior_successes"] + 1
        print(row["prior_successes"][index-1])

"""

#TODO: Change this code
student_ids = train_df['student_id'].unique()


processed_df = pd.DataFrame()
for student_id in student_ids:
    for skill_id in skill_ids:
        filtered_df = train_df[train_df['skill_id'] == skill_id & train_df['skill_id'] == skill_id]
        # apply function to calculate prior successes/failures

        # add calculation results to processed_df

for index in df_student_skill.index:
    print(df_student_skill.iloc[0:index]["prior_failures"].count())
    print("number of iteration is", index)
    if df_student_skill.loc[index,'correct'] == 1:
        df_student_skill.loc[index, 'prior_successes'] = df_student_skill.iloc[0:index]["correct"].sum() + 1

    elif df_student_skill.loc[index, "correct"] == 0:
        df_student_skill.loc[index, "prior_failures"] = df_student_skill.iloc[0:index]["prior_failures"].sum() + 1

#TODO: Question: How to sum over all rows where correct = 0 and indexing [0:index]???
print(df_student_skill.iloc[0:3].sum())


# print(df_student_skill.loc[df_student_skill["correct"].isin(["1"]), "correct"].sum())
# df_student_skill.iloc[0:index]["prior_failures"].sum()
# print(df_student_skill.loc[df_student_skill["correct"] == "0"].sum())


#von pandas zu numpy array:
#select spalten, die ich brauche
df_student_skill.values
#log regression: OUtput = correct, Input = success, failures


"""
for index in df_student_skill.index:
    print("number of iteration is", index)
    if df_student_skill.loc[index,'correct'] == 1:
        df_student_skill.loc[index, 'prior_successes'] = df_student_skill.loc[index, "prior_successes"] + 1

    elif df_student_skill.loc[index, "correct"] == 0:
        df_student_skill.loc[index, "prior_failures"] = df_student_skill.loc[index, "prior_failures"] + 1

"""

#S_k = number of correctly solved tasks
#S_train = train_df[(train_df["correct"] == "1")]
#F_sk = number of incorrectly solved tasks
#F_train = train_df[(train_df["correct"] == "0")]

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