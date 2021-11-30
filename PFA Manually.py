# PFA Manually
# See http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
# Code found in: https://github.com/theophilee/learner-performance-prediction, train_lr.py

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
"""
#Tryout for 1 student (student_id = 0) and 1 skill (191)
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

"""
for index, row in df_student_skill.iterrows():
    print("number of iteration is", index)
    if row["correct"] == 1:
        row["prior_successes"] = row["prior_successes"] + 1
        print(row["prior_successes"][index-1])
"""

#TODO: Interate over student ids and skill ids
#1. Loop over student ids and skill ids
#2. make Loop correct to fill in prior failures
#3. put all together



"""processed_df = pd.DataFrame()
for student_id in student_ids:
    for skill_id in skill_ids:
        filtered_df = train_df[train_df['skill_id'] == skill_id & train_df['skill_id'] == skill_id]
        # apply function to calculate prior successes/failures

        # add calculation results to processed_df"""

#TODO: prior successes need to start at NA and failures too
# if time == 1: prior_successes = NA
"""
for index in df_student_skill.index:
    #print(df_student_skill.iloc[0:index]["prior_failures"].count())
    print("number of iteration is", index)
    if df_student_skill.loc[index,'correct'] == 1:
        df_student_skill.loc[index, 'prior_successes'] = df_student_skill.iloc[0:index]["correct"].sum() + 1
        if index >= 1:
            last_row = index-1
            df_student_skill.loc[index, "prior_failures"] = df_student_skill.loc[last_row, "prior_failures"]
    elif df_student_skill.loc[index, "correct"] == 0:
        len = index
        #df_student_skill.loc[index, "prior_failures"] = (df_student_skill.iloc[0:index]["correct"].sum())-len + 1
        df_student_skill.loc[index, "prior_failures"] = (len+1) - df_student_skill.iloc[0:index]["correct"].sum()
        if index >= 1:
            last_row = index-1
            df_student_skill.loc[index, "prior_successes"] = df_student_skill.loc[last_row, "prior_successes"]


"""

#1. include time per student and skill
processed_df = pd.DataFrame(train_df)
processed_df = processed_df.sort_values(["user_id", "skill_id", "timestamp"], ascending= (True))
counts = processed_df.groupby(["user_id","skill_id"]).count()

time = []
for i in counts['timestamp']:
    for j in range(i):
        time.append(j)

processed_df["time"] = time
#filtered_df = train_df[train_df['skill_id'] == skill_id & train_df['skill_id'] == skill_id]

#Make loop over whole dataset
prior_successes_count = 0
prior_successes = []
prior_failures = []
prior_failures_count = 0

for i in processed_df.index:
    if processed_df.time[i] == 0:
        prior_successes.append(None)
        prior_successes_count = 0
    else:
        if processed_df.loc[i,"correct"] == 1:
            prior_successes_count += 1
            prior_successes.append(prior_successes_count)
        else:
            prior_successes.append(prior_successes_count)
            prior_failures_count += 1
            prior_failures.append(prior_failures_count)

processed_df["prior_successes"] = prior_successes
processed_df["prior_failures"] = prior_failures

print(processed_df)



#loop over user_id, skill --> save x=i
#if i==x, keep adding previous attempt
#if different means start a new round, becuase new user OR skill




#TODO: Implement dataset for all students and all skills


#



#2. include prior failures and prior successes per student and per skill
groups = processed_df.groupby(['user_id'])
# extract keys from groups
keys = groups.groups.keys() #gives the student & skill groups (like 0,1 is group student 0 with skill 1)

processed_df["prior_successes"] = 0
processed_df["prior_failures"] = 0
#for i in counts["correct"]: #i = 21, 45.. = length of each student_skill group package we want to iterate over

#index = row number
processed_df = processed_df.reset_index()
"""for index in processed_df.index: #index = 0,1,2... -> row
    print("number of iteration is", index)
    if time[index] == 0:
        processed_df = processed_df.reset_index()
    if processed_df.loc[index,"correct"] == 1:
                processed_df.loc[index, "prior_successes"] =  processed_df.iloc[0:index]["correct"].sum()+1

"""

def calculate_sucesses_and_failures (df):
    for index in df.index:
        # print(df_student_skill.iloc[0:index]["prior_failures"].count())
        print("number of iteration is", index)
        if df.loc[index, 'correct'] == 1:
            df.loc[index, 'prior_successes'] = df.iloc[0:index]["correct"].sum() + 1
            if index >= 1:
                last_row = index - 1
                df.loc[index, "prior_failures"] = df.loc[last_row, "prior_failures"]
        elif df.loc[index, "correct"] == 0:
            len = index
            # df_student_skill.loc[index, "prior_failures"] = (df_student_skill.iloc[0:index]["correct"].sum())-len + 1
            df.loc[index, "prior_failures"] = (len + 1) - df.iloc[0:index]["correct"].sum()
            if index >= 1:
                last_row = index - 1
                df.loc[index, "prior_successes"] = df.loc[last_row, "prior_successes"]

# calculate_sucesses_and_failures(df = df_student_skill)

#new_data = processed_df.groupby(["user_id"]).apply(lambda processed_df: calculate_sucesses_and_failures(df = processed_df))


#TODO: Implement logistic Regression for PFA



#von pandas zu numpy array:
#select spalten, die ich brauche
#df_student_skill.values
#log regression: OUtput = correct, Input = success, failures


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