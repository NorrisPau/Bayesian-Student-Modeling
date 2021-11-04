### Performance Factor Analysis
# Method from paper: Performance Factor Analysis - A new alternative to Knowledge Tracing
# http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
# Code found in: https://github.com/theophilee/learner-performance-prediction, train_lr.py

# TODO: Question -> Doesn't logistic regression mean we have a hot-encoded dependant variable, which would be accumulated learning = yes/no
# So why don't we use Linear Regression here then?

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss


# TODO: Question -> How can we compare y and y_pred if this is accumulated learning and we never know for sure what that is?
def compute_metrics(y_pred, y):
    acc = accuracy_score(y, np.round(y_pred))
    auc = roc_auc_score(y, y_pred)
    nll = log_loss(y, y_pred)
    mse = brier_score_loss(y, y_pred)
    return acc, auc, nll, mse


# Load data manually
dataset = pd.read_csv("skill_builder_data_2009_2010_ASSISTment_Data.csv")
print(dataset)

###TODO: Question-> WHAT IS X_FILE? WHERE DOES IT COME FROM?

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train logistic regression on sparse feature matrix.')  # container to hold arguments
    # add_argument = What we need
    parser.add_argument('--X_file', type=str, default=dataset)  # -- means optional argument, if not set value is none
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--iter', type=int, default=1000)
    args = parser.parse_args()  # parse the arguments

    features_suffix = (args.X_file.split("-")[-1]).split(".")[0]

    # TODO: Question -> What happens here? In script encode.py a sparse matrix is created...
    # Load sparse dataset
    X = csr_matrix(load_npz(args.X_file))

    train_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_train.csv', sep="\t")
    test_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t")

    # Student-wise train-test split
    # TODO: Question -> Why do we do this?
    user_ids = X[:, 0].toarray().flatten()
    users_train = train_df["user_id"].unique()
    users_test = test_df["user_id"].unique()
    train = X[np.where(np.isin(user_ids, users_train))]
    test = X[np.where(np.isin(user_ids, users_test))]

    # First 5 columns are the original dataset, including label in column 3
    X_train, y_train = train[:, 5:], train[:, 3].toarray().flatten()
    X_test, y_test = test[:, 5:], test[:, 3].toarray().flatten()

    # Train
    model = LogisticRegression(solver="lbfgs", max_iter=args.iter)
    model.fit(X_train, y_train)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    # Write predictions to csv
    test_df[f"LR_{features_suffix}"] = y_pred_test
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    acc_train, auc_train, nll_train, mse_train = compute_metrics(y_pred_train, y_train)
    acc_test, auc_test, nll_test, mse_test = compute_metrics(y_pred_test, y_test)
    print(f"{args.dataset}, features = {features_suffix}, "
          f"auc_train = {auc_train}, auc_test = {auc_test}, "
          f"mse_train = {mse_train}, mse_test = {mse_test}")
