# Bayesian Knowledge Tracing: Parameter Learning with Baum-Welch Algorithm
# Sources: https://github.com/myudelson/hmm-scalable
# http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
# Alternativ package nehmen: https://github.com/CAHLR/pyBKT

import pandas as pd
import numpy as np

# Load dataset: ASSESment Data 2009-2010
data = pd.read_csv('skill_builder_data_2009_2010_ASSISTment_Data.csv', index_col=0)
print(data)
data.head()
data_student = data[["user_id", "problem_id", "skill_id", "skill_name", "correct", "opportunity"]]
print(data_student)
V = data_student["correct"].values

# Look at 1 student and 1 Skill: Ordering Integers
is_student_1 = data_student['user_id'] == 64525  # boolean
student_1 = data_student[is_student_1]  # subset with boolean
student_1.head()
len(student_1)  # 800 exercises
s = student_1.groupby("skill_name")
s.first()
student_1_ordering = student_1[student_1["skill_name"] == "Ordering Integers"]
student_1_ordering.head()
len(student_1_ordering) #39 observations of the skillset Ordering Integers

V_student = student_1_ordering["correct"].values

### 0. Initialize A (Transition Probabilities) and B (Emission Probabilities)
p_transit = 0.1
p_slip = 0.3
p_guess = 0.2

# Initial Probabilities
initial_distribution = np.array((0.4, 0.6))  # knows before, doesn't know before

# Transition Probabilities
# 1. #mastered - mastered, not mastered - mastered (forget)
a = np.array(((1, 0),
              (p_transit, 1 - p_transit)))

# Emission Probabilities
# 1. mastered - correct, not mastered - incorrect, not mastered - correct, not mastered - incorrect
b = np.array(((1 - p_slip, p_slip),
              (p_guess, 1 - p_guess)))

# Generate Data:
## Based on Prior, Transition and Emission Probabilites, we sample corresponding hidden states of having mastered the skill
#1. Generate Hidden State MASTERED
import random
hidden_states = [0, 1]
# would need a different initial_hidden_weight = [] for beginning
hidden_weights = [1-p_transit, p_transit] # mastered -> not mastered, not mastered -> mastered
mastered_rnd = random.choices(hidden_states, hidden_weights, k=20)

#Generate Observed State CORRECT
correct_elements = [0, 1]
correct = []
#1. if mastered: 1-p(slip) = correct
correct_weights_mastered = [p_slip, 1-p_slip] #incorrect, correct
#2. #if not mastered: p(guess) = correct
correct_weights_notmastered = [1-p_guess, p_guess]

for i in mastered_rnd:
    if i == 1:
        correct += (random.choices(correct_elements, correct_weights_mastered, k=1))
    elif i == 0:
        correct += (random.choices(correct_elements, correct_weights_notmastered, k=1))

## TODO: QUESTION: How to generate data based on emission and transition probability? To account for all options (See notes)?
# Because for instance mastered - mastered has a probability  of 1, not 0.1 because we calculate no forgetting here?


# Recursive dynamic programming = Forward and Backward Algorithm
### 1. Forward Algorithm
def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]  # choose 2. column, all rows

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha


### Backward Algorithm
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta


### Baum-Welch Algorithm
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]  # number of different types of observations (correct, incorrect)
    T = len(V)  # T = Number of observed values (correct, incorrect sequence per student)

    for n in range(n_iter):  # for nominator
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))  # 2x2xnumber observed values-1 (because interested in transitions?)

        ##1. THIS IS THE E-STEP
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):  # for each different type of observation (correct or incorrect)
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        ##2. THIS IS THE M-STEP
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]  # TODO: Question: What is this?
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return {"a": a, "b": b}


print(baum_welch(V, a, b, initial_distribution, n_iter=100))

# Could do: Validate Result with hmm package, split in train and test data?