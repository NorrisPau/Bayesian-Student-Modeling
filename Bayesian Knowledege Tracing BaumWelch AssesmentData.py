import pandas as pd
import numpy as np
import random
random.seed(30)

# Load dataset: ASSESment Data 2009-2010
data = pd.read_csv('skill_builder_data_2009_2010_ASSISTment_Data.csv', index_col=0)
#select important columns
data_student = data[["user_id", "problem_id", "skill_id", "skill_name", "correct", "opportunity"]]
V = data_student["correct"].values

# Look at 1 student and 1 Skill: Ordering Integers
is_student_1 = data_student['user_id'] == 64525  # boolean
student_1 = data_student[is_student_1]  # subset with boolean
len(student_1)  # 800 exercises
student_1_ordering = student_1[student_1["skill_name"] == "Ordering Integers"] #select 1 skill: Ordering Integers
len(student_1_ordering) #39 observations of the skillset Ordering Integers
V_student = student_1_ordering["correct"].values

#Generate Data for Hidden State Mastered
#If Correct: 90% of cases = mastered
hidden_elements = [0, 1]
hidden_generated = []
#1. if correct: 90% mastered, if incorrect: 10% mastered
mastered_weights_correct = [0.1, 0.9]
mastered_weights_incorrect = [0.9, 0.1]

for i in V_student:
    if i == 1:
        hidden_generated += (random.choices(hidden_elements, mastered_weights_correct, k=1))
    elif i == 0:
        hidden_generated += (random.choices(hidden_elements, mastered_weights_incorrect, k=1))

df = pd.DataFrame(
    {'Hidden': hidden_generated,
     'Visible': V_student})

#Calculate Emission and Transition Probability based on generated dataset
#1. Emission
total_mastered = hidden_generated.count(1)
total_not_mastered = hidden_generated.count(0)

mastered_mastered = []
mastered_not_mastered = []
not_mastered_not_mastered = []
not_mastered_mastered = []

for i in range(len(hidden_generated)-1): #because for last element no emission possible
    if hidden_generated[i] == 1 and hidden_generated[i+1] == 1:
        mastered_mastered += "1"
    elif hidden_generated[i] == 1 and hidden_generated[i+1] == 0:
        mastered_not_mastered += "1"
    elif hidden_generated[i] == 0 and hidden_generated[i+1] == 0:
        not_mastered_not_mastered += "1"
    elif hidden_generated[i] == 0 and hidden_generated[i+1] == 1:
        not_mastered_mastered += "1"

mastered_not_mastered = len(mastered_not_mastered)/total_not_mastered #9
mastered_mastered = len(mastered_mastered)/total_mastered #17
not_mastered_mastered = len(not_mastered_mastered)/total_mastered #9
not_mastered_not_mastered = len(not_mastered_not_mastered)/total_not_mastered #3

emission = np.array(((mastered_mastered, mastered_not_mastered),
              (not_mastered_mastered, not_mastered_not_mastered)))

#2. Transition
mastered_correct = []
not_mastered_correct = []
mastered_incorrect = []
not_mastered_incorrect = []

for i in range(len(V_student)):
    if hidden_generated[i] == 1 and V_student[i] == 1:
        mastered_correct += "1"
    if hidden_generated[i] == 0 and V_student[i] == 1:
        not_mastered_correct += "1"
    if hidden_generated[i] == 1 and V_student[i] == 0:
        mastered_incorrect += "1"
    if hidden_generated[i] == 0 and V_student[i] == 0:
        not_mastered_incorrect += "1"

mastered_correct = len(mastered_correct)/total_mastered
not_mastered_correct = len(not_mastered_correct)/total_not_mastered
mastered_incorrect = len(mastered_incorrect)/total_mastered
not_mastered_incorrect = len(not_mastered_incorrect)/total_not_mastered

transition = np.array(((mastered_correct, mastered_incorrect),
              (not_mastered_correct, not_mastered_incorrect)))

initial_distribution_generated = np.array((1, 0)) #knows before, doesn't know before (because in this case the student knew skill before)


### Functions for Algorithm

###Forward Algorithm
def forward(V_student, transition, emission, initial_distribution):
    alpha = np.zeros((V_student.shape[0], transition.shape[0]))
    alpha[0, :] = initial_distribution * emission[:, V_student[0]]  # choose 2. column, all rows

    for t in range(1, V_student.shape[0]):
        for j in range(transition.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(transition[:, j]) * emission[j, V[t]]

    return alpha


### Backward Algorithm
def backward(V_student, transition, emission):
    beta = np.zeros((V_student.shape[0], transition.shape[0]))

    # setting beta(T) = 1
    beta[V_student.shape[0] - 1] = np.ones((transition.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V_student.shape[0] - 2, -1, -1):
        for j in range(transition.shape[0]):
            beta[t, j] = (beta[t + 1] * emission[:, V_student[t + 1]]).dot(transition[j, :])

    return beta

#print(backward(V_student, transition, emission))


### Baum-Welch Algorithm
def baum_welch(V_student, transition, emission, initial_distribution, n_iter=100):
    M = transition.shape[0]  # number of different types of observations (correct, incorrect)
    T = len(V_student)  # T = Number of observed values (correct, incorrect sequence per student)

    for n in range(n_iter):  # for nominator
        alpha = forward(V_student, transition, emission, initial_distribution)
        beta = backward(V_student, transition, emission)

        xi = np.zeros((M, M, T - 1))  # 2x2xnumber observed values-1 (because interested in transitions?)

        ##1. THIS IS THE E-STEP
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, transition) * emission[:, V_student[t + 1]].T, beta[t + 1, :])
            for i in range(M):  # for each different type of observation (correct or incorrect)
                numerator = alpha[t, i] * transition[i, :] * emission[:, V_student[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        ##2. THIS IS THE M-STEP
        transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = emission.shape[1]  # TODO: Question: What is this?
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            emission[:, l] = np.sum(gamma[:, V_student == l], axis=1)

        emission = np.divide(emission, denominator.reshape((-1, 1)))

    return {"a": transition, "b": emission}


transition
emission
print(baum_welch(V_student, transition, emission, initial_distribution_generated, n_iter=1000))
#Result: The algorithm comes to different results, only parts are pretty close like 2nd row of b (emission)


#Komplett Daten selbst generieren
#erste 20 items not mastered, 21-30 mastered
#based on this: incorrect/correct
#mehr correct/incorrect sequence- better evaluation

#wie groß ist fehler abhängig menge an daten - Lernpfad/trajectory

