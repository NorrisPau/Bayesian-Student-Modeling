### Bayesian Knowledge Tracing
# source: https://www.upenn.edu/learninganalytics/MOOT/slides/W004V002.pdf

import pandas as pd

# Make dataset with practice sequences
data = {'practice_opportunity': [0, 1, 2, 3, 4],
        'answered_correctly': [0, 1, 1, 0, 0],
        'p_knows_before': [0.4, 0, 0, 0, 0], #P(L_n-1)
        'p_knows_before_given_outcome': [0, 0, 0, 0, 0], #P(L_n-1|outcome=correct) or P(L_n-1|outcome=incorrect)
        'p_knows_now': [0, 0, 0, 0, 0] #P(L_n)
        }
df = pd.DataFrame(data, columns=['practice_opportunity', 'answered_correctly',
                                 'p_knows_before', 'p_knows_before_given_outcome', 'p_knows_now'])

# Set parameters
# p_transit: p(T)
p_transit = 0.1
# p-slip: p(S)
p_slip = 0.3
# p-guess: p(G)
p_guess = 0.2

# Calculate conditional probabilities & Update posterior after each practice sequence in dataframe

for index in df.index:
    print("number of iteration is", index)
    if df.loc[index,'answered_correctly'] == 0:
        df.loc[index,'p_knows_before_given_outcome'] = df.loc[index,'p_knows_before'] * p_slip/ (df.loc[index,'p_knows_before'] * p_slip + (1-df.loc[index,'p_knows_before']) * (1-p_guess))

        #update p_knows_now and assign value to p_knows before of next row
        df.loc[index, 'p_knows_now'] = df.loc[index,'p_knows_before_given_outcome'] + ((1 - df.loc[index,'p_knows_before_given_outcome']) * p_transit)
        df.loc[index + 1, 'p_knows_before'] = df.loc[index, 'p_knows_now']
    elif df.loc[index,'answered_correctly'] == 1:
        df.loc[index, 'p_knows_before_given_outcome'] = df.loc[index,'p_knows_before'] * (1-p_slip) / (df.loc[index,'p_knows_before'] * (1-p_slip) + (1-df.loc[index,'p_knows_before']) * p_guess)

        #update p_knows_now and assign value to p_knows before of next row
        df.loc[index, 'p_knows_now'] =  df.loc[index,'p_knows_before_given_outcome'] + ((1- df.loc[index,'p_knows_before_given_outcome']) * p_transit)
        df.loc[index+1, 'p_knows_before'] = df.loc[index, 'p_knows_now']

#print(df.p_knows_before_given_outcome[0] + (1-df.p_knows_before_given_outcome))

print("hi")