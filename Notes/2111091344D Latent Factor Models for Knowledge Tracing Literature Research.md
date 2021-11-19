# Latent Factor Models for Knowledge Tracing Literature Research


### Short Summary:
| Name                                                          | **Change to original model**                                                                                                                       | **Important formulas**                                                                                           | 
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | 
| **1. Learning Factor Analysis (Additive Factor Model - AFM)** | -                                                                                                                                                  | $ln( \frac{p}{1-p}) = \sum \alpha_i X_i+\sum \beta_jY_j+ \sum\gamma_jY_jT_j$                                     |  
| **2. Performance Factor Analysis - PFA**                      | Compared to LFA, PFA differentiates between correct and incorrect practice opportunities. The coefficient for each student ($\alpha$) was removed. | $m(i,j \in KC_s,s,f) = \sum_{j \in KC_s}(\beta_j + \gamma_js_{i,j} +\rho_jf_{i,j})$, $p(m) = \frac{1}{1+e^{-m}}$ |    
| **3. Instructional Factor Analysis (IFA)**                    | From PFM, IFM adds a third performance feature $T_{ik}$, the number of times student i has previously got told on relevant KC $k$                                                                                                                                                   | $ln\frac{p_{ij}}{1-p_{ij}}=\theta_i+\sum_k\beta_kQ_{kj}+\sum_kQ_{kj}(\mu_kS_{ik}+\rho_kF_{ik}+v_kT_{ik}))$       |            |                       |     |
| **4. Conjunctive Factor Model (CFM)**                         |  Compared to AFM, which models conjunctivity as an addition of skill parameters, CFM models the conjunctivity as a multiplication of skill parameters.                                                                                                                                                  | $p_{ij}=\Pi_{k=1}^K(\frac{e^\theta_i+\beta_k+\gamma_kT_{ik}}{1+e^\theta_i+\beta_k+\gamma_kT_{jk}})q_{jk}$, $ll_{PMLE}=ll_{MLE}-\frac{1}{2}\lambda\sum_{i=1}^l\theta_i^2, \lambda=1$                                                                                                                 |            |                       |     |


                                                            
---
---
### Long Summary
#### 1. **Learning Factor Analysis – A General Method for Cognitive Model Evaluation and Improvement (Additive Factor Model - AFM)**
**Change to original model:** 
-> None

**Important Formulas**
Statistical Model 
$ln( \frac{p}{1-p}) = \sum \alpha_i X_i+\sum \beta_jY_j+ \sum\gamma_jY_jT_j$
Where: 
$p$ = probability to get an item right
$X$ = covariates for students
$Y$ = covariates for skills (response of student on item)
$T$ = covariates for the number of opportunities practiced on the skills
$YT$= covariates for interaction between skills nad the number of practice opportunities for that skill
$\alpha$ = the coefficient for each student, i.e. the student intercept
$\beta$ = the coefficient for each rule, i.e. the production intercept (coefficient for difficulty of skill)
$\gamma$ = the coefficient for interaction between a production and its opportunities, i.e. the production slope (coefficient for the leanring rate of skill)

**Method**
LFA is a data-mining method for evaluating cognitive models and analyzing student-tutor log data. It combines a statistical model, human expertise and a combinatorial search.  Intuitively speaking, the probability of a student getting a step correct is proportional to the amount of required knowledge the student knows, plus the “easiness” of that skill, plus the amount of learning gained for each practice opportunity.

**Basic Assumptions**
- Method is based on a *power relationship* found by Newell and Rosenbloom stating that the error rate decreases accordint to a power function as the amount of practice increases: $Y = aX^b$ ($Y$=error rate,$X$=number of practice opportunities, $a$ = error rate on first trial reflecting intrinsic difficulty, $b$= learning rate how wasy a skill is to learn)
- Extending the power law, 4 assumptions were made about student learning: 
1. Different students may initially know more or less. Thus, we use an intercept parameter for each student. 
2. Students learn at the same rate. Thus, slope parameters do not depend on student. This is a simplifying assumption to reduce the number of parameters in equation. We chose this simplification, following Draney, Wilson and Pirolli, because we are focused on refining the cognitive model rather than evaluating student knowledge growth. 
3. Some productions are more likely to be known than others. Thus, we use a intercept parameter for each production. 
4. Some productions are easier to learn than others. Thus, we need a slope parameter for each production

Note: The LFA is later also described as Additive Factor Model (AFM) as it adds up 3 parameters in a sum. In the paper "Dynamic Bayesian Networks for Student Modeling" the formula is slightly different formulated, but means the same, just that $\alpha$ is taken by $\theta$ and $Y_j$ by $q_{kt}$ and the equation is rearranged. 

**Source:**
https://www.researchgate.net/publication/225127457_Learning_Factors_Analysis_-_A_General_Method_for_Cognitive_Model_Evaluation_and_Improvement 
---
#### 2. **Performance Factor Analysis - A new Alternative to Knowledge Tracing** 
**Change to original model:** 
Compared to LFA, PFA differentiates between correct and incorrect practice opportunities. Also, $\alpha$, the coefficient for each student was removed, because it is not estimated ahead of time in adaptive situations. 

**Important Formulas**
$m(i,j \in KC_s,s,f) = \sum_{j \in KC_s}(\beta_j + \gamma_js_{i,j} +\rho_jf_{i,j})$
$p(m) = \frac{1}{1+e^{-m}}$
Where:
$m$ = accumulated learning for student $i$
$\beta$ = easiness for each KC (knowledge component)
function of the $n$ of prior observations for student $i$ = Benefit of frequency of prior practice for each KC 
$s$ = tracks the prior successes for the KC for the student, scaled by $\gamma$
$f$ = tracks the prior failures for the KC for the student, scaled by $\rho$

**Method**
Fitting parameters  ($\beta, \gamma,\alpha$ and/or $\rho$) to maximize loglikelihood of the model 

**Basic Assumptions**
The strongest indicator of student learning is their performance on tasks. Correct responses are indicative that current strength is already high and correct responses lead to more learning than incorrect responses. The model is specifically sensitive to incorrectness, because it then acts as indicator and measure of learning in an inverse to correctness. 

**Source:**
http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf 

---
#### 3. Instructional Factor Analysis (IFM) 
**Change to original model**
From PFM, IFM adds a third performance feature $T_{ik}$, the number of times student i has previously got told on relevant KC $k$

**Method**
To predict unseen student's performances, Student ID was treated as a random factor. Leave-one-student-out Cross Validation was performed to compare the different models (AFM, PFA, IFM)

**Basic Assumptions**
For data involving multiple learning interventions, IFM is a more robust choice for student and cognitive modeling. 

**Important Formulas**
$ln\frac{p_{ij}}{1-p_{ij}}=\theta_i+\sum_k\beta_kQ_{kj}+\sum_kQ_{kj}(\mu_kS_{ik}+\rho_kF_{ik}+\nu_kT_{ik}))$

Where: 
$Q_{kj}$ = the Q-matrix cell for step j using skill k

*Performance features:* 
$S_{ik}$=the number of times student i has previously practiced successfully relevant KC $k$
$F_{ik}$ = the number of times student i has previously practiced unsuccessfully relevant KC $k$
$T_{ik}$ = the number of prior Tells student i has had on the skill k

*Skill parameters*: 
$\beta_k$ = coefficient for difficulty of the skill or KC k
$\mu_k$ = the coefficient for the benefit of previous successes on skill k
$\rho_k$ = the coefficient for the benefit of previous failures on skill k
$\nu_k$ = the coefficient for the benefit of previous tells on skill k

**Source:**
https://www.researchgate.net/publication/221570569_Instructional_Factors_Analysis_A_Cognitive_Model_For_Multiple_Instructional_Interventions 
---

#### 4. Conjunctive Factor Model (CFM)
**Change to original model**
Compared to AFM, which models conjunctivity as an addition of skill parameters, CFM models the conjunctivity as a multiplication of skill parameters.

**Method**
CFM is a special case of Embretson’s multicomponent latent trait model and is customized for the high dimensional feature of ITS, as there are many more skills in a cognitive model than the number of cognitive attributes in a traditional assessment.
To fit the parameters, they developed a **penalized maximum likelihood estimation model (PMLE)** to fight overfitting. It penalizes the oversized student parameters in the joint estimation of the student and skill parameters. Maximizing Equation is equivalent to finding a posterior mode for a Bayesian model, with a normal prior on $\theta$ and flat priors on $\beta$  and $\lambda$ . A higher value for $\lambda$ below corresponds to lower prior variance. The BFGS optimization algorithm is used in computing PMLE.

**Basic Assumptions**
Often times in ITS, students face steps with conjunctive skill requirements - The student needs multiple skills to solve the whole step. 

**Important Formulas**
$p_{ij}=\Pi_{k=1}^K(\frac{e^\theta_i+\beta_k+\gamma_kT_{ik}}{1+e^\theta_i+\beta_k+\gamma_kT_{jk}})q_{jk}$
where: 
$Y_{ij}$ = the response of student i on item j 
$\theta_i$ = coefficient for proficiency of student i 
$\beta_k$ = coefficient for difficulty of skill k 
$\gamma_k$ = coefficient for learning rate of skill k 
$T_{ij}$= the number of practice opportunities student i has had on the skill k 
$q_{jk}$ = 1 if item j uses skill k, 0 otherwise
$K$ = the total number of skills in the Q-Matrix

Penalized Maximum Likelihood Estimation method (PMLE): 
$ll_{PMLE}=ll_{MLE}-\frac{1}{2}\lambda\sum_{i=1}^l\theta_i^2, \lambda=1$ by default, where $I$= the total number of students 

**Source:**
http://pact.cs.cmu.edu/pubs/Cen,%20Koedinger%20&%20Junker%2008.pdf 
---
---


**Tags:**
#PFA #IFA #KT #LFA #CFM 