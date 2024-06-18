import math
import numpy as np
from tqdm import tqdm
from public_package import load_json_repeat, rank, kendall, save_result_json


# Calculate the preference matrix for one user
def preference_matrix(r_i): # Input: Ranking for each user
    P_i =[]
    for s, r_s in enumerate(r_i):
        p_i = []
        for t, r_t in enumerate(r_i):
            if r_t > r_s and r_s > 0:
                p_st = 1
            else:
                p_st = 0
            p_i.append(p_st)
        P_i.append(p_i)
    return P_i


# Calculate the preference matrix for all users
def pre_matrix_ensemble(R): # Input: Ranking matrix
    P = []
    for i, r_i in enumerate(R):
        P_i = preference_matrix(r_i)
        P.append(P_i)
    return P


# Calculate the competition matrix
def competition_matrix(R, P, C): # Input: Ranking matrix, preference matrix, credit matrix
    A = np.zeros(shape=(len(R[0]), len(R[0])))  # Initialize the competition matrix
    for i, P_i in enumerate(P):
        for s, p_s in enumerate(P_i):
            for t, p_st in enumerate(p_s):
                A[s,t] += C[i]*p_st
    return A


# Calculate the reliability matrix
def credibility(R, A): # Input: Ranking matrix, competition matrix
    W = np.zeros(shape=(len(R[0]), len(R[0])))  # Initialize the reliability matrix
    for s, a_s in enumerate(A):
        for t, a_st in enumerate(a_s):
            a_ts = A[t, s]
            if a_st == a_ts and a_st == 0:
                w_st = 0
            else:
                w_st = (a_st-a_ts)/(a_st+a_ts)
            W[s,t] = w_st
    return W


# Update the credit for all users
def update(P, W, R): # Input: Preference matrix, reliability matrix, ranking matrix
    C = []
    for i, P_i in enumerate(P):
        L_i, C_i = 0, 0
        for r_i in R[i]:
            if r_i != 0:
                L_i +=1
        for s, p_s in enumerate(P_i):
            for t, p_st in enumerate(p_s):
                C_i += 2*(p_st*W[s,t])/(L_i*(L_i-1))
        if C_i < 0:
            C_i = 0
        C.append(C_i)
    return C


def rank_aggregation(A):
    Q, Q_hat = [], []
    for s, a_s in enumerate(A):
        K_in, K_out = 0, 0
        for t, a_st in enumerate(a_s):
            K_in += a_st
            K_out += A[t,s]
        Q_s = K_in -K_out
        Q_s_hat = (K_in-K_out)/(K_in+K_out)
        Q.append(Q_s)
        Q_hat.append(Q_s_hat)
    return Q, Q_hat



if __name__ == "__main__":
    results = {}
    kendall_max = 50 * (50 - 1) / 4 # items:50
    for prob in range(0, 11):
        filename = "data/data_malicious_prob_%s.jsonl" % (prob / 10)
        data_list = load_json_repeat(filename)
        count = len(data_list)
        D_ave = 0
        for data in tqdm(data_list):
            iteration_max = 100 # Maximum number of iterations
            C = [1] * len(data['R']) # Initialize each user’s credit to 1
            C_last = C

            P = pre_matrix_ensemble(data['R']) # Preference matrix for all users

            for it in tqdm(range(iteration_max)):
                A = competition_matrix(data['R'], P, C) # Competition matrix
                W = credibility(data['R'], A) # Reliability matrix
                C = update(P, W, data['R']) # Credit matrix

                delta_C = 0
                for i, C_i in enumerate(C):
                    delta_C += math.pow((C_i - C_last[i]), 2)
                if math.sqrt(delta_C) < 1 / math.pow(10, 4):
                    print("迭代次数：", it + 1)
                    break
                C_last = C

            Q, Q_hat = rank_aggregation(A) # Ratio of out- and in-degrees (ROID)
            S = rank(Q_hat) # The final aggregated ranking
            D = kendall(data.get('R_0'), S) # Kendall tau distance
            D_ave += D / count

        prob_id = "prob_%s" % (prob / 10)
        relative_prob_id = "relative_prob_%s" % (prob / 10)
        if prob_id not in results:
            results[prob_id] = {}
        results[prob_id] = D_ave
        if relative_prob_id not in results:
            results[relative_prob_id] = {}
        results[relative_prob_id] = D_ave / kendall_max

        print("kendall_prob_%s:" % (prob / 10), D_ave)
        save_result_json("data/results/oc.jsonl", results)