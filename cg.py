"""
competition graph (CG) method
"""

import numpy as np
from public_package import load_json_repeat, rank, kendall, save_result_json
from tqdm import tqdm


# Calculate the preference matrix for one user
def preference_matrix(r_i):
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
def pre_matrix_ensemble(R):
    P = []
    for i, r_i in enumerate(R):
        P_i = preference_matrix(r_i)
        P.append(P_i)
    return P


# Calculate the competition matrix
def competition_matrix(R, P):
    A = np.zeros(shape=(len(R[0]), len(R[0])))
    for i, P_i in enumerate(P):
        for s, p_s in enumerate(P_i):
            for t, p_st in enumerate(p_s):
                A[s,t] += p_st
    return A


def rank_aggregation(A):
    Q= []
    for s, a_s in enumerate(A):
        d_in, d_out = 0, 0
        for t, a_st in enumerate(a_s):
            d_in += a_st
            d_out += A[t,s]
        Q_s = (d_in+1)/(d_out+1)
        Q.append(Q_s)
    return Q


if __name__ == "__main__":
    results = {}
    kendall_max = 50 * (50 - 1) / 4  # items:50
    for prob in range(0, 11):
        filename = "data/data_malicious_prob_%s.jsonl" % (prob / 10)
        data_list = load_json_repeat(filename)
        count = len(data_list)
        D_ave = 0
        for data in tqdm(data_list):
            P = pre_matrix_ensemble(data["R"])
            A = competition_matrix(data["R"], P)
            Q = rank_aggregation(A)
            S = rank(Q)
            D = kendall(data["R_0"], S)
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
        save_result_json("data/results/cg.jsonl", results)