"""
Minimum Violations Ranking Method (MVR)
"""

import pulp as pulp
import numpy as np
from tqdm import tqdm
from public_package import load_json_repeat, rank2, kendall, save_result_json


def constant_matrix(R,n):
    C1 = np.zeros(shape=(n, n))
    C2 = np.zeros(shape=(n, n))
    for r_i in R:
        for s, r_s in enumerate(r_i):
            for t, r_t in enumerate(r_i):
                if r_s < r_t and r_s != 0 and r_t != 0:
                    C1[s, t] += 1
                elif r_s > r_t and r_s != 0 and r_t != 0:
                    C2[s, t] += 1
    C = C1 - C2
    return C


def solve_ilp(objective , constraints) :
    prob = pulp.LpProblem('LP1', pulp.LpMaximize)
    prob += objective
    for cons in constraints :
        prob += cons
    status = prob.solve()
    if status != 1 :
        return None
    else:
        return prob.variables(), [v.varValue.real for v in prob.variables()]



if __name__ == "__main__":
    results = {}
    kendall_max = 50 * (50 - 1) / 4  # items:50
    for prob in range(0, 11):
        filename = "data/data_malicious_prob_%s.jsonl" % (prob / 10)
        data_list = load_json_repeat(filename)
        count = len(data_list)
        D_ave = 0
        for data in tqdm(data_list):
            V_NUM = len(data.get('R_0'))
            C = constant_matrix(data['R'], V_NUM)

            variables = [pulp.LpVariable('X_%d_%d' % (s, t), lowBound=0, cat=pulp.LpInteger) for s, C_s in enumerate(C) for t, C_st in enumerate(C_s)]
            # print("variables:", variables)

            objective = sum([C_st * variables[V_NUM * s + t] for s, C_s in enumerate(C) for t, C_st in enumerate(C_s)])
            # print("objective:", objective)

            constraints = []
            for s, C_s in enumerate(C):
                for t, C_st in enumerate(C_s):
                    if s < t:
                        constraints.append(variables[V_NUM * s + t] + variables[V_NUM * t + s] == 1)
                    for k, C_tk in enumerate(C_s):
                        if s < t and s < k and t != k:
                            constraints.append(
                                variables[V_NUM * s + t] + variables[V_NUM * t + k] + variables[V_NUM * k + s] <= 2)
            # print("constraints:", constraints)

            res, res_value = solve_ilp(objective, constraints)
            # print("res:", res)
            # print("res_value:", res_value)

            # result
            X = np.zeros(shape=(V_NUM, V_NUM))
            for j, r in enumerate(res):
                s = r.name.split('_')[1]
                t = r.name.split('_')[2]
                X[int(s), int(t)] = res_value[j]

            S = rank2(X.sum(axis=0))
            D = kendall(data.get('R_0'), S)
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
        save_result_json("data/results/mvr.jsonl", results)