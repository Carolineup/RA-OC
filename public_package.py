"""
Functions that might be used in ranking aggregation methods.
"""

import json

def rank(S):
    result_rank = [0] * len(S)
    sort_rank = sorted(S, reverse=True) # descending order

    for j, s in enumerate(S):
        result_rank[j] = sort_rank.index(s) + 1
    return result_rank

def rank2(S):
    result_rank = [0] * len(S)
    sort_rank = sorted(S, reverse=False) # ascending order
    for j, s in enumerate(S):
        result_rank[j] = sort_rank.index(s) + 1
    return result_rank

def kendall(R_0, S):
    count = 0
    for i, a_i in enumerate(R_0):
        for j, a_j in enumerate(R_0):
            if i<j and a_i<a_j and S[i]>=S[j]:
                count +=1
            if i<j and a_i>a_j and S[i]<=S[j]:
                count +=1
    return count

def load_json_repeat(filename):
    with open(filename, "r") as fh:
        data = json.loads(fh.read())
    return data

def save_result_json(filename, data):
    with open(filename, "w") as fh:
        fh.write(json.dumps(data, indent=2))