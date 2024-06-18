import json
import random
import copy as cp
from tqdm import tqdm


def save_json_repeat(filename, data):
    with open(filename, "w") as fh:
        fh.write(json.dumps(data, ensure_ascii=False)+"\n")


def sort_data(fai_j_dict, N, L_i):
    R_0 = [0] * N
    true_rank = sorted(fai_j_dict.items(), key=lambda x: x[1], reverse=True)
    for i, value in enumerate(true_rank):
        id = int(value[0][value[0].index("_")+1:])
        R_0[id] = i + 1
        if i<L_i:
            R_0[id] = i + 1
        else:
            R_0[id] = 0
    return R_0


def probability(R_i, delta):
    P_list = []
    for r in R_i:
        if r!=0:
            p = r**(0-delta)
            P_list.append(p)
        else:
            P_list.append(0)
    return P_list


def reorder_data(R_i):
    R_i_reorder = [0] * len(R_i)
    rank_list = sorted(R_i) # [1,2,3...]
    rank_list = [i for i in rank_list if i !=0]
    for j,rank in enumerate(R_i):
        if rank !=0:
            rank_index = rank_list.index(rank)
            R_i_reorder[j] = rank_index+1
    return R_i_reorder


def data_generation(M, N, prob): # M: the number of users, N: the number of items, prob: the proportion of malicious users
    fai_j_dict = {}
    for j in range(N):
        id = "a_%s" % j
        fai_j = random.uniform(0, 1)
        fai_j_dict[id] = fai_j
    R_0 = sort_data(fai_j_dict, N, N) # A seed ranking

    sample_index = []
    R = [R_0 for i in range(M)]
    for index, value in enumerate(random.sample(R, int(prob * M))):
        sample_index.append(index)

    # Data distribution
    R_1 = []
    for index, value in enumerate(R):
        value_before = cp.deepcopy(value)
        value_after = [0] * len(value)
        # print('index:', index)
        # print("value:", value)
        if index in sample_index:
            delta = random.uniform(-1,0)
        else:
            delta = random.uniform(0,1)
        for j in range(len(value)):
            P_list = probability(value_before, delta)
            element = random.choices(value_before, weights=P_list, k=1)[0]
            element_index = value_before.index(element)
            value_after[element_index] = j+1
            value_before[element_index] = 0
            value_before = reorder_data(value_before)
        # print("value_before:", value_before)
        # print("value_after:", value_after)

        R_1.append(value_after)

    return R_0, R_1



if __name__ == "__main__":
    iter = 100
    for prob in range(0, 11):
        data_list = []
        for i in tqdm(range(iter)):
            data = {}
            R_0, R = data_generation(100, 50, prob / 10)
            data["R_0"] = R_0
            data["R"] = R
            data_list.append(data)
        filename = "data/data_malicious_prob_%s.jsonl" % (prob / 10)
        save_json_repeat(filename, data_list)