"""
Borda’s method (BM)
"""

from public_package import load_json_repeat, rank, kendall, save_result_json
from tqdm import tqdm


def borda_method(R):
    S = [0] * len(R[0])
    for r_i in R:
        L_i = 0
        for r_ij in r_i:
            if r_ij != 0:
                L_i += 1
        for j, r_ij in enumerate(r_i):
            if r_ij != 0:
                s_ij = L_i - r_ij
            else:
                s_ij = -1
            # print(s_ij)
            S[j] += s_ij
    return S



if __name__ == "__main__":
    results = {}
    kendall_max = 50 * (50 - 1) / 4  # items:50
    for prob in range(0, 11):
        filename = "data/data_malicious_prob_%s.jsonl" % (prob / 10)
        data_list = load_json_repeat(filename)
        count = len(data_list)
        D_ave = 0
        for data in tqdm(data_list):
            S = rank(borda_method(data['R']))
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
        save_result_json("data/results/borda.jsonl", results)