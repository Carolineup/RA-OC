"""
FLAGR：A flexible high-performance library for rank aggregation - SoftwareX 2023
"""

from public_package import load_json_repeat, kendall, save_result_json
from tqdm import tqdm
import pyflagr.Weighted as Weighted


if __name__ == "__main__":
    results = {}
    kendall_max = 50 * (50 - 1) / 4  # items:50
    for prob in tqdm(range(0, 11)):
        lists = "testdata/testdata_prob_%s.csv" % (prob / 10)
        # method_1 = Weighted.PreferenceRelationsGraph(alpha=0.5, beta=0.5) # alpha/bata: apply the values as suggested in the respective studies
        method_1 = Weighted.Agglomerative(c1=2.5, c2=1.5) # c1/c2: apply the values as suggested in the respective studies
        # method_1 = Weighted.DIBRA(aggregator='combsum:borda', dist='footrule', prune=False)
        df_out, _ = method_1.aggregate(input_file=lists)

        data_list = load_json_repeat("data/data_malicious_prob_%s.jsonl" % (prob / 10))
        rank_list = [[0 for i in range(len(data_list[0]['R_0']))] for j in range(len(data_list))]
        df_out_list = df_out[['Voter', 'ItemID']].values.tolist()
        for data in df_out_list:
            id = data[0].split('-')
            topic_id = int(id[0][1:]) - 1
            item_id = int(id[1][1:]) - 1
            item_rank = int(data[1])
            rank_list[topic_id][item_id] = item_rank

        count = len(data_list)
        D_ave = 0
        for i, rank_i in enumerate(rank_list):
            D = kendall(data_list[i].get('R_0'), rank_i)
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
        # save_result_json("data/results/PreferenceRelationsGraph.jsonl", results)
        save_result_json("data/results/Agglomerative.jsonl", results)
        # save_result_json("data/results/BC-SFD.jsonl", results)