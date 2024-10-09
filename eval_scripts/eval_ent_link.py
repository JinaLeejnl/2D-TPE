import json
import argparse


def main(args):
    data = []
    with open(args.pred_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    correct_count = 0 
    multi_candidates_example_count = 0
    for i in range(len(data)):
        # candidate_list = data[i]["candidates_entity_desc_list"]
        ground_truth = data[i]["output"].strip("<>").lower()
        predict = data[i]["predict"].strip("<>").lower()
        # import pdb
        # pdb.set_trace()

        if ground_truth == predict:
            correct_count += 1
        # if len(candidate_list) > 1:
        #     multi_candidates_example_count += 1


    print("correct_count:", correct_count)
    print("acc:", correct_count/len(data))

    # print("multi_candidates_example_count:", multi_candidates_example_count)
    # print("multi_candidates_example_ratio:", multi_candidates_example_count/len(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='../res/ent_link_res.json', help='')
    args = parser.parse_args()
    main(args)