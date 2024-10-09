import json
import sys
import argparse
from table_utils import evaluate


def main(args):
    data = []
    with open(args.pred_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    pred_list = []
    gold_list = []
    for i in range(len(data)):
        if len(data[i]["predict"].strip("</s>").split(">, <")) > 1:
            instance_pred_list = data[i]["predict"].strip("</s>").split(">, <")
            pred_list.append(instance_pred_list)
            gold_list.append(data[i]["output"].strip("</s>").split(">, <"))
        else:
            pred_list.append(data[i]["predict"].strip("</s>"))
            gold_list.append(data[i]["output"].strip("</s>"))

    print(evaluate(gold_list, pred_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='../res/hitab_res.json', help='')
    args = parser.parse_args()
    main(args)