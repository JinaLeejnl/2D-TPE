import argparse
# import evaluate
import json
from rouge import Rouge
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_rouge(answers, targets):
    rouger = Rouge()
    rouge_1_f_scores = []
    rouge_2_f_scores = []
    rouge_l_f_scores = []

    for idx, (answer, target) in enumerate(zip(answers, targets)):
        try:
            scores = rouger.get_scores(answer, target)[0]
            rouge_1_f_scores.append(scores['rouge-1']['f'])
            rouge_2_f_scores.append(scores['rouge-2']['f'])
            rouge_l_f_scores.append(scores['rouge-l']['f'])
        except ValueError as e:
            print(f"Error at index {idx}: {e}")
            print(f"Answer: {answer}")
            print(f"Target: {target}")
            continue

    avg_rouge_1_f = sum(rouge_1_f_scores) / len(rouge_1_f_scores)
    avg_rouge_2_f = sum(rouge_2_f_scores) / len(rouge_2_f_scores)
    avg_rouge_l_f = sum(rouge_l_f_scores) / len(rouge_l_f_scores)

    return {'rouge_1': avg_rouge_1_f,
            'rouge_2': avg_rouge_2_f,
            'rouge_l': avg_rouge_l_f}

def compute_bleu(labels, preds, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[label],
                                  hypothesis=pred,
                                  smoothing_function=SmoothingFunction().method1,
                                  weights=weights) for label, pred in zip(labels, preds)])

def main(args):
    data = []
    with open(args.pred_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    test_examples_answer = [x["output"] for x in data]
    test_predictions_pred = [x["predict"].strip("</s>") for x in data]
    predictions = test_predictions_pred
    references = test_examples_answer

    results = compute_rouge(answers=predictions, targets=references)
    print(results)

    results = compute_bleu(labels=predictions, preds=references)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='../res/fetaqa_res.json', help='')
    args = parser.parse_args()
    main(args)