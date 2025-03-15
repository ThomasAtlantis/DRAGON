import sys; sys.path.append(".")
import ast
import json
import re
import string
import pandas as pd
from collections import Counter
from dragon.utils.nlp_tools.normalize import normalize_text
from dragon.utils.mlogging import Logger


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(pred_path, gold_path):
    hypos = [line.strip() for line in open(pred_path, "r").readlines()]
    answers = []

    data = pd.read_csv(gold_path, sep="\t", header=None)
    for answer_list in data[1]:
        ground_truths = ast.literal_eval(answer_list)
        answers.append(ground_truths)

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    return f1, em


if __name__ == "__main__":
    logger = Logger.build(__name__, "INFO")

    pred_path="outputs/CRAG-tok/qa_preds.tsv" 
    gold_path="outputs/CRAG-tok/dev.question_answers"
    logs_path="outputs/CRAG-tok/intermediate_data.json"
    
    with open(logs_path) as logs_file:
        data = json.load(logs_file)
    with open(pred_path, "w") as pred_file, open(gold_path, "w") as gold_file:
        for item in data:
            answer = item['output']['pred']
            answer = answer.replace("answer", " ")
            answer = normalize_text(answer)
            pred_file.write(f"{answer}\n")
            gold_file.write(f"{item['question']}\t{item['golden_answers']}\n")

    f1, em = get_scores(pred_path, gold_path)

    logger.info(f"F1-score: {f1:.2f}")
    logger.info(f"EM-score: {em:.2f}")
