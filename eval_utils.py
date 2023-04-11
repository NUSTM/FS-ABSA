# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import random
from fuzzy_matching import *

# sentiment_word_list = ['positive', 'negative', 'neutral']
sentiment_word_list = ['great', 'bad', 'ok']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type, output_type, special_token_list, use_x_shot, few_shot_data, use_french_data, use_dutch_data, dataset, is_test_mode, do_fuzzy_matching, sample_id):
    special_token_aspect, special_token_opinion, special_token_category, special_token_sentiment, special_token_seperate = special_token_list
    
    targets_seq = seq.replace("</s>", "").replace("<pad>", "").strip()

    if task == 'ASPE':
        as_pairs = []
        # <extra_id_0> aspect0 <extra_id_2> sentiment0 <extra_id_0> aspect1 <extra_id_2> sentiment1...
        if output_type == 'span':
            all_ls = re.split("<..........>", targets_seq)
            is_a = 0
            for i in range(len(all_ls)):
                if all_ls[i] == "":
                    continue
                if (is_a % 2) == 0:
                    aspect = all_ls[i].strip()
                else:
                    sentiment = all_ls[i].strip()
                    if sentiment not in sentiment_word_list:
                        sentiment = sentiment_word_list[random.randrange(0, 3)]
                    as_pairs.append((aspect, sentiment))
                is_a += 1
        # {aspect} is {sentiment} [SSEP] ...
        elif output_type == 'paraphrase':
            pair_seq_list = [s.strip() for s in targets_seq.split('[SSEP]')]
            for pair_seq in pair_seq_list:
                try:
                    aspect, sentiment = pair_seq.split(' is ')
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    aspect, sentiment = '', ''
                as_pairs.append((aspect, sentiment))
        # ({aspect}, {sentiment}); (...)
        elif output_type == 'extraction':
            pair_seq_list = [s.strip().replace('(', '').replace(')', '') for s in targets_seq.split(';')]
            for pair_seq in pair_seq_list:
                try:
                    aspect, sentiment = [s.strip() for s in pair_seq.split(',')]
                except ValueError:
                    try:
                        # print(f'In {seq_type} seq, cannot decode: {s}')
                        pass
                    except UnicodeEncodeError:
                        # print(f'In {seq_type} seq, a string cannot be decoded')
                        pass
                    aspect, sentiment = '', ''
                as_pairs.append((aspect, sentiment))
    else:
        raise NotImplementedError
    
    targets = []
    if task == 'ASPE':
        targets = list(set(as_pairs))
        if is_test_mode != 0:
            formated_targets = []
            if do_fuzzy_matching:
                if use_french_data:
                    if few_shot_data != 0 and use_x_shot != 0:
                         test_data_path = f'./data4fewshot/{use_x_shot}shot/{task}/{dataset}/test_{few_shot_data}.txt'
                    else:
                        test_data_path = f'./data4ml/french/{dataset}/test.txt'
                elif use_dutch_data:
                    if few_shot_data != 0 and use_x_shot != 0:
                         test_data_path = f'./data4fewshot/{use_x_shot}shot/{task}/{dataset}/test_{few_shot_data}.txt'
                    else:
                        test_data_path = f'./data4ml/dutch/{dataset}/test.txt'
                elif few_shot_data != 0 and use_x_shot != 0:
                    test_data_path = f'./data4fewshot/{use_x_shot}shot/{task}/{dataset}/test_{few_shot_data}.txt'
                else:
                    test_data_path = f'./data/{task}/{dataset}/test.txt'
                with open(test_data_path, 'r') as f:
                    lines = f.readlines()
                    review = lines[sample_id].strip()
                    for pred in targets:
                        predict_aspect = pred[0]
                        predict_sentiment = pred[1]
                        formated_aspect = fuzzy_matching(review, predict_aspect)
                        formated_targets.append((formated_aspect, predict_sentiment))
                return formated_targets
    elif task == 'AOPE':
        targets = list(set(ao_pairs))
    elif task == 'ABSC':
        targets = list(set(senti_eles))
    
    return targets


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision*100.0, 'recall': recall*100.0, 'f1': f1*100.0}
    return scores 


def compute_scores(pred_seqs, gold_seqs, task, output_type, special_token_list, use_x_shot, few_shot_data, use_french_data, use_dutch_data, dataset, is_test_mode, do_fuzzy_matching):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(task, gold_seqs[i], 'gold', output_type, special_token_list, use_x_shot, few_shot_data, use_french_data, use_dutch_data, dataset, is_test_mode, do_fuzzy_matching, i)
        pred_list = extract_spans_para(task, pred_seqs[i], 'pred', output_type, special_token_list, use_x_shot, few_shot_data, use_french_data, use_dutch_data, dataset, is_test_mode, do_fuzzy_matching, i)
        print(f"gold:{gold_list}")
        print(f"pred:{pred_list}")
        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    return scores, all_preds, all_labels