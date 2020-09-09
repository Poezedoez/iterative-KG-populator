from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as accs
import numpy as np
import json
import argparse
from typing import List, Tuple, Dict, Set
import os
from pathlib import Path
from collections import defaultdict
import shutil
import os
import random
import csv
import pandas as pd

# From spert.evaluator class
# https://github.com/markus-eberts/spert/blob/master/spert/evaluator.py

def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)


def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)


def _compute_retrieval_metrics(gt, pred, types, print_results: bool = False):
    labels = [t for t in types]
    per_type = prfs(gt, pred, labels=labels, average=None)
    micro = prfs(gt, pred, labels=labels, average='micro')[:-1]
    macro = prfs(gt, pred, labels=labels, average='macro')[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    return [m * 100 for m in micro + macro], total_support


## Tuple = 
#   (start, end, entity_type, entity_phrase)
#   or
#   ((head_start, head_end, pseudo_head_type), (tail_start, tail_end, pseudo_tail_type), pred_rel_type, rel_phrase)
def _align_flat(gt: List[List[Tuple]], pred: List[List[Tuple]], contexts):
    assert len(gt) == len(pred) # same amount of sequences, but not predictions per sequence

    gt_flat = []
    pred_flat = []
    phrases_flat = []
    context_flat = []
    unique_predicted_phrases = set()
    known = []
    unseen_yield = []
    types = set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()
        union.update(sample_gt)
        union.update(sample_pred)
        
        for s in union:
            phrases_flat.append(s[3])
            if s in sample_gt:
                know = True
                t = s[2] if s[2].lower()!="other" else 0
                gt_flat.append(t)
                types.add(t)
            else:
                gt_flat.append("0")

            if s in sample_pred:
                t = s[2]
                pred_flat.append(t)
                types.add(t)
            else:
                pred_flat.append("0")

            context_flat.append((s[3], s[2], contexts[s]))
    
    return gt_flat, pred_flat, phrases_flat, context_flat, types


def _score(gt_flat, pred_flat, overlap_flat, types, contexts, split="all", print_results=True):
    assert split.lower() in {"all", "exact", "partial", "new"}
    if split == "all":
        gt_split, pred_split = gt_flat, pred_flat
    else:
        gt_split, pred_split = [], []
        gt_split_known, pred_split_known = [], []
        for gt, pred, overlap in zip(gt_flat, pred_flat, overlap_flat):
            if overlap.lower() == split:
                gt_split.append(gt)
                pred_split.append(pred)
    
    if not gt_split:
        print("No overlap occurrences of split type:", split)
        print("Evaluating on all split types")
        retrieval_metrics, total_support = _compute_retrieval_metrics(gt_flat, pred_flat, types, print_results)
    else:
        retrieval_metrics, total_support = _compute_retrieval_metrics(gt_split, pred_split, types, print_results)

    return retrieval_metrics, total_support


def _convert_entities(sequence, contexts):
    entity_tuples = []
    entities = sequence["entities"]
    tokens = sequence["tokens"]
    for entity in entities:
        phrase = " ".join(tokens[entity["start"]:entity["end"]])
        tuple_ = (entity["start"], entity["end"], entity["type"], phrase)
        entity_tuples.append(tuple_)
        contexts[tuple_] = " ".join(tokens)
        
    return entity_tuples


def _convert_relations(sequence, contexts, pseudo_type="ENTITY"):
    relation_tuples = []
    entities = sequence["entities"]
    relations = sequence["relations"]
    tokens = sequence["tokens"]
    for relation in relations:
        # head
        head_entity = entities[relation["head"]]
        head_phrase = "_".join(tokens[head_entity["start"]:head_entity["end"]])
        head_tuple = (head_entity["start"], head_entity["end"], pseudo_type)

        # tail
        tail_entity = entities[relation["tail"]]
        tail_phrase = "_".join(tokens[tail_entity["start"]:tail_entity["end"]])
        tail_tuple = (tail_entity["start"], tail_entity["end"], pseudo_type)

        # combined
        relation_phrase = " ".join([head_phrase, tail_phrase])
        relation_tuple = (head_tuple, tail_tuple, relation["type"], relation_phrase)
        relation_tuples.append(relation_tuple)
        contexts[relation_tuple] = " ".join(tokens)

        # contexts[relation_tuple] = " |{}| |{}| |{}| >> {} << ".format(head_phrase, relation["type"], tail_phrase, " ".join(tokens))
        
    return relation_tuples


def _convert_train(train_dataset, f_conversion):
    stop_words = set(". , : ; ? ! [ ] ( ) \" ' % - the on 's of a an".split())
    train_exact = defaultdict(lambda: set())
    train_partial = defaultdict(lambda: set())
    contexts = defaultdict(str)
    for sequence in train_dataset:
        conversions = f_conversion(sequence, contexts)
        for c in conversions:
            type_, phrase = c[2], c[-1]
            train_exact[type_].add(phrase)
            partial_terms = set(phrase.split())-stop_words
            train_partial[type_].update(partial_terms)

    return train_exact, train_partial

def _measure_overlap(phrases, gt_types, pred_types, train_exact, train_partial):
    assert(len(phrases)==len(gt_types))
    overlaps = []
    for phrase, type_, pred_type in zip(phrases, gt_types, pred_types):
        if phrase in train_exact[type_]:
           overlap = "exact"
        elif set(phrase.split()).intersection(train_partial[type_]): 
            overlap = "partial"
        else:
            overlap = "new"
        overlaps.append(overlap)
    
    return overlaps


def _calculate_variation(scores):
    a = np.array(scores)
    mean = np.mean(a)
    dif_square = (a-mean)**2
    std = np.sqrt(np.sum(dif_square)/len(scores))
    error_margin = std/np.sqrt(len(scores))

    return mean, error_margin

def evaluate(gt_path, pred_path, train_path=None, split="all", print_results=True):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_dataset = json.load(f)

    if train_path:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
    else:
        train_dataset = []

    train_exact_entities, train_partial_entities = _convert_train(train_dataset, _convert_entities)
    train_exact_relations, train_partial_relations = _convert_train(train_dataset, _convert_relations)

    gt_entities = []
    pred_entities = []

    gt_relations = []
    pred_relations = []

    entity_contexts = defaultdict(str)
    relation_contexts = defaultdict(str)

    for gt_sequence, pred_sequence in zip(gt_dataset, pred_dataset):
        gt_entity_tuples = _convert_entities(gt_sequence, {})
        pred_entity_tuples = _convert_entities(pred_sequence, entity_contexts)
        gt_entities.append(gt_entity_tuples)
        pred_entities.append(pred_entity_tuples)

        gt_relation_tuples = _convert_relations(gt_sequence, {})
        pred_relation_tuples = _convert_relations(pred_sequence, relation_contexts)
        gt_relations.append(gt_relation_tuples)
        pred_relations.append(pred_relation_tuples)

    gt_entities, pred_entities, entity_phrases, entity_contexts, entity_types = _align_flat(gt_entities, pred_entities, entity_contexts)
    gt_relations, pred_relations, relation_phrases, relation_contexts, relation_types = _align_flat(gt_relations, pred_relations, relation_contexts)
    entity_overlap = _measure_overlap(entity_phrases, gt_entities, pred_entities,
                                      train_exact_entities, train_partial_entities)
    relation_overlap = _measure_overlap(relation_phrases, gt_relations, pred_relations, 
                                        train_exact_relations, train_partial_relations)

    print("")
    print("Evaluating overlap type *** {} ***".format(split))
    print("")
    print("--- Entities (named entity recognition (NER)) ---")
    print("An entity is considered correct if the entity type and span is predicted correctly")
    print("")
    ner_eval, total_entities = _score(gt_entities, pred_entities, entity_overlap, 
                                      entity_types, entity_contexts, split, print_results=print_results)
    print("")
    print("--- Relations ---")
    print("")
    print("A relation is considered correct if the relation type and the two "
            "related entity spans are predicted correctly")
    print("")
    rel_eval, total_relations = _score(gt_relations, pred_relations, relation_overlap, 
                                       relation_types, relation_contexts, split, print_results=print_results)

    
    ## Return micro p/r/f1
    return ner_eval[:3]+[total_entities], rel_eval[:3]+[total_relations]


