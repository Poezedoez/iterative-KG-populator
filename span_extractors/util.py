import csv
import json
import os
import random
import shutil
import string
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

CSV_DELIMETER = ';'


def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)


def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)

    return lines[0], lines[1:]


def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith('/') else source + '/'
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = '/'.join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)

        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue

        create_directories_dir(new_dir)

        for file_name in file_names:
            if file_name.endswith('.py'):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def summarize_dict(summary_writer, dic, name):
    table = 'Argument|Value\n-|-'

    for k, v in vars(dic).items():
        row = '\n%s|%s' % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def flatten(l):
    return [i for p in l for i in p]


def get_as_list(dic, key):
    if key in dic:
        return [dic[key]]
    else:
        return []


def extend_tensor(tensor, c, dim=0, fill=0):
    shape = list(tensor.shape)
    shape[dim] = c
    extension = torch.zeros(shape, dtype=tensor.dtype).to(tensor.device)
    extension = extension.fill_(fill)
    extended_tensor = torch.cat([tensor, extension], dim=dim)
    return extended_tensor


def padded_stack(tensors, padding=0):
    max_size = max([t.shape[0] for t in tensors])
    padded_tensors = []

    for t in tensors:
        s = max_size - t.shape[0]
        e = extend_tensor(t, s, dim=0, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def padded_entries(entries, padding={"type_string": "<PAD>", "phrase": "<PAD>", "type_index": -1}):
    lengths = [len(l) for l in entries]
    if not lengths:
        return []
    
    max_size = max(lengths)
    padded_entries = []

    for l in entries:
        s = max_size - len(l)
        for _ in range(0, s):
            l.append(padding)
        padded_entries.append(l)

    return padded_entries


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_nonzero(tensor, padding=0):
    indices = padded_stack([tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding)
    return indices


def swap(v1, v2):
    return v2, v1


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == "\xa0":
        return True
    return False


def split(text):
    doc_tokens = []
    char_to_word_offset = []
    new_token = True
    for c in text:
        if is_whitespace(c):
            new_token = True
        else:
            if c in string.punctuation:
                doc_tokens.append(c)
                new_token = True
            elif new_token:
                doc_tokens.append(c)
                new_token = False
            else:
                doc_tokens[-1] += c
                new_token = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word_offset

def glue_subtokens(subtokens, remove_special_tokens=False):
    glued_tokens = []
    tok2glued = []
    glued2tok = []
    extra = 1 if remove_special_tokens else 0
    for i, token in enumerate(subtokens):
        if token.startswith('##'):
            glued_tokens[len(glued_tokens) - 1] = glued_tokens[len(glued_tokens) - 1] + token.replace('##', '')
        else:
            glued2tok.append(i)
            glued_tokens.append(token)

        tok2glued.append(len(glued_tokens) - 1)

    return glued_tokens[extra:len(glued_tokens)-extra], tok2glued, glued2tok


def convert_to_json_dataset(raw_input, output_path='data/save/za_inference/', save=False):
    sequences, entities, relations = raw_input
    dataset = []
    
    # adjust for special token offset
    i = -1

    for sequence, sample_entities, sample_relations in zip(sequences, entities, relations):
        print(sequence._tokens)
        print([(e[0], e[1], e[2].short_name) for e in sample_entities])
        print([(r[0], r[1], r[2].short_name) for r in sample_relations])
        glued_tokens, tok2glued, _ = glue_subtokens([str(s) for s in sequence], remove_special_tokens=False)

        # entities 
        json_entities = []
        position_mapping = {}
        for start, end, type_, _ in sample_entities:
            # print(start, end, type_)
            # entity_start = tok2glued[start]+i
            # entity_end = tok2glued[end]+i
            entity_start, entity_end = tok2glued[start+i:end+i+1][0], tok2glued[start+i:end+i+1][-1]
            entity_type = type_.short_name
            position_mapping[(entity_start, entity_end, entity_type)] = len(json_entities)
            json_entities.append({"start": entity_start, "end": entity_end, "type": entity_type})

        # relations
        json_relations = []
        # print(position_mapping)
        # for head_entity, tail_entity, type_, _ in sample_relations:
        #     head_start, head_end, head_type = head_entity
        #     tail_start, tail_end, tail_type = tail_entity
        #     relation_head = position_mapping[(head_start+i, head_end+i, head_type.short_name)]
        #     relation_tail = position_mapping[(tail_start+i, tail_end+i, tail_type.short_name)]
        #     relation_type = type_.short_name
        #     json_relations.append({"head": relation_head, "tail": relation_tail, "type": relation_type})

        dataset.append({"tokens": glued_tokens, "entities": json_entities, "relations": json_relations, "orig_id": hash("".join(glued_tokens))})

    if save:
        directory = os.path.dirname(output_path)
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(output_path+'inference_output.json', 'w', encoding='utf-8') as json_file:
            json.dump(dataset, json_file)

    return dataset

def save_json(file, path, name="predictions.json"):
    output_path = os.path.join(path, name)
    with open(output_path, 'w') as json_out:
        json.dump(file, json_out)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad != None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()