from collections import defaultdict
import string
import os
from pathlib import Path
import math
    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=="\xa0":
        return True
    return False

def chunk(sequence, chunk_size=64):
    chunk = []
    for item in sequence:
        if len(chunk)==chunk_size:
            yield chunk
            chunk = []
        chunk.append(item)
    yield chunk


def create_dir_structure(path_dict):
    for _, path in path_dict.items():
        create_dir(path)

def create_dir(path):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)

