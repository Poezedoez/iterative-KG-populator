import sys
SPERT_PATH='../spert/'
sys.path.append(SPERT_PATH)
sys.path.append('../NearestNeighBERT/')
import argparse
import subprocess
from NearestNeighBERT import NearestNeighBERT

def train(model_name, train_config, train_path, validation_path, types_path, tokenizer_path, save_path):
    """
    Train a model based on the given model_name. 
    """
    if model_name.lower() == 'spert':
        subprocess.run(["python", SPERT_PATH+"spert.py", "train", 
                        "--config", train_config,
                        "--train_path", train_path,
                        "--valid_path", validation_path,
                        "--types_path", types_path,
                        "--tokenizer_path", tokenizer_path,
                        "--model_path", tokenizer_path,
                        "--save_path", save_path])
    else:
        knn = NearestNeighBERT().configure(train_config)
        knn.train(train_path, tokenizer_path, save_path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train an entity/relation extractor')
    parser.add_argument('model', type=str, choices=['knn','spert'],
                        help="name of the model to train")    
    parser.add_argument('config', type=str,
                        help="path to the model specific train parameters")
    parser.add_argument('--train_path', type=str, default="data/train_dataset.json",
                        help="path to the train set")
    parser.add_argument('--validation_path', type=str, default="data/dev_dataset.json",
                        help="path to the validation set")
    parser.add_argument('--types_path', type=str, default="data/types.json",
                        help='path to the ontology entity and relation types')
    parser.add_argument('--save_path', type=str, default="data/save/", 
                        help="path to the directory to save trained model in")
    parser.add_argument('--tokenizer_path', type=str, default="data/scibert_scivocab_uncased/")

    args = parser.parse_args()
    train(args.model, args.config, args.train_path, args.validation_path,
                           args.types_path, args.tokenizer_path, args.save_path)

