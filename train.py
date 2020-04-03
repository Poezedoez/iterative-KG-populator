import sys
sys.path.append('../spert/')
sys.path.append('../NearestNeighBERT/')
sys.path.append('../distantly-supervised-dataset/')
import argparse
import spert
from NearestNeighBERT import NearestNeighBERT

class TrainWrapper:
    def __init__(self, model_name, model_config, dataset, types, tokenizer, save_path): 
        if model_name.lower() == 'spert':
            spert.train()
        else:
            nn = NearestNeighBERT().configure(model_config)
            nn.train(dataset, tokenizer, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train an entity/relation extractor')
    parser.add_argument('model', type=str, choices=['knn','spert'],
                        help="name of the model to train")    
    parser.add_argument('train_config', type=str,
                        help="path to the model specific config file")
    parser.add_argument('--train_path', type=str, default="data/train_dataset.json",
                        help="path to the train set")
    parser.add_argument('--validation_path', type=str, default="data/dev_dataset.json",
                        help="path to the validation set")
    parser.add_argument('--types_path', type=str, default="data/types.json",
                        help='path to the ontology entity and relation types')
    parser.add_argument('--save_path', type=str, default="data/save/", 
                        help="path to the directory to save trained model in")
    parser.add_argument('--tokenizer', type=str, default="data/scibert_scivocab_uncased")

    args = parser.parse_args()
    trainer = TrainWrapper(args.model, args.train_config, args.train_path, 
                           args.types_path, args.tokenizer, args.save_path)

