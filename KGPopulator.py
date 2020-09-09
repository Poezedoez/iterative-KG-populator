import argparse
import os 
import json

from nearest_neighbert.NearestNeighBERT import NearestNeighBERT
from span_extractors.SpanBasedExtractor import SpanBasedExtractor
from distant_supervisor.DistantSupervisor import init_from_config
from distant_supervisor.write import print_dataset
from evaluate import _convert_train, _measure_overlap, _convert_entities, _convert_relations, evaluate

class KGPopulator():
    def __init__(self):
        self.extractor = None
    
    def supervise(self, config_path, data_path, ontology_path, tokenizer_path, output_path):
        supervisor, selection, strat = init_from_config(config_path, data_path, ontology_path, 
                                                        tokenizer_path, output_path)
        supervisor.supervise(strat, selection)
        data_path, types_path = supervisor.get_paths(strat)

        return data_path, types_path


    def train_extractor(self, extractor_type, config_path, train_path, dev_path, 
                        types_path, tokenizer_path, save_path):
        if extractor_type == "span":
            self.extractor = SpanBasedExtractor()
            self.extractor.train(config_path, train_path, dev_path, types_path, 
                                 tokenizer_path, tokenizer_path, save_path)

        elif extractor_type == "knn":
            self.extractor = NearestNeighBERT()
            self.extractor.train(train_path, tokenizer_path, save_path)

        else:
            print("extractor type {} unknown".format(extractor_type))

    
    def ready_inference(self, extractor_type, config_path, types_path, model_path, 
                        tokenizer_path):
        '''
        Load trained extractor model from path with config etc.
        '''
        if extractor_type == "span":
            self.extractor = SpanBasedExtractor()
            self.extractor.ready_inference(config_path, types_path, model_path, tokenizer_path)

        elif extractor_type == "knn":
            self.extractor = NearestNeighBERT()
            self.extractor.configure(config_path)
            self.extractor.ready_inference(model_path, tokenizer_path)

        else:
            print("extractor type {} unknown".format(extractor_type))    
    

    def infer(self, document, output_path=None):
        '''
        Do inference on given json formatted document. 
        Optional: save predictions in output path
        '''
        if not self.extractor:
            print("No extractor trained!")

        extractions = self.extractor.infer(document)

        if output_path:
            with open(output_path, 'w') as json_out:
                json.dump(extractions, json_out)
                print_dataset(output_path)
            

        return extractions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate population for the knowledge graph')
    parser.add_argument('mode', type=str, choices=['supervise','train','inference'],
                        help="mode of the KG population process")     
    parser.add_argument('config', type=str,
                        help="config path to settings for either the distant supervisor or the extractor")
    parser.add_argument('--extractor_type', type=str, choices=['knn','span'],
                        help="name of the extractor model", default="span")  
    parser.add_argument('--train_path', type=str, help="path to the train file")
    parser.add_argument('--dev_path', type=str, help="path to the dev file")
    parser.add_argument('--types_path', type=str, help="path to the types of the KG")
    parser.add_argument('--data_path', type=str, default="data/ScientificDocuments/",
                        help="path to the folder of document objects to label")
    parser.add_argument('--model_path', type=str, default="data/models/",
                        help="path to the folder of the trained model")
    parser.add_argument('--ontology_path', type=str, default="data/ontology/",
                        help="path to the ontology folder")
    parser.add_argument('--tokenizer_path', type=str, default="data/scibert_scivocab_uncased/")
    parser.add_argument('--output_path', type=str, default=None,
                        help="output path for the distant supervision or extractor model")
    parser.add_argument('--target_document', type=str, default="data/test_document.json",
                        help="target document to extract from")


    args = parser.parse_args()
    populator = KGPopulator()

    
    if args.mode == "supervise":
        data_path, types_path = populator.supervise(args.config, args.data_path, args.ontology_path, 
                                                           args.tokenizer_path, args.output_path)

    if args.mode == "train":
        populator.train_extractor(args.extractor_type, args.config, args.train_path, args.dev_path, args.types_path, 
                                  args.tokenizer_path, args.output_path)

    if args.mode == "inference":
        populator.ready_inference(args.extractor_type, args.config, args.types_path, 
                                  args.model_path, args.tokenizer_path)
        populator.infer(args.target_document, args.output_path)