import sys
sys.path.append('../spert/')
sys.path.append('../NearestNeighBERT/')
sys.path.append('../distantly-supervised-dataset/')
import argparse
import spert
from NearestNeighBERT import NearestNeighBERT

class InferenceWrapper:
    def __init__(self, model_name, model_config, types, tokenizer, model_path):
        if model_name.lower() == 'spert':
            model = spert.load_inference_model(model_config, types, tokenizer)
        else:
            model = NearestNeighBERT().configure(model_config)
            model.ready_inference(model_path, tokenizer)

        self.model = model

    def infer(self, document):
        results = self.model.infer(document)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do inference with an entity/relation extractor')
    parser.add_argument('model', type=str, choices=['knn','spert'],
                        help="name of the model to train")    
    parser.add_argument('infer_config', type=str,
                        help="path to the model specific config file for inference")
    parser.add_argument('--trained_model_path', type=str, default="data/save/",
                        help="path to the directory of trained model objects")
    parser.add_argument('--types_path', type=str, default="data/types.json",
                        help='path to the file with ontology entity and relation types')
    parser.add_argument('--tokenizer_path', type=str, default="data/scibert_scivocab_uncased",
                        help="path to the directory of tokenizer/vocab/weights")

    args = parser.parse_args()
    example_inference_document = {
        'guid': 'IDtest',
        'sentences': [
            'In contrast with the normal auto-encoder, denoising auto-encoder (Vincent etal., 2010) could improve the model learning ability by introducing noise in the form of random tokendeleting and swapping in this input sentence',
            'Neural machine translations (NMT) (Bahdanau et al., 2015; Vaswani et al., 2017) have set several state-of-the-art  new  benchmarks  (Bojar  et  al.,  2018;  Barrault  et  al.,  2019)',
            'Our empirical results show that the UNMT model outperformed the SNMT model, although both of their performances decreased significantly  in  this  scenario.'
        ]
    }
    inferencer = InferenceWrapper(args.model, args.infer_config, args.types_path, args.tokenizer, args.trained_model_path)
    inferencer.infer(example_inference_document)