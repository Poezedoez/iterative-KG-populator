import sys
sys.path.append('../spert/')
sys.path.append('../NearestNeighBERT/')
sys.path.append('../distantly-supervised-dataset/')
import argparse
from spert import load_inference_model, infer
from NearestNeighBERT import NearestNeighBERT


class InferenceWrapper:
    def __init__(self, args):
        self.args = args
        if args.model.lower() == 'spert':
            model = load_inference_model(args)
            f_infer = self._infer_spert
        else:
            model = NearestNeighBERT().configure(args.config)
            model.ready_inference(args.model_path, args.tokenizer_path)
            f_infer = self._infer_knn

        self._f_infer = f_infer
        self.model = model

    def _infer_spert(self, document):
        return infer(self.model, document, self.args.types_path)

    def _infer_knn(self, document):
        return self.model.infer(document)

    def inference(self, document):
        results = self._f_infer(document)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do inference with an entity/relation extractor')
    parser.add_argument('model', type=str, choices=['knn','spert'],
                        help="name of the model to train")    
    parser.add_argument('config', type=str,
                        help="path to the model specific config file for inference")
    parser.add_argument('--model_path', type=str, default="data/save/",
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
    wrapper = InferenceWrapper(args)
    wrapper.inference(example_inference_document)