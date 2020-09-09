import argparse

from span_extractors.args import train_argparser, eval_argparser, infer_argparser, add_base_args
from span_extractors.config_reader import process_configs, process_configs_serial
from span_extractors import input_reader
from span_extractors.frameworks import SpERTFramework
from span_extractors import util

class SpanBasedExtractor():
    '''
    A SpanBased extractor for entities and/or relations.
    Wrapper class with currently supported frameworks:
        - SpERT (entities + relations with global classification) 
    
    Planned for later (work in progress TM)
        - SpET (entities with global classification)
        - SpRT (relations with global classification, predicted entities as input)
        - SpEER (entities + relations using an encoder with local classification using k-nn)

    Optional parameters can be passed. Primary arguments are listed below.
    Attr:
        framework_type (str): spert/spet/sprt/speer
        framework (BaseFramework) : here we store the framework 
    '''

    def __init__(self, framework_type="spert"):
        self.framework_type = framework_type
        self.framework = None
        self.inference_model = None
        self.types_path = None
        self.input_reader = None

    def train(self, train_config, train_path, valid_path, types_path, model_path, tokenizer_path, save_path,
              log_path="../data/log/", input_reader=input_reader.JsonInputReader):
        arg_parser = train_argparser()
        args, _ = arg_parser.parse_known_args()
        add_base_args(args, train_config, log_path, tokenizer_path, 
                      model_path=model_path, save_path=save_path)
        run_args = process_configs_serial(arg_parser, args)
        run_args.label = "{}_train_{}".format(self.framework_type, run_args.feature_enhancer)
        self.framework = get_framework(run_args.model_type)(run_args)
        self.framework.train(train_path=train_path, valid_path=valid_path, types_path=types_path, 
                         input_reader_cls=input_reader)

    def eval(self, eval_config, eval_path, types_path, model_path, tokenizer_path, 
             log_path="../data/log/", input_reader=input_reader.JsonInputReader):
        arg_parser = eval_argparser()
        args, _ = arg_parser.parse_known_args()
        add_base_args(args, eval_config, log_path, tokenizer_path, 
                      model_path=model_path)
        run_args = process_configs_serial(arg_parser, args)
        run_args.label = "{}_eval_{}".format(self.framework_type, run_args.feature_enhancer)
        self.framework = get_framework(run_args.model_type)(run_args)
        self.framework.eval(eval_path=run_args.eval_path, types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)
    
    def ready_inference(self, infer_config, types_path, model_path, tokenizer_path, 
             log_path="../data/log/", input_reader=input_reader.JsonInputReader):
        arg_parser = eval_argparser()
        args, _ = arg_parser.parse_known_args()
        add_base_args(args, infer_config, log_path, tokenizer_path, 
                      model_path=model_path)
        run_args = process_configs_serial(arg_parser, args)
        run_args.label = "{}_infer_{}".format(self.framework_type, run_args.feature_enhancer)
        self.framework = get_framework(run_args.model_type)(run_args)
        self.inference_model = self.framework.ready_inference(types_path, input_reader)
        self.types_path = types_path
        self.input_reader = input_reader

    def infer(self, document):
        if not self.framework:
            print("Ready inference model first")
        predictions = self.framework.infer(self.inference_model, document, self.types_path, self.input_reader)

        return predictions

def __train(run_args):
    model = get_framework(run_args.model_type)(run_args)
    model.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                          types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)

def __eval(run_args):
    model = get_framework(run_args.model_type)(run_args)
    model.eval(eval_path=run_args.eval_path, types_path=run_args.types_path, 
               input_reader_cls=input_reader.JsonInputReader)

def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)

_FRAMEWORKS = {
    'spert': SpERTFramework,
}

def get_framework(name, default=SpERTFramework):
    return _FRAMEWORKS.get(name, default)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")
