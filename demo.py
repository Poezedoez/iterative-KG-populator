from KGPopulator import KGPopulator

kg_populator = KGPopulator()

####################################################################
###############           CREATE DATASET             ###############
####################################################################

run_name = "python_demo"
tokenizer_path = "data/scibert_scivocab_cased/"
ontology_path = "data/ontology/v7/"
data_path = "data/zeta_objects/"

# Create a DSD - train set
output_path_train = "data/demo/{}/train/".format(run_name)
config = "configs/distant_supervision/distant_supervisor_train_config.json"
train_path, types_path = kg_populator.supervise(config, data_path, ontology_path, 
                                                tokenizer_path, output_path_train)

# Create a DSD - dev set
output_path_dev = "data/demo/{}/dev/".format(run_name)
config = "configs/distant_supervision/distant_supervisor_dev_config.json"
dev_path, _ = kg_populator.supervise(config, data_path, ontology_path, 
                                              tokenizer_path, output_path_dev)


###################################################################
##############          TRAIN MODELS                ###############
###################################################################

labeling = "combined_labeling"
mode = "train"
types_path = "{}ontology_types.json".format(ontology_path) # equal to dev set types
train_path = "{}{}/dataset.json".format(output_path_train, labeling)
dev_path = "{}{}/dataset.json".format(output_path_dev, labeling)
 

# train knn extractor
extractor_type = "knn"
config = "configs/extractors/knn_train_config.json"
save_path_knn = "data/demo/{}/save/{}/".format(run_name, extractor_type)
kg_populator.train_extractor(extractor_type, config, train_path, 
                             dev_path, types_path, tokenizer_path, save_path_knn)

# train span extractor
extractor_type = "span"
config = "configs/extractors/span_train_config.conf"
save_path_span = "data/demo/{}/save/{}/".format(run_name, extractor_type)
kg_populator.train_extractor(extractor_type, config, train_path, 
                             dev_path, types_path, tokenizer_path, save_path_span)


####################################################################
###############          DO INFERENCE                ###############
####################################################################

test_document = "data/test_document.json"

# do inference with knn extractor
extractor_type = "knn"
output_path = "data/inference_output/knn/"
config = "configs/extractors/knn_eval_config.json"
kg_populator.ready_inference(extractor_type, config, types_path, save_path_knn, 
                             tokenizer_path)
extractions = kg_populator.infer(test_document)


# do inference with span extractor
extractor_type = "span"
output_path = "data/inference_output/span/"
config = "configs/extractors/span_eval_config.conf"
kg_populator.ready_inference(extractor_type, config, types_path, save_path_span, 
                             tokenizer_path)
kg_populator.infer(test_document)
