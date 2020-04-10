RUN_NAME="`date +%d_%m_%Y_%H_%M_%S`";

####################################################################
###############           CREATE DATASET             ###############
####################################################################

LABELING="combined_labeling"
V=5
ENT_PATH="data/ontology/v${V}_ontology_entities.csv" 
REL_PATH="data/ontology/v${V}_ontology_relations.csv" 
EMB_PATH="data/ontology/v${V}_entity_embeddings.json"
DATA_PATH="data/zeta_objects/"

# Create a DSD - train set
OUTPUT_PATH_TRAIN="data/demo/${RUN_NAME}/train/"
python create_dataset.py  \
    --data_path $DATA_PATH \
    --selection 0 2 \
    --output_path $OUTPUT_PATH_TRAIN \
    --timestamp_given \
    --label_function 2 \
    --ontology_entities_path $ENT_PATH \
    --ontology_relations_path $REL_PATH \
    --entity_embedding_path $EMB_PATH

# Create a DSD - validation set
OUTPUT_PATH_DEV="data/demo/${RUN_NAME}/dev/"
python create_dataset.py  \
    --data_path $DATA_PATH \
    --selection 2 3 \
    --output_path $OUTPUT_PATH_DEV \
    --timestamp_given \
    --label_function 2 \
    --ontology_entities_path $ENT_PATH \
    --ontology_relations_path $REL_PATH \
    --entity_embedding_path $EMB_PATH


####################################################################
###############          TRAIN MODELS                ###############
####################################################################

TOKENIZER_PATH="data/scibert_scivocab_cased/"
TYPES_PATH="${OUTPUT_PATH_TRAIN}ontology_types.json" # equal to dev set types
TRAIN_PATH="${OUTPUT_PATH_TRAIN}${LABELING}/dataset.json"
VALIDATION_PATH="${OUTPUT_PATH_DEV}${LABELING}/dataset.json"

# train knn model
SAVE_PATH_KNN="data/demo/${RUN_NAME}/save/knn/"
python train.py \
    knn \
    configs/knn_config.json \
    --train_path $TRAIN_PATH \
    --validation_path $VALIDATION_PATH \
    --types_path $TYPES_PATH \
    --save_path $SAVE_PATH_KNN \
    --tokenizer_path $TOKENIZER_PATH

# train spert
SAVE_PATH_SPERT="data/demo/${RUN_NAME}/save/spert/"
python train.py \
    spert \
    configs/spert_train_config.conf \
    --train_path $TRAIN_PATH \
    --validation_path $VALIDATION_PATH \
    --types_path $TYPES_PATH \
    --save_path $SAVE_PATH_SPERT \
    --tokenizer_path $TOKENIZER_PATH


####################################################################
###############          DO INFERENCE                ###############
####################################################################

# do inference with knn
python infer.py \
    knn \
    configs/knn_config.json \
    --model_path $SAVE_PATH_KNN \
    --types_path $TYPES_PATH \
    --tokenizer_path $TOKENIZER_PATH

# do inference with spert
python infer.py \
    spert \
    configs/spert_infer_config.conf \
    --model_path $SAVE_PATH_SPERT \
    --types_path $TYPES_PATH \
    --tokenizer_path $TOKENIZER_PATH