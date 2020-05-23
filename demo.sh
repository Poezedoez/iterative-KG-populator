RUN_NAME="`date +%d_%m_%Y_%H_%M_%S`";

####################################################################
###############           CREATE DATASET             ###############
####################################################################

LABELING="combined_labeling"
V=5
DATA_PATH="data/zeta_objects/"
COS_THETA=0.83

# Create a DSD - train set
OUTPUT_PATH_TRAIN="data/demo/${RUN_NAME}/train/"
python create_dataset.py  \
    --data_path $DATA_PATH \
    --selection 0 2 \
    --output_path $OUTPUT_PATH_TRAIN \
    --timestamp_given \
    --label_strategy 2 \
    --cos_theta $COS_THETA \
    --ontology_version $V \
    --filter_sentences

# Create a DSD - validation set
OUTPUT_PATH_DEV="data/demo/${RUN_NAME}/dev/"
python create_dataset.py  \
    --data_path $DATA_PATH \
    --selection 2 3 \
    --output_path $OUTPUT_PATH_DEV \
    --timestamp_given \
    --label_strategy 2 \
    --cos_theta $COS_THETA \
    --ontology_version $V \
    --filter_sentences


###################################################################
##############          TRAIN MODELS                ###############
###################################################################

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