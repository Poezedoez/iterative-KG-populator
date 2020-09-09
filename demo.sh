RUN_NAME="`date +%d_%m_%Y_%H_%M_%S`";

TOKENIZER_PATH="data/scibert_scivocab_cased/"
ONTOLOGY_PATH="data/ontology/v7/"
DATA_PATH="data/zeta_objects/"

####################################################################
###############           CREATE DATASET             ###############
####################################################################

MODE="supervise"

# Create a DSD - train set
OUTPUT_PATH_TRAIN="data/demo/${RUN_NAME}/train/"
CONFIG="configs/distant_supervision/distant_supervisor_train_config.json"

python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --data_path $DATA_PATH \
    --ontology_path $ONTOLOGY_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path $OUTPUT_PATH_TRAIN

# Create a DSD - dev set
OUTPUT_PATH_DEV="data/demo/${RUN_NAME}/dev/"
CONFIG="configs/distant_supervision/distant_supervisor_dev_config.json"

python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --data_path $DATA_PATH \
    --ontology_path $ONTOLOGY_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path $OUTPUT_PATH_DEV


###################################################################
##############          TRAIN MODELS                ###############
###################################################################

LABELING="combined_labeling"
MODE="train"
TYPES_PATH="${ONTOLOGY_PATH}ontology_types.json" # equal to dev set types
TRAIN_PATH="${OUTPUT_PATH_TRAIN}${LABELING}/dataset.json"
DEV_PATH="${OUTPUT_PATH_DEV}${LABELING}/dataset.json"


# train knn extractor
EXTRACTOR_TYPE="knn"
CONFIG="configs/extractors/knn_train_config.json"
SAVE_PATH_KNN="data/demo/${RUN_NAME}/save/${EXTRACTOR_TYPE}/"
python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --extractor_type $EXTRACTOR_TYPE \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path $SAVE_PATH_KNN \
    --train_path $TRAIN_PATH \
    --dev_path $DEV_PATH \
    --types_path $TYPES_PATH

# train span extractor
EXTRACTOR_TYPE="span"
CONFIG="configs/extractors/span_train_config.conf"
SAVE_PATH_SPAN="data/demo/${RUN_NAME}/save/${EXTRACTOR_TYPE}/"
python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --extractor_type $EXTRACTOR_TYPE \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path $SAVE_PATH_SPAN \
    --train_path $TRAIN_PATH \
    --dev_path $DEV_PATH \
    --types_path $TYPES_PATH


####################################################################
###############          DO INFERENCE                ###############
####################################################################

MODE="inference"

# do inference with knn extractor
EXTRACTOR_TYPE="knn"
CONFIG="configs/extractors/knn_eval_config.json"
python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --extractor_type $EXTRACTOR_TYPE \
    --tokenizer_path $TOKENIZER_PATH \
    --model_path $SAVE_PATH_KNN \
    --types_path $TYPES_PATH \

# do inference with span extractor
CONFIG="configs/extractors/span_eval_config.conf"
EXTRACTOR_TYPE="span"
python KGPopulator.py  \
    $MODE \
    $CONFIG \
    --extractor_type $EXTRACTOR_TYPE \
    --tokenizer_path $TOKENIZER_PATH \
    --model_path $SAVE_PATH_SPAN \
    --types_path $TYPES_PATH