TRAIN_DATA_DIR= './data/snli_1.0/snli_1.0_train.jsonl'
VAL_DATA_DIR='./data/snli_1.0/snli_1.0_dev.jsonl'
TEST_DATA_DIR='./data/snli_1.0/snli_1.0_test.jsonl'


MODELS_DIR='./saved_models'
RUNS_DIR='./experiments'
RESULTS_DIR='./results'



##CONFIGURATION


BATCH_SIZE=32
PAD_TOKEN='<pad>'

#Constant configurations
RNN_CONFIG_CONSTANT_CONFIG={'num_classes':3, 'embeddings_dim':300, 'hidden_pre2_classifier_linear_dim':256,  \
                            'hidden_pre1_classifier_linear_dim':64, 'pad_token':PAD_TOKEN,'dropout_rate':0.1}

RNN_COMBINE_CONSTANT_CONFIG={'num_classes':3, 'hidden_size':512, 'embeddings_dim':300,  'linear_pre_1dim': 32,  'linear_pre_2dim': 64,\
                             'pad_token':PAD_TOKEN, 'dropout_rate':0.1}

BERT_CONSTANT_CONFIG={'num_classes':3, 'output_hidden_states':False}

TRANSFORMER_CONSTANT_CONFIG={'num_classes':3, 'max_len':1000, 'hidden_pre2_classifier_linear_dim':256, 'hidden_pre1_classifier_linear_dim': 64, 'embeddings_dim': 300,\
                             'num_enc_layers':6, 'dropout_rate':0.1}





