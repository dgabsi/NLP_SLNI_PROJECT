TRAIN_DATA_DIR= './data/snli_1.0/snli_1.0_train.jsonl'
VAL_DATA_DIR='./data/snli_1.0/snli_1.0_dev.jsonl'
TEST_DATA_DIR='./data/snli_1.0/snli_1.0_test.jsonl'


MODELS_DIR='./saved_models'
RUNS_DIR='./experiments'
RESULTS_DIR='./results'


BEST_MODEL_BASIC_RNN='./saved_models/best_models/best_basic_rnn.th'
BEST_MODEL_TRANSFORMER='./saved_models/best_models/best_basic_rnn.th'
BEST_MODEL_RNN_COMBINE='./saved_models/best_models/best_basic_rnn.th'
BEST_MODEL_BERT='./saved_models/best_models/best_basic_rnn.th'

##CONFIGURATION


BATCH_SIZE=32
#LOAD_VOC=True
#MAX_LEN=512
LOAD_MODELS_FROM_CHECKPOINT=False
#EMBEDDING_SIZE=512
#NUM_HEAD=4
#DROPOUT_RATE=0.1
#WEIGHT_DECAY=0.01
PAD_TOKEN='<pad>'
#HIDDEN_SIZE=500


RNN_CONFIG_CONSTANT_CONFIG={'num_classes':3, 'hidden_size':512, 'embeddings_dim':300, 'hidden_pre2_classifier_linear_dim':256,  \
                            'hidden_pre1_classifier_linear_dim':64, 'pad_token':PAD_TOKEN,'dropout_rate':0.1}

RNN_COMBINE_CONSTANT_CONFIG={'num_classes':3, 'hidden_size':512, 'embeddings_dim':300, 'attention_dim': 256, 'linear_pre_1dim': 32,  'linear_pre_2dim': 64,\
                             'pad_token':PAD_TOKEN, 'dropout_rate':0.1}


BERT_CONSTANT_CONFIG={'num_classes':3, 'output_attentions': False, 'output_hidden_states':False}

TRANSFORMER_CONSTANT_CONFIG={'num_classes':3, 'hidden_size':512, 'embeddings_dim':300, 'attention_dim': 256, 'linear_pre_1dim': 32,  'linear_pre_2dim': 64,\
                             'pad_token':PAD_TOKEN, 'dropout_rate':0.1}




BERT_CONFIG_LIST=[{'run_name': ' lr:5e-5', 'num_epochs':4, 'lr':5e-5,'embedding_type': 'bert', 'checkpoint': None}]#, {'run_name': 'lr 3e-5','num_epochs':4,'lr':3e-5}]

RNN_CONFIG_LIST=[{'run_name': 'lr:0.003 bs:32 embedding: glove','num_epochs':4, 'lr':0.003, 'batch_size':BATCH_SIZE, 'embedding_type': 'glove', 'checkpoint': None}]

#RNN_WITH_ATTN_CONFIG_LIST=[{'run_name': ' last1 lr 0.001','num_epochs':1, 'lr':0.001}]

RNN_COMBINE_CONFIG_LIST=[{'run_name': 'lr:0.0003 bs:32 embedding: glove','num_epochs':4, 'lr':0.0003, 'batch_size':BATCH_SIZE, 'embedding_type': 'glove', 'checkpoint': None}]


TRANSFORMER_CONFIG_LIST=[{'run_name': 'lr:5e-5 bs:32 embedding: glove', 'num_epochs':2, 'lr':5e-5, 'batch_size':BATCH_SIZE, 'embedding_type':'glove', 'checkpoint': 'Transformerlr:5e-5 bs:32 embedding: glove  epoch:5 train_iter:85840 val_iter:154003202021 07.pth'}]

#{'run_name': ' lr 5e-5','num_epochs':5, 'lr':5e-5}