import torch
import torch.utils
from torch.utils import data
import json
import io
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
import spacy
import params
import os
import utils
#python -m spacy download en
import time
import random


class SNLIDataset(data.Dataset):
    '''
    This is a dataset class for SLNI 1.0 data.
    The dataset contains pairs of sentences and the label is their inference of types ["neutral", "contradiction", "entailment"]
    Loading the data and process using spacy/Bert tokenizer
    There are two different sentence construction: once concatenated combined sentence or 2 separate sentences
    '''

    #constants
    LABELS=["neutral", "contradiction", "entailment"]
    SEP_TOKEN='<sep>'
    UNK_TOKEN='<unk>'
    PAD_TOKEN='<pad>'
    MAX_LEN=512
    GLOVE_EMBEDDINGS='glove.42B.300d'
    SPACY_TOKENIZER="en_core_web_sm"
    MAX_SIZE_VOCAB=10000
    MIN_FREQ_VOCAB=2



    def __init__(self, data_path, saved_dir, device, eng_mode='one_sentence', vocab_external=None, tokenized_datapoints_file=None, vocab_file=None):

        self.data_size=0
        #Load data
        self.datapoints=self.load_data(data_path)
        self.vocab = None
        #initial tokenizer is always spacy
        self.tokenizer_type = 'spacy'
        self.eng_mode = eng_mode
        self.vocab_file = vocab_file
        self.vocab_external=vocab_external
        self.device = device
        self.saved_dir = saved_dir
        #Prepare tokenizer
        self.change_tokenizer_and_vocab(tokenizer=self.tokenizer_type, eng_mode=eng_mode)
        #prepare in advance tokenized sentences for better performance.
        if tokenized_datapoints_file is None:
            self.tokenized_datapoints=self.prepare_tokenized_datapoints()
        else:
            self.tokenized_datapoints=utils.load_from_pickle(os.path.join(self.saved_dir, tokenized_datapoints_file))


        #self.spacy_tokenized_combined_sentences=
        #self.spacy_tokenized_two_sentences=

    def load_data(self, data_path):
        '''
        Load data from the dataset json file
        :param data_path: data path to dataset json file
        :return: list of tuples of form (sentence1, sentence2, label)
        '''
        all_datapoints= [] #{'sentence1':[], 'sentence2':[], 'label':[]}

        with io.open(data_path, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = json.loads(line)
                #check that label is valid. otherwaise discard
                if line['gold_label'] in self.LABELS:
                    sentence1=line['sentence1']
                    sentence2 = line['sentence2']
                    label = line['gold_label']
                    all_datapoints.append((sentence1,sentence2, label))
                    self.data_size+=1

        return all_datapoints


    def shuffle_sort_datapoints(self):
        '''

        :return:
        '''

        len_datapoint = lambda datapoint: len(datapoint[0]) + len(datapoint[1])

        if self.tokenizer_type=='bert':
            random.shuffle(self.datapoints)
            self.datapoints = sorted(self.datapoints, key=len_datapoint)

        elif self.tokenizer_type=='spacy':
            random.shuffle(self.tokenized_datapoints)
            self.tokenized_datapoints = sorted(self.tokenized_datapoints, key=len_datapoint)


    def transform_label(self, string_label):

        return self.LABELS.index(string_label)

    def prepare_spacy_vocab(self, vocab_external=None, vocab_file=None):
        self.tokenizer_type = 'spacy'
        self.tokenizer = spacy.load(self.SPACY_TOKENIZER)
        if not vocab_external:
            if vocab_file is None:
                counter = Counter()
                print('Creating vocab')
                for ind in range(self.data_size):
                    sentence1 = self.datapoints[ind][0]
                    sentence2 = self.datapoints[ind][1]
                    list_sentence1 = [token.text for token in self.tokenizer(sentence1.lower())]
                    counter.update(list_sentence1)
                    list_sentence2 = [token.text for token in self.tokenizer(sentence2.lower())]
                    counter.update(list_sentence2)
                    utils.save_to_pickle(counter, os.path.join(self.saved_dir, "vocab_counter.pkl"))
            else:
                counter = utils.load_from_pickle(os.path.join(self.saved_dir, vocab_file))
            self.vocab = Vocab(counter, max_size=self.MAX_SIZE_VOCAB, min_freq=self.MIN_FREQ_VOCAB,
                               specials=[self.PAD_TOKEN, self.SEP_TOKEN, self.UNK_TOKEN])
            self.vocab.load_vectors(self.GLOVE_EMBEDDINGS, unk_init=torch.Tensor.random_)
        else:
            self.vocab = vocab_external




    def change_tokenizer_and_vocab(self, tokenizer='bert' ,eng_mode='one_sentence'):
        '''
        change tokenizer- Can be of form 'bert or 'spacy'. If spacy the eng structue can be of type one sentence (conctenated sentence) or two sentence structure
        :param tokenizer: 'bert' or 'spacy'
        :param eng_mode: 'one_sentence' or 'two_sentence'
        :return: None
        '''
        if tokenizer=='spacy':
            self.tokenizer=spacy.load(self.SPACY_TOKENIZER)# get_tokenizer('spacy', language='en')
            self.eng_mode=eng_mode
            self.tokenizer_type = 'spacy'
            self.prepare_spacy_vocab(self.vocab_external, self.vocab_file)
        else:
            self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.tokenizer_type = 'bert'


    def prepare_tokenized_datapoints(self):
        '''
        Tokenize the full dataset according to the inner vocabulary
        :return: list containing items of structure (tokenized_sentence1,  tokenized_sentence2, label) of the full dataset
        '''

        tokenized_datapoints=[]
        for datapoint in self.datapoints:
            sentence1, sentence2, label=datapoint
            tokenized_sentence1 = [self.vocab.stoi[token.text] if token.text in self.vocab.stoi.keys() else self.vocab.stoi[self.UNK_TOKEN] \
                                   for token in list(self.tokenizer(sentence1.lower()))]
            tokenized_sentence2 = [self.vocab.stoi[token.text] if token.text in self.vocab.stoi.keys() else self.vocab.stoi[self.UNK_TOKEN]\
                                   for token in list(self.tokenizer(sentence2.lower()))]
            tokenized_datapoints.append((tokenized_sentence1,  tokenized_sentence2, label))

        return tokenized_datapoints

    def __getitem__(self, idx):
        '''
        get an iterm from the dataset. This is essentinal function of the dataset.Run by the dataloader to fetch the item
        We will fetch a different output depending on the state of the dataset. If it of Bert model it will be fetch untokenized. If spacy it will be fetched already tokenized.

        :param idx: index of the item
        :return: item that contain sentence 1, sentence 2, label
        '''

        if idx==0:
            self.shuffle_sort_datapoints()

        if self.tokenizer_type == 'bert':
            #Untokenized sentences
            return self.datapoints[idx]
        else:
            # tokenized spacy sentences
            return self.tokenized_datapoints[idx]


    def __len__(self):

        return self.data_size


    def prepare_eng_sentence(self, tokenized_sentence1,tokenized_sentence2, mode):
        #print(list(self.tokenizer(sentence1.lower())))

        if mode=='one_sentence':
            tokenized_combined_sentence=tokenized_sentence1+[self.vocab.stoi[self.SEP_TOKEN]]+tokenized_sentence2
            return torch.tensor(tokenized_combined_sentence, dtype=torch.long).to(self.device)
        else:
            tokenized_sentence1=tokenized_sentence1+[self.vocab.stoi[self.PAD_TOKEN]]*(self.MAX_LEN-len(tokenized_sentence1))
            tokenized_sentence2 = tokenized_sentence2 + [self.vocab.stoi[self.PAD_TOKEN]] * (self.MAX_LEN - len(tokenized_sentence2))
            return torch.tensor(tokenized_sentence1, dtype=torch.long).to(self.device), torch.tensor(tokenized_sentence2, dtype=torch.long).to(self.device)



    def change_from_external_voc(self, vocab, tokenizer_type, eng_mode):
        '''

        :param vocab:
        :param tokenizer_type:
        :param eng_mode:
        :return:
        '''
        self.vocab=vocab
        self.tokenizer_type=tokenizer_type
        self.eng_mode=eng_mode

    def collate_fun(self, batch):
        '''
        the collate function is run by the dataloader immediately after the data is fetched.We will pad the sequences fetched to the same length,
        in case of one sentence construction we will add a seperator token and pad the seqnce to the longest lenght in batch
        in case of two sentences construction, the two sentences are padd to the max lenght constnat value
        In case of Bert we call the bert tokenizer with the batch of inputs to prepare them for the Bert model
        :param batch:
        :return:
        '''

        #print(f"Start collate{time.time()}")
        batch_sentence1 = []
        batch_sentence2 = []
        batch_sentence_combined = []
        batch_labels = []
        sentence_masks=[]

        #Loop through the batch and prepare lists of inputs according to tokenizer type and mode. In case of spacy inputs are already tokenized
        for sentence1, sentence2, label in batch:
            batch_labels.append(torch.tensor(self.transform_label(label)))

            if self.tokenizer_type=='spacy':
                if self.eng_mode=='one_sentence':
                    combined_sentence=self.prepare_eng_sentence(sentence1,sentence2, mode=self.eng_mode)
                    batch_sentence_combined.append(combined_sentence)
                else:
                    senetence1_tokenized, sentence2_tokenized=self.prepare_eng_sentence(sentence1,sentence2, mode=self.eng_mode)
                    batch_sentence1.append(senetence1_tokenized)
                    batch_sentence2.append(sentence2_tokenized)
            if self.tokenizer_type=='bert':
                batch_sentence1.append(sentence1)
                batch_sentence2.append(sentence2)


        #Preparing the batch- Handle padding and  for bert also tokenization
        if self.tokenizer_type=='spacy':
                if self.eng_mode=='one_sentence':#Pad the sequence to the length of the longest in batch
                    prepared_batch_sentences= pad_sequence(batch_sentence_combined, batch_first=True, padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    padding_mask=(prepared_batch_sentences==self.vocab.stoi[self.PAD_TOKEN])
                    prepared_batch = {'inputs': prepared_batch_sentences, 'labels': torch.stack(batch_labels),
                                      'attention_padding_mask': padding_mask}
                else:
                    prepared_sentences1 = pad_sequence(batch_sentence1, batch_first=True, padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    prepared_sentences2 = pad_sequence(batch_sentence2, batch_first=True,  padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    prepared_batch = {'inputs_1': prepared_sentences1, 'inputs_2':  prepared_sentences2, 'labels': torch.stack(batch_labels)}
        else:
            #If Bert call the Bert tokenizer the prepares the batch of inputs to th longest length and add separator and classification tokens and masks
            tokenized_sentences=self.tokenizer(batch_sentence1,batch_sentence2,padding='longest', add_special_tokens=True,return_tensors="pt")
            prepared_batch = {'inputs_ids': tokenized_sentences['input_ids'],
                          'token_type_ids': tokenized_sentences['token_type_ids'],
                          'attention_mask': tokenized_sentences['attention_mask'], 'labels': torch.stack(batch_labels)}

        #print(f"End collate{time.time()}")
        return prepared_batch



