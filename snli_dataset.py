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

    LABELS=["neutral", "contradiction", "entailment"]
    SEP_TOKEN='<sep>'
    UNK_TOKEN='<unk>'
    PAD_TOKEN='<pad>'
    MAX_LEN=512

    def __init__(self, data_dir, device, mode='train', voc=None):

        self.data_size=0
        self.datapoints=self.load_data(data_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer_type='bert'
        self.vocab=None
        self.eng_mode='one_sentence'
        self.device=device
        #self.spacy_tokenized_combined_sentences=
        #self.spacy_tokenized_two_sentences=

    def load_data(self, data_dir):
        all_datapoints= [] #{'sentence1':[], 'sentence2':[], 'label':[]}

        with io.open(data_dir, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = json.loads(line)
                if line['gold_label'] in self.LABELS:
                    sentence1=line['sentence1']
                    sentence2 = line['sentence2']
                    label = line['gold_label']
                    all_datapoints.append((sentence1,sentence2, label))
                    self.data_size+=1

        return all_datapoints


    def shuffle_sort_datapoints(self):
        random.shuffle(self.datapoints)
        len_datapoint=lambda datapoint:len(datapoint[0])+len(datapoint[1])
        self.datapoints=sorted(self.datapoints, key=len_datapoint)

    def transform_label(self, string_label):

        return self.LABELS.index(string_label)

    def change_tokenizer_and_vocab(self, tokenizer='bert' ,eng_mode='one_sentence', vocab_external=None, load_counter_from_saved=False):
        if tokenizer=='spacy':
            self.tokenizer=spacy.load("en_core_web_sm")# get_tokenizer('spacy', language='en')
            self.eng_mode=eng_mode
            self.tokenizer_type = 'spacy'
            if not vocab_external:
                if not load_counter_from_saved:
                    counter = Counter()
                    print('Creating vocab')
                    for ind in range(self.data_size):
                        sentence1=self.datapoints[ind][0]
                        sentence2 = self.datapoints[ind][1]
                        list_sentence1= [token.text for token in self.tokenizer(sentence1.lower())]
                        counter.update(list_sentence1)
                        list_sentence2= [token.text for token in self.tokenizer(sentence2.lower())]
                        counter.update(list_sentence2)
                        utils.save_to_pickle(counter, os.path.join(params.MODELS_DIR, "vocab_counter.pkl"))
                else:
                    counter=utils.load_from_pickle(os.path.join(params.MODELS_DIR, "vocab_counter.pkl"))
                self.vocab=Vocab(counter, max_size=10000, min_freq=2, specials=[self.PAD_TOKEN, self.SEP_TOKEN, self.UNK_TOKEN])
                self.vocab.load_vectors('glove.42B.300d', unk_init=torch.Tensor.random_)
            else:
                self.vocab=vocab_external

        else:
            self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.tokenizer_type = 'bert'

    def __getitem__(self, idx):

        if idx==0:
            self.shuffle_sort_datapoints()

        return self.datapoints[idx]


    def __len__(self):

        return self.data_size


    def prepare_eng_sentence(self, sentence1,sentence2, mode):
        #print(list(self.tokenizer(sentence1.lower())))
        tokenized_sentence1= [self.vocab.stoi[token.text] if token.text in self.vocab.stoi.keys() else self.vocab.stoi[self.UNK_TOKEN] for token in list(self.tokenizer(sentence1.lower()))]
        tokenized_sentence2 = [self.vocab.stoi[token.text] if token.text in self.vocab.stoi.keys() else self.vocab.stoi[self.UNK_TOKEN] for token in list(self.tokenizer(sentence2.lower()))]

        if mode=='one_sentence':
            tokenized_combined_sentence=tokenized_sentence1+[self.vocab.stoi[self.SEP_TOKEN]]+tokenized_sentence2
            return torch.tensor(tokenized_combined_sentence, dtype=torch.long).to(self.device)
        else:
            tokenized_sentence1=tokenized_sentence1+[self.vocab.stoi[self.PAD_TOKEN]]*(self.MAX_LEN-len(tokenized_sentence1))
            tokenized_sentence2 = tokenized_sentence2 + [self.vocab.stoi[self.PAD_TOKEN]] * (self.MAX_LEN - len(tokenized_sentence2))
            return torch.tensor(tokenized_sentence1, dtype=torch.long).to(self.device), torch.tensor(tokenized_sentence2, dtype=torch.long).to(self.device)

    '''
    def prepare_spacy_sentences(self):

        for ind in range(self.data_size):
            sentence1=self.datapoints_dict['sentence1'][ind]
            sentence2 = self.datapoints_dict['sentence2'][ind]
            label=self.datapoints_dict['label'][ind]
            prepared_combined_sentence=self.prepare_eng_sentence(sentence1, sentence2, mode = 'one_sentence')
            #self.spacy_tokenized_combined_sentences.append((prepared_combined_sentence, label))

            prepared_sentence1, prepared_sentence2=self.prepare_eng_sentence(sentence1, sentence2, mode = 'two_sentence')
           # self.spacy_tokenized_two_sentences.append((prepared_sentence1, prepared_sentence2, label))

    '''

    def change_from_external_voc(self, vocab, tokenizer_type, eng_mode):
        self.vocab=vocab
        self.tokenizer_type=tokenizer_type
        self.eng_mode=eng_mode

    def collate_fun(self, batch):

        #print(f"Start collate{time.time()}")
        batch_sentence1 = []
        batch_sentence2 = []
        batch_sentence_combined = []
        batch_labels = []

        #Loop through the batch and prepare lists of inputs according to tokenizer type and mode. In case of spacy the inputs are also tokenized in this stage.
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
                if self.eng_mode=='one_sentence':
                    prepared_batch_sentences= pad_sequence(batch_sentence_combined, batch_first=True, padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    padding_mask=(prepared_batch_sentences==self.vocab.stoi[self.PAD_TOKEN])
                    prepared_batch = {'inputs':  prepared_batch_sentences, 'labels': torch.stack(batch_labels), 'attention_padding_mask':padding_mask }
                else:
                    prepared_sentences1 = pad_sequence(batch_sentence1, batch_first=True, padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    prepared_sentences2 = pad_sequence(batch_sentence2, batch_first=True,  padding_value=self.vocab.stoi[self.PAD_TOKEN])
                    prepared_batch = {'inputs_1': prepared_sentences1, 'inputs_2':  prepared_sentences2, 'labels': torch.stack(batch_labels)}
        else:
            tokenized_sentences=self.tokenizer(batch_sentence1,batch_sentence2,padding='longest', add_special_tokens=True,return_tensors="pt")
            prepared_batch = {'inputs_ids': tokenized_sentences['input_ids'],
                          'token_type_ids': tokenized_sentences['token_type_ids'],
                          'attention_mask': tokenized_sentences['attention_mask'], 'labels': torch.stack(batch_labels)}

        #print(f"End collate{time.time()}")
        return prepared_batch



