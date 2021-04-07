import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import params

class RNN_Combined_Model(nn.Module):
    '''
    RNN Combined model . Two bidirectional LSTM each for different sentences, each followed by linear layer and and attention matmul
    and softmax. The outputs are then concatenated and followed by linear layer and Adaptive pooling
    to averages the sequence results and 2 linear layers with LeakyRelu activations and batchNorm, output is logits with number of classes
    '''

    def __init__(self, vocab, device, num_classes=3, hidden_size=512, attention_dim=256, embeddings_dim=300, linear_pre_1dim=32, linear_pre_2dim=64, pad_token='<pad>', use_pretrained_embeddings=True, dropout_rate=0.1):

        super(RNN_Combined_Model, self).__init__()

        self.embedding_size =  (embeddings_dim if not use_pretrained_embeddings else vocab.vectors.shape[1])
        self.vocabulary_size = len(vocab)
        self.hidden_size=hidden_size
        self.device=device


        self.sentence1_embedding = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=vocab.stoi[pad_token])
        self.sentence2_embedding = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=vocab.stoi[pad_token])

        if use_pretrained_embeddings:
            self.sentence1_embedding.from_pretrained(vocab.vectors, freeze=False, padding_idx=vocab.stoi[pad_token])
            self.sentence2_embedding.from_pretrained(vocab.vectors, freeze=False, padding_idx=vocab.stoi[pad_token])

        self.sentence1_dropout = nn.Dropout(dropout_rate)
        self.sentence2_dropout = nn.Dropout(dropout_rate)

        self.sentence1_lstm=nn.LSTM(input_size=self.embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.sentence2_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        self.linear_sentence1=nn.Linear(self.hidden_size * 2, attention_dim)
        self.linear_sentence2 = nn.Linear(self.hidden_size * 2, attention_dim)

        self.linear_combined = nn.Linear(3*attention_dim, attention_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool1 = nn.AdaptiveMaxPool1d(1)

        self.linear_pre2_classifier=nn.Linear(attention_dim,linear_pre_2dim)
        self.bn_pre2_classifier=nn.BatchNorm1d(linear_pre_2dim)
        self.relu2 = nn.ReLU()
        self.linear_pre1_classifier=nn.Linear(linear_pre_2dim, linear_pre_1dim)
        self.bn_pre1_classifier = nn.BatchNorm1d(linear_pre_1dim)
        self.relu1 = nn.ReLU()
        self.classifier=nn.Linear(linear_pre_1dim, num_classes)

    def forward(self, inputs_1, inputs_2):
        sentence1_embedded=self.sentence1_dropout(self.sentence1_embedding(inputs_1))
        sentence2_embedded = self.sentence2_dropout(self.sentence2_embedding(inputs_2))
        sentence1_lstm_outputs,_=self.sentence1_lstm(sentence1_embedded)
        sentence2_lstm_outputs,_ = self.sentence2_lstm(sentence2_embedded)

        linear_sentence1 = self.linear_sentence1(sentence1_lstm_outputs)

        linear_sentence2 = self.linear_sentence1(sentence2_lstm_outputs)
        first_bmm = linear_sentence2.bmm(linear_sentence1.transpose(1, 2))
       # scores=F.softmax(first_bmm/(linear_sentence1.size(2)**0.5), dim=2)
        scores = F.softmax(first_bmm/(linear_sentence1.size(2)**0.5) , dim=2)
        attention_values=scores.bmm(linear_sentence1)

        concat_outputs=torch.cat(( linear_sentence1, linear_sentence2, attention_values), dim=-1)
        linear_outputs=self.linear_combined(concat_outputs)
        avg_outputs = self.pool(linear_outputs.transpose(1, 2)).squeeze()+self.pool1(linear_outputs.transpose(1, 2)).squeeze()

        linear_pre2_outputs=self.relu2(self.bn_pre2_classifier(self.linear_pre2_classifier(avg_outputs)))
        linear_pre1_outputs = self.relu1(self.bn_pre1_classifier(self.linear_pre1_classifier(linear_pre2_outputs)))
        outputs=self.classifier(linear_pre1_outputs)

        return outputs
