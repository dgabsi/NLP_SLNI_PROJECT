import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import params

class Basic_RNN(nn.Module):
    '''
    Basic RNN model. Based on bidirectional lstm, topped Adaptive pooling to averages the sequence results and
     2 linear layers with LeakyRelu activations and batchNorm, output is logits with number of classes
    '''

    def __init__(self, vocab, device, num_classes=3, hidden_size=512, embeddings_dim=300, hidden_pre2_classifier_linear_dim=256,  hidden_pre1_classifier_linear_dim=64, use_pretrained_embeddings=True, pad_token='<pad>',dropout_rate=0.1):

        super(Basic_RNN, self).__init__()

        self.embedding_size =  (embeddings_dim if not use_pretrained_embeddings else vocab.vectors.shape[1])
        self.vocabulary_size = len(vocab)
        self.device=device
        self.hidden_size=hidden_size
        self.hidden_pre2_classifier_linear_dim=hidden_pre2_classifier_linear_dim
        self.hidden_pre1_classifier_linear_dim = hidden_pre1_classifier_linear_dim


        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=vocab.stoi[pad_token])

        #Use of pretrained embeddings
        if use_pretrained_embeddings:
            self.embedding.from_pretrained(vocab.vectors, freeze=False, padding_idx=vocab.stoi[pad_token])

        self.dropout = nn.Dropout(dropout_rate)

        self.lstm=nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        #Avarage pooling the results
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear_pre2_classifier = nn.Linear(self.hidden_size * 2, self.hidden_pre2_classifier_linear_dim)
        self.bn_pre2_classifier = nn.BatchNorm1d(self.hidden_pre2_classifier_linear_dim)
        self.relu2=nn.ReLU()
        self.linear_pre1_classifier = nn.Linear(self.hidden_pre2_classifier_linear_dim, self.hidden_pre1_classifier_linear_dim)
        self.bn_pre1_classifier = nn.BatchNorm1d(self.hidden_pre1_classifier_linear_dim)
        self.relu1 = nn.ReLU()
        self.classifier=nn.Linear(self.hidden_pre1_classifier_linear_dim, num_classes)

    def forward(self, inputs, attention_padding_mask=None):

        batch_size = inputs.size(0)
        init_hidden, init_cell= self.init_hidden_and_cell(batch_size)
        embedded_inputs = self.dropout(self.embedding(inputs))

        #bidirectinal lstm on the concatentaed  sentence
        lstm_output, _=self.lstm(embedded_inputs,(init_hidden, init_cell))

        # adaptive avarage pooling for to compress the sequence dim and prepare for classification
        avg_lstm_output = self.pool(lstm_output.transpose(1, 2)).squeeze()

        #Linear layers for preparation to classification
        pre2_outputs = self.relu2(self.bn_pre2_classifier(self.linear_pre2_classifier(avg_lstm_output)))
        pre1_outputs = self.relu1(self.bn_pre1_classifier(self.linear_pre1_classifier(pre2_outputs)))
        #final classifier
        outputs = self.classifier(pre1_outputs)

        return outputs

    def init_hidden_and_cell(self,batch_size):

        return torch.zeros(4, batch_size, self.hidden_size, device=self.device), torch.zeros(4, batch_size, self.hidden_size, device=self.device)