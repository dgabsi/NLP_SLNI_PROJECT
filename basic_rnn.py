import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import params

class Basic_RNN(nn.Module):

    def __init__(self, vocab, device, num_classes=3, hidden_size=512, embeddings_dim=300, hidden_linear_dim=256, use_pretrained_embeddings=False, dropout_rate=0.1):

        super(Basic_RNN, self).__init__()

        self.embedding_size =  (embeddings_dim if not use_pretrained_embeddings else vocab.vectors.shape[1])
        self.vocabulary_size = len(vocab)
        self.device=device
        self.hidden_size=hidden_size
        self.hidden_linear_dim=hidden_linear_dim


        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size, padding_idx=vocab.stoi[params.PAD_TOKEN])

        if use_pretrained_embeddings:
            self.embedding.from_pretrained(vocab.vectors, freeze=False, padding_idx=vocab.stoi[params.PAD_TOKEN])

        self.dropout = nn.Dropout(dropout_rate)

        self.lstm=nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.linear_pre2_classifier = nn.Linear(self.hidden_size * 2, hidden_linear_dim)
        self.bn_pre2_classifier = nn.BatchNorm1d(hidden_linear_dim)
        self.lrelu2=nn.LeakyReLU()
        self.linear_pre1_classifier = nn.Linear(hidden_linear_dim, 64)
        self.bn_pre1_classifier = nn.BatchNorm1d(64)
        self.lrelu1 = nn.LeakyReLU()
        self.classifier=nn.Linear(64, num_classes)

    def forward(self, inputs, attention_padding_mask=None):

        batch_size = inputs.size(0)
        init_hidden, init_cell= self.init_hidden_and_cell(batch_size)
        embedded_inputs = self.dropout(self.embedding(inputs))

        lstm_output, _=self.lstm(embedded_inputs,(init_hidden, init_cell))
        avg_lstm_output = self.pool(lstm_output.transpose(1, 2)).squeeze()
        pre2_outputs = self.lrelu2(self.bn_pre2_classifier(self.linear_pre2_classifier(avg_lstm_output)))
        pre1_outputs = self.lrelu1(self.bn_pre1_classifier(self.linear_pre1_classifier(pre2_outputs)))
        outputs = self.classifier(pre1_outputs)

        return outputs

    def init_hidden_and_cell(self,batch_size):

        return torch.zeros(4, batch_size, self.hidden_size, device=self.device), torch.zeros(4, batch_size, self.hidden_size, device=self.device)