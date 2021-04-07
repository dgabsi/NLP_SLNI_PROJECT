import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import params

class SNLI_Transformer(nn.Module):
    '''
    Transformer encoder. Transformer Encoder layers with multiple heads and layers, followed by Adaptive pooling
    to averages the sequence results and 2 linear layers with LeakyRelu activations and batchNorm, output is logits with number of classes
    '''

    def __init__(self, vocab, device, num_classes=3, max_len=1000, hidden_pre2_classifier_linear_dim=256,  hidden_pre1_classifier_linear_dim=64, embeddings_dim=300,  num_heads=4, num_enc_layers=6, dropout_rate=0.1, use_pretrained_embeddings=True):

        super(SNLI_Transformer, self).__init__()

        self.embedding_size =  (embeddings_dim if not use_pretrained_embeddings else vocab.vectors.shape[1])
        self.vocabulary_size = len(vocab)
        self.device=device
        self.hidden_pre2_classifier_linear_dim=hidden_pre2_classifier_linear_dim
        self.hidden_pre1_classifier_linear_dim = hidden_pre1_classifier_linear_dim

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        if use_pretrained_embeddings:
            self.embedding.from_pretrained(vocab.vectors, freeze=False, padding_idx=vocab.stoi[params.PAD_TOKEN])

        self.pos_encoder = PositionalEncoding(device,embeddings_dim,max_len)

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_encoder_layer=nn.TransformerEncoderLayer(self.embedding_size, num_heads, dim_feedforward=self.embedding_size*4, dropout=dropout_rate)

        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_enc_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        #self.gru=nn.GRU(self.embedding_size, hidden_size=200, batch_first=True, bidirectional=True)

        self.linear_pre2_classifier=nn.Linear(self.embedding_size,self.hidden_pre2_classifier_linear_dim)
        self.bn_pre2_classifier = nn.BatchNorm1d(self.hidden_pre2_classifier_linear_dim)
        self.relu2 = nn.ReLU()
        self.linear_pre1_classifier = nn.Linear(self.hidden_pre2_classifier_linear_dim, self.hidden_pre1_classifier_linear_dim)
        self.bn_pre1_classifier = nn.BatchNorm1d(self.hidden_pre1_classifier_linear_dim)
        self.relu1 = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_pre1_classifier_linear_dim, num_classes)


    def forward(self, inputs, attention_padding_mask):

        batch_size=inputs.size(0)
        #print(self.positional_encoding(inputs.size(1)).expand(batch_size, -1, -1).size())
        embedded_inputs=self.embedding(inputs)
        embedded_inputs=self.dropout(self.pos_encoder(embedded_inputs))


        #embedded_inputs=embedded_inputs+sentences_mask
        #embedded_inputs=self.pos_encoder(embedded_inputs)
        #embedded_inputs=self.dropout(self.embedding(inputs)+self.positional_encoding(inputs.size(1)).expand(batch_size, -1, -1))
        embedded_inputs=embedded_inputs.transpose(0,1)

        transformer_outputs=self.transformer(embedded_inputs,src_key_padding_mask=attention_padding_mask)
        transformer_outputs=transformer_outputs.transpose(0,1)
        #gru_outputs, _=self.gru(transformer_outputs)
        avg_outputs = self.pool(transformer_outputs.transpose(1,2)).squeeze()

        linear_pre2_outputs = self.relu2(self.bn_pre2_classifier(self.linear_pre2_classifier(avg_outputs)))
        linear_pre1_outputs = self.relu1(self.bn_pre1_classifier(self.linear_pre1_classifier(linear_pre2_outputs)))

        #outputs=self.dropout(self.relu(self.linear_hidden(outputs))) #.transpose(1,2)
        #outputs=F.adaptive_avg_pool1d(hidden_linear,1).squeeze()
        outputs = self.classifier(linear_pre1_outputs)

        return outputs



class PositionalEncoding(nn.Module):

    def __init__(self, device, embedding_size, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.pos_embed=nn.Embedding(max_len,embedding_size)
        self.device=device
    def forward(self, x):
        seq_pos=torch.arange(x.size()[1]).to(self.device)
        pos_embedding=self.pos_embed(seq_pos).to(self.device)
        x = x + pos_embedding
        return x


