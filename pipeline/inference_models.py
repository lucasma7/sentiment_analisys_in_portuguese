import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()

def get_model(n_model, VOCAB_SIZE, NUM_CLASSES):

    if n_model == 0:
        print("Model EmbeddingBag select")
        EMB_SYZE = 320

        class TweetSentimentClassfier(nn.Module):
            def __init__(self, VOCAB_SYZE, NUM_CLASSES):
                super(TweetSentimentClassfier, self).__init__() 

                self.embed = nn.EmbeddingBag(VOCAB_SYZE, EMB_SYZE, sparse=False)
                self.expand = nn.Linear(EMB_SYZE, NUM_CLASSES, bias=False)

            def forward(self, input, offsets):
                embedded = self.embed(input, offsets)
                return self.expand(embedded)
        return TweetSentimentClassfier(VOCAB_SIZE, NUM_CLASSES)
    

    if n_model == 1:
        print("Model LSTM select")
        EMBED_LEN = 224
        HIDDEN_DIM = 64
        N_LAYERS = 1

        
        class TweetSentimentClassfier_LSTM(nn.Module):
            def __init__(self, VOCAB_SIZE, num_class):
                super(TweetSentimentClassfier_LSTM, self).__init__()

                self.embedding_layer = nn.Embedding(VOCAB_SIZE, EMBED_LEN)
                self.lstm = nn.LSTM(input_size=EMBED_LEN, hidden_size=HIDDEN_DIM, num_layers=N_LAYERS, batch_first=True)
                self.dropout = nn.Dropout(0.1)
                self.linear = nn.Linear(HIDDEN_DIM, num_class, bias=True)
                self.softmax = nn.Softmax(dim=1)

            
            def forward(self, input):
                input = input.long()
                embedded = self.embedding_layer(input)
                lstm_out, _ = self.lstm(embedded)
                lstm_out = self.dropout(lstm_out)
                lstm_out = lstm_out[:, -1, :]
                final_out = self.linear(lstm_out)
                final_out = self.softmax(final_out)
                return final_out
            
        return TweetSentimentClassfier_LSTM(VOCAB_SIZE, NUM_CLASSES)
    
    else:
        print(n_model)
        raise ValueError("Please, select a model: 0: EmbeddingBag 1: LSTM")
