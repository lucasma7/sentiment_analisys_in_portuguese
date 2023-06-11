import torch
from torchtext.vocab import vocab
from .pre_process import clean_data, vocab, tokenize, tok2vec
from .inference_models import get_model

NUM_CLASSES = 3

vocab = vocab()

def load_model(path, n_model):
    model = get_model(n_model, len(vocab), NUM_CLASSES)
    model.load_state_dict(torch.load(path))

    return model

def pipeline(tweet):
    tweet = clean_data(tweet).clean()
    tweet = tokenize(tweet)
    tweet = tok2vec(vocab, tweet)

    return tweet

def predict(path, n_model, tweet):
    model = load_model(path, n_model)
    tweet = pipeline(tweet)

    if n_model == 0:
        tok2vec_tensor =torch.tensor(tweet, dtype=torch.long).to("cpu")
        offsets = torch.tensor([0], dtype=torch.long).to("cpu")
        with torch.no_grad():
            output = model(tok2vec_tensor, offsets)
        return torch.sigmoid(output)[0]
    
    if n_model == 1:
        tok2vec_tensor = torch.LongTensor(tweet).to("cpu")
        tok2vec_tensor = tok2vec_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(tok2vec_tensor)
        return torch.sigmoid(output)[0]

