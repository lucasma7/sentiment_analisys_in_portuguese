import pipeline as pipe


if __name__ == '__main__':
    path_0 = 'sentiment_analisys_in_portugues/models/modelo_EmbBAG.pt'
    path_1 = 'sentiment_analisys_in_portugues/models/modelo_lstm.pt'


    n_model = int(input("Digite o numero do modelo desejado\n [0]: EmbeddingBag\n [1]: LSTM\n"))
    path = path_0 if n_model == 0 else path_1

    text = input('Insira um texto para análise de sentimentos:')
    probs = pipe.predict(path, n_model, text)
    print(f'Negative   => {round(probs[0].item()*100, 2)}%')
    print(f'Positive   => {round(probs[1].item()*100, 2)}%')
    print(f'Neutral    => {round(probs[2].item()*100, 2)}%')