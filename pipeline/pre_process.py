import re
import pickle
import subprocess
import spacy
import string

subprocess.run(['python', '-m', 'spacy', 'download', 'pt_core_news_md'])


nlp = spacy.load("pt_core_news_md")


class clean_data:

  def __init__(self, tweet):
    self.tweet = tweet

    self.pattern_url_1 = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    self.pattern_url_2 =r'www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    self.pattern_notations = r'@[\w]*'

    self.pattern_hashtag = r'#[\w]*'

    self.subs_emoticon = {r":\)\)|:\)|:D|[Kk]{2,}": "",
                          r":\(|:\(\(": "",
                          r"8\)|:[Pp]": ""}

    self.subs_vic = {r"\s[Qq]\s": " que ",
                          r"\s[Vv]c\s|\svoce\s": " você ",
                          r"\spra\s": " para ",
                          r"\sngm\s": " ninguém ",
                          r"\sto\s|\stô\s": " estou ",
                          r"\s[Nn]\s": " não ",
                          r"\s[Ss]\s": " sim ",
                          r"\s[Ss]o\s": " só ",
                          r"\s{2,}": " "}

    self.pattern_remove_emojis = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\u2000-\u206F\u2600-\u26FF\u2700-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F]+'

    self.pattern_character_special = r'^[^A-ZÁÉÍÓÚÀÂÊÔÃÕÜÇa-záéíóúàâêôãõüç\d+\-]'
    self.pattern_excessive_character = r'([^\w\s\.])\1{2,}'

  def remove_with_replace(self):

    self.tweet = re.sub(self.pattern_url_1, '', self.tweet)
    self.tweet = re.sub(self.pattern_url_2, '', self.tweet)
    self.tweet = re.sub(self.pattern_notations, '', self.tweet)
    self.tweet = re.sub(self.pattern_hashtag, '', self.tweet)
    
    for pattern, replacement in self.subs_emoticon.items():
        self.tweet = re.sub(pattern, replacement, self.tweet)
        
    for pattern, replacement in self.subs_vic.items():
        self.tweet = re.sub(pattern, replacement, self.tweet)
    
    self.tweet = re.sub(self.pattern_character_special, '', self.tweet)
    self.tweet = re.sub(self.pattern_excessive_character, '', self.tweet)
    self.tweet = re.sub(self.pattern_remove_emojis, '', self.tweet)

    self.tweet = self.tweet.strip()

    return self.tweet
  
  def clean(self):
    return self.remove_with_replace()
  

def vocab():
   with open('pipeline/vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)
    return vocab
   
def tokenize(tweet):

   stop_words = list(spacy.lang.pt.stop_words.STOP_WORDS)
   punct = list(string.punctuation)

   remove_char = stop_words + punct + [' ','  ', '...', '🤣']


   doc = nlp.tokenizer(tweet)
   tokens_list = [s.text for s in doc if s.text not in remove_char]

   return tokens_list
   
def tok2vec(vocab, tokens):  
  return [vocab[s] for s in tokens]