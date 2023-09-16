from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class Vocabulary:

    def __init__(self, dataset, min_freq):
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        self.stoi = {k:j for j,k in self.itos.items()}
        self.dataset = dataset
        self.min_freq = min_freq
        self.build_vocabulary()

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        tok = TreebankWordTokenizer()
        text = tok.tokenize(text)
        stopword = set(stopwords.words('english'))
        result = []
        for word in text:
            if word not in stopword:
                result.append(word)
        return result

    def build_vocabulary(self):
        from collections import defaultdict
        voca_dict = defaultdict(int)
        voca_list = []
        for data in self.dataset:
            for word in self.tokenize(data):
                voca_dict[word] += 1

        for k, v in voca_dict.items():
            if v >= self.min_freq:
                voca_list.append(k)

        for i, voca in enumerate(voca_list):
            self.itos[i+2] = voca
            self.stoi[voca] = i+2

    def word2num(self, tokenized_text):
        word2num_text = []
        for token in tokenized_text:
            if token in self.stoi:
                word2num_text.append(self.stoi[token])
            else:
                word2num_text.append(self.stoi['<UNK>'])

        return word2num_text
