import torch
import torch.utils.data as data


class SmsDataset(data.Dataset):
    def __init__(self, text, label, vocab, max_len):
        super(SmsDataset, self).__init__()

        self.text = text
        self.label = label
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        tmp_text = self.vocab.tokenize(self.text[idx])
        if len(tmp_text) > self.max_len:
            tmp_text = tmp_text[:self.max_len]
        else:
            tmp_text = tmp_text + ["<PAD>"]*(self.max_len - len(tmp_text))

        return {'text': torch.Tensor(self.vocab.word2num(tmp_text)).to(torch.int32), 'label': self.label[idx]}

    def __len__(self):
        return len(self.text)
