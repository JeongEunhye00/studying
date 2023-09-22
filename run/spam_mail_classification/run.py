import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from dataset.SMS_dataset import SmsDataset
from dataset.vocab import Vocabulary
from models import RNN_model
from train import train
from evaluate import evaluate


# ===== HyperParameter setting =====
epochs = 50
learning_rate = 0.001
batch_size = 64
embedding_size = 32
hidden_size = 32
num_class = 2
n_layers = 2


# ===== Load dataset =====
df = pd.read_csv('dataset/SMS/spam.csv', encoding='latin-1')
df = df.loc[:, ['v1', 'v2']]
df.dropna(inplace=True)
df['v1'] = df['v1'].replace(['ham', 'spam'], [0, 1])
df.drop_duplicates(inplace=True)

X = df['v2'].to_list()
Y = df['v1'].to_list()


# ===== build vocabulary =====
vocab = Vocabulary(X, min_freq=3)


# ===== set max_seq_len =====
seq_lens = []
for x in X:
    seq_lens.append(len(x))
max_len = sum(seq_lens)//len(X)


# ===== Split dataset -> train : test = 0.7: 0.3 =====
train_idx = int(len(X)*0.7)
train_x, train_y = X[:train_idx], Y[:train_idx]
test_x, test_y = X[train_idx:], Y[train_idx:]
print(f"train - total: {len(train_x)}개, ham: {train_y.count(0)}개, spam: {train_y.count(1)}개")
print(f"test - total: {len(test_x)}개, ham: {test_y.count(0)}개, spam: {test_y.count(1)}개")


# ===== DataLoader =====
train_set = SmsDataset(train_x, train_y, vocab, max_len=max_len)
train_loader = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
test_set = SmsDataset(test_x, test_y, vocab, max_len=max_len)
test_loader = data.DataLoader(test_set, batch_size, shuffle=True, drop_last=True)


# ===== model, loss_fn, optimizer =====
model = RNN_model(batch_size, embedding_size, hidden_size, num_class, len(vocab), n_layers)
# model = LSTM(batch_size, embedding_size, hidden_size, num_class, len(vocab), n_layers)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ===== train =====
print('start train')
all_losses = []
for i in range(1, epochs+1):
    output, loss = train(model, loss_fn, optimizer, train_loader)
    all_losses.append(loss)
    if i % 5 == 0:
        print(f"epoch [{i}] - train loss: {loss}")
print('end train')


# ===== evaluate =====
print('evaluate')
f1, acc = evaluate(model, test_loader)
print(f"f1 score: {f1}, accuracy: {acc}")
