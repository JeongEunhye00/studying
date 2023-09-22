import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
from train import train
from validation import validation
from evaluate import evaluate
from Sentiment_classification import SentimentCLS
from dataset.vocab import Vocabulary
from dataset.IMDB_dataset import IMDBDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--train_path", type=str, default='dataset/IMDB/imdb_train.csv')
parser.add_argument("--val_path", type=str, default='dataset/IMDB/imdb_valid.csv')
parser.add_argument("--test_path", type=str, default='dataset/IMDB/imdb_test.csv')
parser.add_argument("--output_path", type=str, default='run/imdb_classification/output')
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--d_ff", type=int, default=512)
parser.add_argument("--prob", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--n_class", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--min_freq", type=int, default=5)

args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(args.seed)  # Seed 고정

# ===== load dataset =====
train_df = pd.read_csv(args.train_path, encoding='utf-8')
val_df = pd.read_csv(args.val_path, encoding='utf-8')
test_df = pd.read_csv(args.test_path, encoding='utf-8')

# ===== build vocabulary =====
X = train_df['review'].to_list()
s_pad_idx = 0
vocab = Vocabulary(X, min_freq=args.min_freq)

# ===== DataLoader =====
train_set = IMDBDataset(train_df, vocab, args.max_len, device)
train_loader = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
val_set = IMDBDataset(val_df, vocab, args.max_len, device)
val_loader = data.DataLoader(val_set, args.batch_size, shuffle=True, drop_last=True)
test_set = IMDBDataset(test_df, vocab, args.max_len, device)
test_loader = data.DataLoader(test_set, args.batch_size, shuffle=True, drop_last=True)


model = SentimentCLS(s_pad_idx, len(vocab), args.d_model, args.num_heads, args.d_ff, args.n_layers,
                     args.prob, args.max_len, args.n_class, device=device).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=args.lr)
best_valid_loss = float('inf')
start_epoch = 1

if args.load_checkpoint is not None:
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_valid_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch'] + 1


# ===== train =====
print('start train')
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

for epoch in range(start_epoch, start_epoch + args.n_epochs):
    output, train_loss = train(model, loss_fn, optimizer, train_loader, device)
    valid_loss, f1, acc = validation(model, loss_fn, val_loader, device)
    print(f'Epoch: {epoch} | Train loss: {train_loss} | Valid loss: {valid_loss} | Valid ACC: {acc * 100:.2f}% | Valid F1: {f1}')

    if valid_loss < best_valid_loss:
        best_model = model
        best_valid_loss = valid_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss},
                   f'{args.output_path}/{epoch}-model.pt')
        print("Model Saved")
print('end train')

# ===== evaluate =====
print('evaluate')
f1, acc = evaluate(best_model, test_loader, device)
print(f"f1 score: {f1}, accuracy: {acc * 100:.2f}%")
