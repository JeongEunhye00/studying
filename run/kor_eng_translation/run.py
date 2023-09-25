import argparse
import random
import sentencepiece as spm
import torch
import torch.nn as nn
import random
import pandas as pd
import torch.utils.data as data
from torch.utils.data.dataset import random_split
from torch.optim import Adam
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from train import train
from validation import validation
from evaluate import evaluate
from dataset.KorEngTranslation.KorEngTranslationDataset import KorEngTranslationDataset
from Transformer.model import Transformer


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--data_path", type=str, default='dataset/KorEngTranslation/2_대화체.csv')
parser.add_argument("--output_path", type=str, default='run/kor_eng_translation/output')
parser.add_argument("--s_vocab_path", type=str, default='dataset/KorEngTranslation/src_vocab/tokenizer_32000.model')
parser.add_argument("--t_vocab_path", type=str, default='dataset/KorEngTranslation/trg_vocab/tokenizer_15462.model')
parser.add_argument("--s_voca_len", type=int, default=32000)
parser.add_argument("--t_voca_len", type=int, default=15462)
parser.add_argument("--pad_idx", type=int, default=3)
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--d_ff", type=int, default=512)
parser.add_argument("--prob", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--s_max_len", type=int, default=61)
parser.add_argument("--t_max_len", type=int, default=283)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

seed_everything(args.seed)  # Seed 고정

s_tok = spm.SentencePieceProcessor()
s_tok.Load(args.s_vocab_path)
t_tok = spm.SentencePieceProcessor()
t_tok.Load(args.t_vocab_path)

df = pd.read_csv(args.data_path, encoding='utf-8')

# ===== DataLoader =====
datasets = KorEngTranslationDataset(df, args.s_max_len, args.t_max_len, s_tok, t_tok)
train_set, val_set, test_set = random_split(datasets, [0.7, 0.1, 0.2])
train_loader = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
val_loader = data.DataLoader(val_set, args.batch_size, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_set, args.batch_size, shuffle=True, drop_last=True)


model = Transformer(args.d_model, args.d_ff, args.num_heads, args.n_layers, args.s_voca_len, args.t_voca_len,
                    args.s_max_len+2, args.t_max_len+2, args.prob, args.pad_idx, args.pad_idx, device).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_idx).to(device)
optimizer = Adam(params=model.parameters(), lr=1e-4)
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

output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)

for epoch in range(start_epoch, start_epoch + args.n_epochs):
    output, train_loss = train(model, loss_fn, optimizer, train_loader, device)
    valid_loss, val_ppl, val_bleu = validation(model, loss_fn, val_loader, device)

    print(f'Epoch: {epoch} | Train loss: {train_loss} | Valid loss: {valid_loss} | Valid PPL: {val_ppl} | Valid BLEU: {val_bleu}')

    if valid_loss < best_valid_loss:
        best_model = model
        best_valid_loss = valid_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss},
                   f'{output_path}/{epoch}-model.pt')
        print("Model Saved")
print('end train')

# ===== evaluate =====
print('evaluate')
test_bleu = evaluate(best_model, test_loader, device)
print(f"BLEU: {test_bleu}")
