import torch
from sklearn.metrics import f1_score, accuracy_score


def evaluate(model, test_loader):
    model.eval()
    f1, acc = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            texts = data['text']
            y_true = data['label']
            output = model(texts)

            _, y_pred = torch.max(output, 1)

            f1 += f1_score(y_true, y_pred)
            acc += accuracy_score(y_true, y_pred)

    return f1/len(test_loader), acc/len(test_loader)
