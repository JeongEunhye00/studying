def train(model, loss_fn, optimizer, train_loader):
    model.train()

    cur_loss = 0

    for i, data in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        texts = data['text']
        y_true = data['label']

        outputs = model(texts)

        loss = loss_fn(outputs, y_true)
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()

    return outputs, cur_loss/(len(train_loader))
