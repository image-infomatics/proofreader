import torch


def get_tag(t, p):
    pre = 'false'
    post = 'correct'
    if t == 0:
        pre = 'true'
    if p != t:
        post = 'wrong'
    return f'{pre}_{post}'


def predict_class_sigmoid(y_hat, threshold=0.5):
    return (y_hat > threshold).int()


def predict_class(y_hat):
    pred_soft = torch.exp(y_hat)
    pred_max = torch.argmax(pred_soft, axis=1)
    return pred_max


def get_accuracy(y, pred):
    correct = (y == pred).type(torch.float32)
    total_acc = torch.mean(correct)
    true_acc = torch.mean(correct[y == 1])
    false_acc = torch.mean(correct[y == 0])
    return {'total_acc': total_acc.item(), 'true_acc': true_acc.item(), 'false_acc': false_acc.item()}


def accumulate_accuracy_per_batch(totals, y, pred, bid):
    ids = torch.unique(bid)
    for i in ids:
        i = i.item()
        inx = bid == i
        acc = get_accuracy(y[inx], pred[inx])
        count = inx.count_nonzero().item()
        d = {'count': count}
        for k, v in acc.items():
            d[k] = v * count
        if i in totals:
            totals[i] = {k: totals[i].get(k, 0) + d.get(k, 0)
                         for k in set(d)}
        else:
            totals[i] = d

    return totals


def average_batch(batch):

    num_batches = len(batch.keys())
    all_acc = {}
    for accs in batch.values():
        all_acc = {k: (accs.get(k, 0)/accs['count']) + all_acc.get(k, 0)
                   for k in set(accs)}

    return {k: all_acc[k]/num_batches for k in set(all_acc)}
