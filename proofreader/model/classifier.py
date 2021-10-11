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


def get_accuracy(y, pred, ret_perfect=False):
    correct = (y == pred).type(torch.float32)
    total_acc = torch.mean(correct)
    true_acc = torch.mean(correct[y == 1])
    false_acc = torch.mean(correct[y == 0])
    accs = {'total_acc': total_acc.item(), 'true_acc': true_acc.item(),
            'false_acc': false_acc.item()}
    if ret_perfect:
        accs['perfect'] = int(total_acc.item() == 1.0)
    return accs


def get_accuracy_sums(y, pred):
    correct = (y == pred).type(torch.float32)
    total_acc = torch.sum(correct)
    true_acc = torch.sum(correct[y == 1])
    false_acc = torch.sum(correct[y == 0])
    accs = {'total_acc': total_acc.item(), 'true_acc': true_acc.item(
    ), 'false_acc': false_acc.item(), 'false_count': len(correct[y == 0]), 'true_count': len(correct[y == 1])}

    return accs


def average_accuracy_sums(all_accs):
    avg_acc = {
        'total_acc': all_accs['total_acc'] / (all_accs['false_count'] + all_accs['true_count']),
        'true_acc': all_accs['true_acc'] / (all_accs['true_count']),
        'false_acc': all_accs['false_acc'] / (all_accs['false_count']),
    }

    return avg_acc


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
