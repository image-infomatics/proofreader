import torch


def get_tag(t, p):
    pre = 'false'
    post = 'correct'
    if t == 0:
        pre = 'true'
    if p != t:
        post = 'wrong'
    return f'{pre}_{post}'


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
