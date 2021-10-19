import torch


def predict_class(y_hat, true_threshold=0.5):
    pred_soft = torch.exp(y_hat)
    pred = (pred_soft[..., 1] > true_threshold).long()
    return pred


def predict_class_multi(y_hat):
    pred_soft = torch.exp(y_hat)
    pred_max = torch.argmax(pred_soft, axis=-1)
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


def get_accuracy_multi(y, pred):
    b, c = y.shape
    correct = (y == pred).type(torch.float32)
    total_correct = torch.sum(correct, dim=1)
    percent_corrrect = (total_correct == c).count_nonzero().item() / b
    return percent_corrrect


def count_succ_and_errs(y, pred, return_indices=False):

    true_positive_i = torch.logical_and(pred == 1, y == 1)
    true_negative_i = torch.logical_and(pred == 0, y == 0)
    false_positive_i = torch.logical_and(pred == 1, y == 0)
    false_negative_i = torch.logical_and(pred == 0, y == 1)

    accs = {'true_positive': true_positive_i.count_nonzero().item(),
            'true_negative': true_negative_i.count_nonzero().item(),
            'false_positive': false_positive_i.count_nonzero().item(),
            'false_negative': false_negative_i.count_nonzero().item(),
            }
    indices = {'true_positive': true_positive_i,
               'true_negative': true_negative_i,
               'false_positive': false_positive_i,
               'false_negative': false_negative_i,
               }
    if not return_indices:
        return accs
    if return_indices:
        return accs, indices


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


def average_batch(batch):

    num_batches = len(batch.keys())
    all_acc = {}
    for accs in batch.values():
        all_acc = {k: (accs.get(k, 0)/accs['count']) + all_acc.get(k, 0)
                   for k in set(accs)}

    return {k: all_acc[k]/num_batches for k in set(all_acc)}


def max_canidate_prediction(y_hats, ys, bids, threshold=0.0):
    """
    we predict by looking at all the positive probabilites
    for a given set of canidates and only predicting 1 for the
    max probabilty across the set of canidates. This forces only
    one merge per neurite but also possibly reduces merges errors
    """
    uids = torch.unique(bids)
    total_acc = {}
    for uid in uids:
        idxs = bids == uid
        y, y_hat = ys[idxs], y_hats[idxs]
        y_hat_true = y_hat[:, 1]
        max_true = torch.max(y_hat_true)
        if max_true > threshold:
            pred = (max_true == y_hat_true).long()
        else:
            pred = torch.zeros_like(y_hat_true)

        acc = count_succ_and_errs(y, pred)
        total_acc = {k: acc.get(k, 0) + total_acc.get(k, 0) for k in set(acc)}

    return total_acc
