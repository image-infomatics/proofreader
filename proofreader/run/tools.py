import numpy as np
from multiprocessing.pool import ThreadPool
from collections import defaultdict
from proofreader.utils.voi import voi
import matplotlib.pyplot as plt

pr_config = {'xlabel': 'Error', 'ylabel': 'Success',
             'title': 'Merge Error and Success for Thresholds',
             'xlim': [0.0, 0.65], 'ylim': [0.5, 1.0]}

voi_config = {'xlabel': 'Threshold', 'ylabel': 'Voi',
              'title': 'Voi per Threshold',
              'xlim': [0.0, 1.0]}


def build_mesh_figs(data, title='', marker='.', lim=None):
    figs = []
    for mesh in data:
        mesh = mesh.cpu()
        x, y, z = mesh[:, 0], mesh[:, 1], mesh[:, 2]
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', title=title)

        if lim is not None:
            ax.set_xlim3d(-lim, lim)
            ax.set_ylim3d(-lim, lim)
            ax.set_zlim3d(-lim, lim)

        ax.scatter(x, y, z, marker=marker)
        figs.append(fig)

    return figs


def build_curve_fig(x, y, thresholds=None, rm_last=False, config={}):

    if rm_last:
        x = x[:-1]
        y = y[:-1]
        thresholds = thresholds[:-1]

    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, y, zorder=1)
    plt.scatter(x, y, c=thresholds, s=26, edgecolors='k', zorder=2)
    if 'xlabel' in config:
        plt.xlabel(config['xlabel'])
    if 'ylabel' in config:
        plt.ylabel(config['ylabel'])
    if 'title' in config:
        plt.title(config['title'])
    if 'xlim' in config:
        plt.xlim(config['xlim'])
    if 'ylim' in config:
        plt.ylim(config['ylim'])

    plt.grid()
    # label
    if thresholds is not None:
        for i in range(len(thresholds)):
            plt.annotate(f"{thresholds[i]}", (x[i], y[i]), textcoords="offset points",  xytext=(
                20, -5),  ha='center')

    return fig


def plot_merge_curve(accs, thresholds, writer, dataset_stats, tag, epoch):

    total_neurites = dataset_stats['total_neurites']
    merge_opps = dataset_stats['merge_opportunities']
    # normal curve
    succ, err, tp, fp, tn, fn = [], [], [], [], [], []
    for t in thresholds:
        t_acc = accs[t]
        succ.append(
            t_acc['true_positive'] / merge_opps)
        err.append(
            t_acc['false_positive'] / total_neurites)

        tp.append(t_acc['true_positive'])
        fp.append(t_acc['false_positive'])
        tn.append(t_acc['true_negative'])
        fn.append(t_acc['false_negative'])

    # merge curve
    writer.add_pr_curve_raw(tag, tp, fp, tn, fn,
                            succ, err, global_step=epoch, num_thresholds=len(thresholds))

    # merge curve plt figures
    fig = build_curve_fig(err, succ, thresholds=thresholds,
                          rm_last=True, config=pr_config)
    writer.add_figure(f'{tag}_plt', fig,
                      global_step=epoch)


def plot_voi_curve(vols, infos, y_hats, thresholds, writer, epoch):
    print('computing voi curve...')
    total, merge, split = [], [], []

    # multithreaded
    args = [(vols, infos, y_hats, t) for t in thresholds]
    with ThreadPool(processes=len(thresholds)) as pool:
        vois_arr = pool.map(do_voi, args)

    # aggregate
    for (avg_split_del, avg_merge_del) in vois_arr:
        split.append(avg_split_del)
        merge.append(avg_merge_del)
        total.append(avg_split_del + avg_merge_del)

# Voi Split
    voi_config['title'] = 'ΔVoi Split per Threshold'
    fig = build_curve_fig(
        thresholds, split, config=voi_config)
    writer.add_figure(f'voi_curve_split', fig,
                      global_step=epoch)
    # Voi Merge
    voi_config['title'] = 'ΔVoi Merge per Threshold'
    fig = build_curve_fig(
        thresholds, merge, config=voi_config)
    writer.add_figure(f'voi_curve_merge', fig,
                      global_step=epoch)
    # Voi Total
    voi_config['title'] = 'ΔVoi Total per Threshold'
    fig = build_curve_fig(
        thresholds, total, config=voi_config)
    writer.add_figure(f'voi_curve_total', fig,
                      global_step=epoch)
    print('finished voi curve!')


def do_voi(args):
    vols, infos, y_hats, threshold = args
    y_hats = y_hats.cpu().numpy()
    # select infos according to y_hats and threshold
    true_infos = infos[y_hats[:, 1] > threshold]

    if len(true_infos) <= 0:
        return 0, 0

    # group infos into dict: vol_i -> drop_start -> [infos]
    grouped = defaultdict(lambda: defaultdict(list))
    for info in true_infos:
        vol_i, drop_start = info['volume_i'], info['drop_start']
        grouped[vol_i][drop_start].append(info)

    # get total voi
    split_del, merge_del = 0, 0
    num_drops = 0
    for (vol_i, drops) in grouped.items():
        num_drops += len(drops)
        # multithreaded
        args = [(vols[vol_i], drop_start, true_infos)
                for drop_start, true_infos in drops.items()]
        with ThreadPool(processes=len(drops)) as pool:
            vois_arr = pool.map(compute_voi_for_drop, args)
        # aggregate
        for (split, merge) in vois_arr:
            split_del += split
            merge_del += merge

    (avg_split_del, avg_merge_del) = split_del/num_drops, merge_del/num_drops
    return (avg_split_del, avg_merge_del)


def compute_voi_for_drop(args):
    gt, drop_start, true_infos = args
    drop_end = true_infos[0]['drop_end']

    # create gt segmentation and split segmentation
    temp = gt[drop_start:drop_end]  # store drop slices to reset later
    gt[drop_start:drop_end] = 0  # drop slices from gt
    split = gt.copy()
    offset = int(np.max(gt))+1  # relabel bot section for split
    split[drop_end:] += offset

    # measure voi before intervention
    (split_pre, merge_pre) = voi(split, gt)

    for info in true_infos:
        # find classes merge neurites
        top_c = info['top_class']
        # corresponding class in relabel bot section
        bot_c = info['bot_class'] + offset
        split[split == bot_c] = top_c  # perform merge on split segmentation

    # measure voi after intervention
    (split_post, merge_post) = voi(split, gt)
    # get change in voi
    split_del, merge_del = split_post - split_pre, merge_post - merge_pre

    # reset dropped slice so vol is unaltered
    gt[drop_start:drop_end] = temp

    return (split_del, merge_del)
