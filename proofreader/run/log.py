import random
import matplotlib.pyplot as plt
from proofreader.run.metrics import voi, adapted_rand
from proofreader.utils.vis import color_segmentation
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import numpy as np
import networkx as nx
from skimage.segmentation import find_boundaries

pr_config = {'xlabel': 'Error', 'ylabel': 'Success',
             'title': 'Merge Error and Success for Thresholds',
             'xlim': [0.0, 0.65], 'ylim': [0.5, 1.0]}

voi_config = {'xlabel': 'Threshold', 'ylabel': 'Voi',
              'title': 'Voi per Threshold',
              'xlim': [0.0, 1.1]}


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


def build_curve_fig(x, y, labels=None, rm_last=False, config={}):

    if rm_last:
        x = x[:-1]
        y = y[:-1]
        if labels is not None:
            labels = labels[:-1]

    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, y, zorder=1)
    plt.scatter(x, y, c=labels, s=32, edgecolors='k', zorder=2)
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

    plt.xticks(np.arange(min(x), max(x), 0.1))
    plt.grid()
    # label
    if labels is not None:
        for i in range(len(labels)):
            plt.annotate(f"{round(labels[i],3)}", (x[i], y[i]), textcoords="offset points",  xytext=(
                20, 4),  ha='center')

    return fig


def plot_merge_curve(accs, thresholds, writer, dataset_stats, tag, dglobal, epoch):

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

    # write global data
    if dglobal is not None:
        dglobal[tag].append((succ, err))

    # merge curve
    writer.add_pr_curve_raw(tag, tp, fp, tn, fn,
                            succ, err, global_step=epoch, num_thresholds=len(thresholds))

    # merge curve plt figures
    fig = build_curve_fig(err, succ, labels=thresholds,
                          rm_last=True, config=pr_config)
    writer.add_figure(f'{tag}_plt', fig,
                      global_step=epoch)


def plot_voi_curve(vols, infos, y_hats, thresholds, num_slices, writer, dglobal, epoch):
    print('computing voi curve...')

    # multithreaded
    args = [(vols, infos, y_hats, t, num_slices) for t in thresholds]
    with ThreadPool(processes=len(thresholds)) as pool:
        vois_arr = pool.map(do_voi, args)

    # aggregate
    total, split, merge, arand, borders = [
        [t[i] for t in vois_arr] for i in range(len(vois_arr[0]))]

    # log voi metrics
    for (data, title) in zip([split, merge, total, arand], ['Voi Split', 'Voi Merge', 'Voi Total', 'Arand']):
        voi_config['title'] = f'{title} per Threshold'
        fig = build_curve_fig(thresholds, data, labels=data, config=voi_config)
        writer.add_figure(f'{title}_Curve', fig, global_step=epoch)
        # write global data
        dglobal[title].append((thresholds, data))

    # log border examples
    for t, border in zip(thresholds, borders):
        colored = color_segmentation(
            border, colors=['tomato', 'forestgreen', 'cornflowerblue'])
        writer.add_image(f'border_seg/{t}', colored,
                         global_step=epoch, dataformats='HWC')

    print('finished voi curve!')


def do_voi(args):
    vols, infos, y_hats, threshold, num_slices = args

    # first init for all vols and drops
    grouped = defaultdict(dict)  # 'vol_i' -> 'drop_start' -> [infos]
    for info in infos:
        grouped[info['volume_i']][info['drop_start']] = []
    # select infos according to y_hats and threshold
    true_infos = infos[y_hats[:, 1] > threshold]
    # then add for just true infos
    for info in true_infos:
        grouped[info['volume_i']][info['drop_start']].append(info)

    # get total voi
    split_total, merge_total, arand_total = 0, 0, 0
    num_drops = 0
    borders = []
    for (vol_i, drops) in grouped.items():
        num_drops += len(drops)
        # multithreaded
        args = [(vols[vol_i], drop_start, drop_start+num_slices, true_infos)
                for drop_start, true_infos in drops.items()]
        with ThreadPool(processes=len(drops)) as pool:
            vois_arr = pool.map(compute_voi_for_drop, args)
        # aggregate
        for (split, merge, arand, border) in vois_arr:
            split_total += split
            merge_total += merge
            arand_total += arand
            borders.append(border)

    (avg_split, avg_merge, avg_arand) = split_total / \
        num_drops, merge_total/num_drops, arand_total/num_drops

    ret_border = random.choice(borders)
    return (avg_merge+avg_split, avg_split, avg_merge, avg_arand, ret_border)


def compute_voi_for_drop(args):
    vol, drop_start, drop_end, true_infos = args
    max_l = int(np.max(vol))*4
    wrong, right, neither, bound = max_l, max_l+1, max_l+2, 0

    # create gt segmentation and split segmentation
    gt = vol.copy()  # avoid label collision with wrong, right, none
    gt[drop_start:drop_end] = 0  # drop slices from gt
    zeros = gt == 0  # retain zero mask
    z_btop = gt[drop_start - 1] == 0
    z_bbot = gt[drop_end] == 0  # zeros for border
    split = gt.copy()
    offset = int(np.max(gt))+1  # relabel bot section for split
    split[drop_end:] += offset

    merges = []
    for info in true_infos:
        # get merge classes in relabel bot section
        top_c = info['top_class']
        bot_c = info['bot_class'] + offset
        merges.append([top_c, bot_c])

    # add top/bot slice to view post intervention segmentation
    blen = split[drop_end].shape[0]
    border = np.concatenate([split[drop_start-1], split[drop_end]])
    border_boundries = find_boundaries(border, mode='thick')

    # combine all merges, any merge which have a nonempty intersection should be merged
    # basically like graph connected components
    if len(merges) > 0:
        merges = connected_components(merges)
        new_label = int(np.max(split))
        for m in merges:
            new_label += 1
            mask = np.isin(split, m)
            split[mask] = new_label

            # to build image to view post intervention segmentation
            num_gt = len(np.unique(gt[mask]))
            is_error_in_merge = num_gt > 1
            bmask = np.isin(border, m)
            if is_error_in_merge:
                border[bmask] = wrong
            else:
                border[bmask] = right

    # mark where no attempt was made
    no_mask = ~(np.isin(border, [wrong, right]))
    border[no_mask] = neither
    # add prev boundries
    border[border_boundries] = bound
    border[:blen][z_btop] = bound
    border[blen:][z_bbot] = bound
    # hack to ensure coloring happens correctly
    border[0, 0] = wrong
    border[0, 1] = right
    border[0, 2] = neither

    # add back previous zeros
    split[zeros] = 0
    # measure voi and arand after intervention
    (split_post, merge_post) = voi(split, gt)
    arand = adapted_rand(split, gt)

    return (split_post, merge_post, arand, border)


def connected_components(L):
    G = nx.Graph()
    for l in L:
        nx.add_path(G, l)
    res = list(nx.connected_components(G))
    res = [list(a) for a in res]
    return res
