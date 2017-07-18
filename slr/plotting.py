import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pylab import setp


def make_eig_box_plots(fm, n_sims, signature, t, sim_eigs, true_eigs,
                       title, outputname, ylims=()):

    id_str = ''.join('{}{}'.format(key, val)
                     for key, val in sorted(signature.items()))

    fig_w = 7
    fig_h = 6

    plt.figure(id_str, figsize=(fig_w, fig_h), dpi=100)

    bp = plt.boxplot([sim_eigs[t[0]], sim_eigs[t[1]]],
                     notch=True, widths=[0.2, 0.2],
                     patch_artist=True)

    violet = "#462066"
    deluge = "#7C71AD"
    bamboo = "#DC5C05"
    orange = "#FF9000"
    # oyster = '#978B7D'
    # yellow = '#FFAC00'
    downy = "#6EC5B8"
    # coral = '#C7BAA7'

    lw = 2.75

    for i in range(2):
        setp(bp['fliers'][i], marker='o',
             markerfacecolor=bamboo, markeredgecolor=bamboo)

    setp(bp['boxes'][0], edgecolor=violet, fill=False, linewidth=lw)
    setp(bp['medians'][0], color=violet, linewidth=lw)
    setp(bp['whiskers'][0], color=violet, linewidth=lw)
    setp(bp['whiskers'][1], color=violet, linewidth=lw)
    setp(bp['caps'][0], color=violet, linewidth=lw)
    setp(bp['caps'][1], color=violet, linewidth=lw)

    setp(bp['boxes'][1], edgecolor=violet, fill=False, linewidth=lw)
    setp(bp['medians'][1], color=violet, linewidth=lw)
    setp(bp['whiskers'][2], color=violet, linewidth=lw)
    setp(bp['whiskers'][3], color=violet, linewidth=lw)
    setp(bp['caps'][2], color=violet, linewidth=lw)
    setp(bp['caps'][3], color=violet, linewidth=lw)

    for x in [1, 2]:
        plt.plot([x], [true_eigs[0]],
                 linestyle='None',
                 color=deluge,
                 markeredgecolor=deluge,
                 marker="s",
                 markersize=9)

        plt.plot([x], [true_eigs[1]],
                 linestyle='None',
                 color=orange,
                 markeredgecolor=downy,
                 marker="o",
                 markersize=10)

        plt.plot([x], [true_eigs[2]],
                 linestyle='None',
                 color=downy,
                 markeredgecolor=downy,
                 marker="^",
                 markersize=11)

    # legend

    vleg = mlines.Line2D([], [],
                         color=deluge,
                         markeredgecolor=deluge,
                         marker='s',
                         linestyle='None',
                         markersize=9,
                         label='$\\lambda_1 (\Sigma)$')

    mleg = mlines.Line2D([], [],
                         color=orange,
                         markeredgecolor=downy,
                         marker='o',
                         linestyle='None',
                         markersize=10,
                         label='$\\lambda_1 (L+S)$')

    lleg = mlines.Line2D([], [],
                         color=downy,
                         markeredgecolor=downy,
                         marker='^',
                         linestyle='None',
                         markersize=11,
                         label='${\\lambda_1 (L)}$')

    plt.legend(handles=[vleg, mleg, lleg],
               numpoints=1, fancybox=True, framealpha=0.25)

    if not ylims:
        ymin = np.min((sim_eigs[t[0]], sim_eigs[t[1]]))
        ymin = 0.95 * np.min(true_eigs + (ymin,))
        ymax = np.max((sim_eigs[t[0]], sim_eigs[t[1]]))
        ymax = 1.05 * np.max(true_eigs + (ymax,))
        ylims = (ymin, ymax)

    plt.ylim(ylims)

    plt.ylabel('Ann. Volatility (%)',
               fontsize=16, color='black')
    plt.yticks(fontsize=13)
    plt.xticks(
        [1, 2],
        ['T = {}'.format(t[0]), 'T = {}'.format(t[1])],
        y=-0.01, fontsize=16, color='black'
    )

    # title from input
    plt.title(title, y=1.01, fontsize=18, color='black')

    # plt.title(title)
    fname = "_".join([outputname, id_str])
    out = os.path.join("img", fname)
    plt.savefig(out, transparent=True, dpi=256)
    plt.show(block=False)


def make_var_ratio_box_plots(fm, n_sims, signature, t, var_ratios,
                             title, outputname, ylims=()):
    # make the plots

    deluge = "#7C71AD"
    bamboo = "#DC5C05"
    orange = "#FF9000"
    oyster = '#978B7D'
    # yellow = '#FFAC00'
    downy = "#6EC5B8"
    coral = '#C7BAA7'

    fig_w = 7
    fig_h = 6

    id_str = ''.join('{}{}'.format(key, val)
                     for key, val in sorted(signature.items()))

    plt.figure(id_str, figsize=(fig_w, fig_h), dpi=100)

    # multiple box plots on one figure
    bp = plt.boxplot(
        [var_ratios['C'], var_ratios['L+S'], var_ratios['D']],
        notch=True,
        widths=[0.2, 0.2, 0.2],
        patch_artist=True
    )

    lw = 2.75

    for i in range(3):
        setp(bp['fliers'][i], marker='o',
             markerfacecolor=bamboo, markeredgecolor=bamboo)

    setp(bp['boxes'][0], facecolor=deluge,
         edgecolor=downy, linewidth=lw)
    setp(bp['medians'][0], color=downy, linewidth=4)
    setp(bp['whiskers'][0], color=deluge, linewidth=lw)
    setp(bp['whiskers'][1], color=deluge, linewidth=lw)
    setp(bp['caps'][0], color=deluge, linewidth=lw)
    setp(bp['caps'][1], color=deluge, linewidth=lw)

    setp(bp['boxes'][1], edgecolor=downy,
         facecolor=orange, linewidth=lw)
    setp(bp['medians'][1], color=downy, linewidth=4)
    setp(bp['whiskers'][2], color=orange, linewidth=lw)
    setp(bp['whiskers'][3], color=orange, linewidth=lw)
    setp(bp['caps'][2], color=orange, linewidth=lw)
    setp(bp['caps'][3], color=orange, linewidth=lw)

    setp(bp['boxes'][2], edgecolor=orange,
         facecolor=oyster, linewidth=lw)
    setp(bp['medians'][2], color=orange, linewidth=4)
    setp(bp['whiskers'][4], color=oyster, linewidth=lw)
    setp(bp['whiskers'][5], color=oyster, linewidth=lw)
    setp(bp['caps'][5], color=oyster, linewidth=lw)
    setp(bp['caps'][5], color=oyster, linewidth=lw)

    plt.ylabel('Forecast Ratio ' +
               '$\\mathscr{R}_{\cdot} ( w \\,|\\, \\theta)$',
               fontsize=16, color='black')
    plt.yticks(fontsize=13)
    plt.xticks([1, 2, 3], ['Portfolio $(\\Sigma)$',
                           'Factor $(S+L)$', 'Specific $(\\Delta)$'],
               y=-0.01, fontsize=16, color='black')

    plt.plot([0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0],
             linestyle=':',
             color=coral,
             linewidth=2)

    if ylims:
        plt.ylim(ylims)

    plt.title(title, y=1.01, fontsize=18, color='black')

    # plt.title(title)
    fname = "_".join([outputname, id_str])
    out = os.path.join("img", fname)
    plt.savefig(out, transparent=True, dpi=256)
    plt.show(block=False)
