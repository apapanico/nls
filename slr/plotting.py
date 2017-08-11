import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pylab import setp
from utils import *


 
def line_plot_eigs(simResult,T,N,estimators,qtile,simulations,save=False):
    
    true_eigs = annualize_vol(simResult[1]/N)
    
    # create a set of subplots and set its parameters
    
    fig_w = 15
    fig_h = 6
    
    
    fig,ar = plt.subplots(nrows=1, ncols=len(estimators),
                          sharex=True,sharey=True,
                          figsize=(fig_w, fig_h), dpi=100,squeeze=False)
    
    
    fig.suptitle('N='+ str(N) + 
                 ', T=' + str(T) + 
                 ', simulations=' + str(simulations),
                 fontsize=14, color='black')
    
    
    fig.text(0.04, 0.5, 'Ann.Volatility(%)', 
             va='center', rotation='vertical')
    
    # iterate over every estimator 
        
    for i in estimators:
        inq = read_inquiry((i,T,N))
        if inq not in simResult[0].keys():
            raise Exception('No simulations run for '+inq+' estimator')
        if simulations!=1:
            raise Exception('Number of simulations for '+inq+' estimator has to be 1')
        else:
            est_eigs  = simResult[0][inq][0]
            est_eigs  = annualize_vol(est_eigs/N)
        #@if
        
        # choose eigenvalues on qtile quantiles
        xcoord      = np.rint(np.linspace(0,N-1,qtile)).astype(int)
        
        # plotting part
        # every estimator should fill a subplot
        
        p = estimators.index(i)
        ax = ar[0,p]
        
        bamboo = "#DC5C05"
        orange = "#FF9000"

        
        # draw a line with true eigenvalues
        
        ax.plot(list(range(1,qtile+1)),
                 true_eigs[xcoord],
                 color = 'C0',
                 marker="o",
                 markerfacecolor=bamboo, 
                 markeredgecolor=bamboo,
                 markersize=3,
                 zorder=2)
        
        # draw a line with estimated eigenvalues 
        
        ax.plot(list(range(1,qtile+1)),
                 est_eigs[xcoord],
                 color='grey',
                 marker="o",
                 markerfacecolor=orange, 
                 markeredgecolor=orange,
                 markersize=3,
                 zorder=2)
        
        # draw titles, axes and etc.
        
        
        # draw titles, axes and etc.
        
        ax.set_title(i ,y=1.01,
                  fontsize=10, color='black')
        
        ax.set_xticks(list(range(1,qtile+1)))
        ax.set_xticklabels(['${\\hat{\lambda}_{'+ str(i) + '}}$' for i in xcoord+1],
                            fontsize='small')
                
        # show the difference between true eigenvalues and centers of boxplots
     
        ax.fill_between(list(range(1,qtile+1)), 
                         est_eigs[xcoord], 
                         true_eigs[xcoord], 
                         color='grey', 
                         alpha='0.5')
        # draw legend
        
        vleg = mlines.Line2D([], [],
                     markeredgecolor=bamboo,
                     markerfacecolor=bamboo,
                     marker='o',
                     linestyle='-',
                     markersize=4,
                     label='$\\lambda(\Sigma)$')
       
        mleg = mlines.Line2D([], [],
                             color='grey',
                             alpha=0.5,
                             markeredgecolor='None',
                             marker='s',
                             linestyle='None',
                             markersize=10,
                             label='Bias')

        ax.legend(handles=[vleg,mleg],
                   numpoints=1, fancybox=True, framealpha=0.25)
        
        # draw bias 
        
        bias = abs_dif(true_eigs,est_eigs)
        bias = round(bias,1)
        ax.text(1, 5, 'Bias= ' + str(bias),
                bbox=dict(facecolor='grey', alpha=0.5))

    #save the result
    if save:
        plt.savefig('N='+ str(N) + 
                    ' T=' + str(T) + 
                    ' simulations=' + str(simulations)+
                    '.jpg',dpi=200)
    
    plt.show()  
    
    #@for
    
#@def



def box_plot_eigs(simResult,T,N,estimators,qtile,simulations,save=False):
    
    # function plots an inquiry of the form (estimator,T,N) with all nonzero elements
    # it compares the distribution of each eigenvalue with the true eigenvalue
    
    # simresult  = output from simulate_eigs
    # estimators = estimator used to find the covariance matrix 
    # T,N        = number of samples and number of stocks
    # qtile      = quantile of eigenvalues which to display (for visualization purposes)
    
    # convert daily variances into annualized std in % 
    
    true_eigs = annualize_vol(simResult[1]/N)
    
    # create a set of subplots and set its parameters
    
    
    fig_w = 15
    fig_h = 6
    
    
    fig,ar = plt.subplots(nrows=1, ncols=len(estimators),
                          sharex=True,sharey=True,
                          figsize=(fig_w, fig_h), dpi=100,squeeze=False)
    
    
    fig.suptitle('N='+ str(N) + 
                 ', T=' + str(T) + 
                 ', simulations=' + str(simulations),
                 fontsize=14, color='black')
    
    
    fig.text(0.04, 0.5, 'Ann.Volatility(%)', 
             va='center', rotation='vertical')
    
    # iterate over every estimator 
    
    for i in estimators:
        inq = read_inquiry((i,T,N))
        if inq not in simResult[0].keys():
            raise Exception('No simulations run for '+inq+' estimator')
        else:
            est_eigs  = np.vstack(simResult[0][inq])
            est_eigs  = annualize_vol(est_eigs/N)
        #@if
        
        # choose eigenvalues on qtile quantiles
        xcoord      = np.rint(np.linspace(0,N-1,qtile)).astype(int)
        box_medians = np.median(est_eigs,0)
                
        # plotting part
        # every estimator should fill a subplot
        
        p = estimators.index(i)
        ax = ar[0,p]
        
        violet = "#462066"
        bamboo = "#DC5C05"
        orange = "#FF9000"
        
        lw = 1.5
    
        #plt.subplot(1,3,estimators.index(i)+1) 
    
        bp = ax.boxplot(est_eigs[:,xcoord],
                    notch=True, 
                    widths=np.repeat(0.2,len(xcoord)).tolist(),
                    patch_artist=True,
                    showfliers=False,
                    zorder=1)
        
        # change boxplots
        
        for j in range(qtile):
            setp(bp['boxes'][j], edgecolor=violet, fill=False, linewidth=lw)
            setp(bp['medians'][j], color=orange, linewidth=lw)
            setp(bp['whiskers'][j], color=violet, linewidth=lw,linestyle='--')
            setp(bp['whiskers'][j+qtile], color=violet, linewidth=lw,linestyle='--')
            setp(bp['caps'][j], color=violet, linewidth=lw)
            setp(bp['caps'][j+qtile], color=violet, linewidth=lw)
            
        #@for
            
        # draw a line with true eigenvalues
        
        ax.plot(list(range(1,qtile+1)),
                 true_eigs[xcoord],
                 marker="o",
                 markerfacecolor=bamboo, 
                 markeredgecolor=bamboo,
                 markersize=3,
                 zorder=2)
        
        # draw titles, axes and etc.
        
        ax.set_title(i ,y=1.01,
                  fontsize=10, color='black')
        
        ax.set_xticks(list(range(1,qtile+1)))
        ax.set_xticklabels(['${\\hat{\lambda}_{'+ str(i) + '}}$' for i in xcoord+1],
                            fontsize='small')
        
        
        # show the difference between true eigenvalues and centers of boxplots
     
        ax.fill_between(list(range(1,qtile+1)), 
                         box_medians[xcoord], 
                         true_eigs[xcoord], 
                         color='grey', 
                         alpha='0.5')
        # draw legend
        
        vleg = mlines.Line2D([], [],
                     markeredgecolor=bamboo,
                     markerfacecolor=bamboo,
                     marker='o',
                     linestyle='-',
                     markersize=4,
                     label='$\\lambda(\Sigma)$')
       
        mleg = mlines.Line2D([], [],
                             color='grey',
                             alpha=0.5,
                             markeredgecolor='None',
                             marker='s',
                             linestyle='None',
                             markersize=10,
                             label='Bias')

        ax.legend(handles=[vleg,mleg],
                   numpoints=1, fancybox=True, framealpha=0.25)
        
        # draw bias 
        bias = abs_dif(true_eigs,box_medians)
        bias = round(bias,1)
        ax.text(1, 5, 'Bias= ' + str(bias),
                bbox=dict(facecolor='grey', alpha=0.5))

    
    #save the result
    if save:
        plt.savefig('N='+ str(N) + 
                    ' T=' + str(T) + 
                    ' simulations=' + str(simulations)+
                    '.jpg',dpi=200)
    
    plt.show()
    
    #@for
    
#@def

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
