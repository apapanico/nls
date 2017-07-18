import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool as CPU
from pylab import setp
from itertools import repeat, starmap
from importlib import reload

import factor
import ADMM
# reload any changes to code
reload (factor)
reload (ADMM)

from itertools import repeat
from factor import fm_
from ADMM import SLR
from utils import nostdout

# computes the top eigenvalues
def eigen (fm, T, seed=1):

    N = fm.N
    R = fm.R (T=T, seed=seed)

    if (T > N):
        W = R.T.dot (R) / T # N x N matrix
    else:
        W = R.dot (R.T) / T # T x T matrix
    
    val = eigsh (W, 1, which='LA',
        maxiter = int (1e9),
        return_eigenvectors=False)[0]
    
    return val
#@ def


def eigen_slr (fm, T, lam, seed=1):

    N = fm.N
    S, L, D = fm.covariances()
    
    # genetate security returns
    R = fm.R (T=T, seed=seed)
    # sample covariance matrix
    W = R.T.dot (R) / T

    ## use initial values for L0 and S0 to speed up
    with nostdout():
        st = SLR (W, lam = lam, gam = 6/N)
    #L0=L, S0=S+D, rit = 1000,
   
    val = eigsh (st['L'], 1, which='LA',
        maxiter=int (1e9),
        return_eigenvectors=False)[0]
    
    return val
#@ def


# LEFT PANEL

# number of simulation trials to run
nt = 100
cpus_ = 4
N = 128 

# factor model
signature = dict (N=N, K0 = 4, K1=32, K2=16, seed=1)
fm = fm_(**signature)

# pool = CPU (processes=cpus_)

# SLR paramaters
lam1 = 1e-3 * np.sqrt (N/250)
lam2 = lam1*np.sqrt (250/1000)

# seed values for sample covariance matrices
seeds = range (1,1+nt)

# T = 250 and T = 1000 experiments
X1 = zip (repeat (fm),repeat (250),repeat (lam1),seeds)
X2 = zip (repeat (fm),repeat (1000),repeat (lam2),seeds)

# run forcast nt times using 4 threads
# WR1 = pool.starmap (eigen_slr, X1)
# WR2 = pool.starmap (eigen_slr, X2)
WR1 = list(starmap (eigen_slr, X1))
WR2 = list(starmap (eigen_slr, X2))

# pool.close()
# pool.join()

S, L, D = fm.covariances()
V = S + L + D
    
V_val = eigsh (V, 1, which='LA', 
    maxiter=int (1e9),
    return_eigenvectors=False)[0]
L_val = eigsh (L, 1, which='LA', 
    maxiter=int (1e9),
    return_eigenvectors=False)[0]

# rescale to ann. vols. %

WR1 = np.sqrt (np.asarray (WR1) / fm.N * 256) * 100
WR2 = np.sqrt (np.asarray (WR2) / fm.N * 256) * 100
L_val = np.sqrt (L_val / fm.N * 256) * 100
V_val = np.sqrt (V_val / fm.N * 256) * 100

# make the plots

id_str = ''.join('{}{}'.format(key, val) 
    for key, val in sorted(signature.items()))

fig_w = 7
fig_h = 6

plt.figure ('SLR' + id_str, figsize=(fig_w,fig_h), dpi=100)

bp = plt.boxplot ([WR1, WR2], 
    notch = True, widths = [0.2,0.2],
    patch_artist=True)

plt.show (block=False)


violet = "#462066"
deluge = "#7C71AD"  
bamboo = "#DC5C05"
orange = "#FF9000"
oyster = '#978B7D'
yellow = '#FFAC00'
downy = "#6EC5B8"
coral = '#C7BAA7'

lw = 2.75

for i in range (2):
    setp(bp['fliers'][i], marker = 'o',
        markerfacecolor=bamboo, markeredgecolor=bamboo)

setp (bp['boxes'][0], edgecolor=violet, fill=False, linewidth=lw)
setp (bp['medians'][0], color=violet, linewidth=lw)
setp (bp['whiskers'][0], color=violet, linewidth=lw)
setp (bp['whiskers'][1], color=violet, linewidth=lw)
setp (bp['caps'][0], color=violet, linewidth=lw)
setp (bp['caps'][1], color=violet, linewidth=lw)

setp (bp['boxes'][1], edgecolor=violet, fill=False, linewidth=lw)
setp (bp['medians'][1], color=violet, linewidth=lw)
setp (bp['whiskers'][2], color=violet, linewidth=lw)
setp (bp['whiskers'][3], color=violet, linewidth=lw)
setp (bp['caps'][2], color=violet, linewidth=lw)
setp (bp['caps'][3], color=violet, linewidth=lw)


plt.plot ([1,2], [V_val, V_val], 
    linestyle='None',
    color = deluge,
    markeredgecolor = deluge,
    marker="s", 
    markersize=9)

plt.plot ([1,2], [L_val, L_val], 
    linestyle='None',
    color = downy,
    markeredgecolor = downy,
    marker="^", 
    markersize=11)

# legend

vleg = mlines.Line2D ([], [], 
                    color=deluge, 
                    markeredgecolor=deluge,
                    marker='s',
                    linestyle = 'None',
                    markersize=9, 
                    label='$\\lambda_1 (\Sigma)$')

lleg = mlines.Line2D ([], [], 
                    color=downy, 
                    markeredgecolor = downy,
                    marker='^',
                    linestyle = 'None',
                    markersize=10, 
                    label='$\\lambda_1 (L)$')

plt.legend (handles=[vleg, lleg], 
    numpoints = 1, fancybox=True, framealpha=0.25)

ymin = np.min ((WR1, WR2))
ymin = 0.95*np.min ((ymin, V_val, L_val))
ymax = np.max ((WR1, WR2))
ymax = 1.05*np.max ((ymax, V_val, L_val))

#ymin = 14.5 
#ymax = 21.5

plt.ylim ([ymin, ymax])

plt.ylabel ('Ann. Volatility (%)',
    fontsize=16, weight=250, color='black')
plt.yticks (fontsize=13)
plt.xticks ([1, 2], ['T = 250', 'T = 1000'], y = -0.01,
    fontsize=16, weight=250, color='black')

plt.title ('SLR (N = ' + str (fm.N) + ')', 
    y = 1.01, fontsize=18)

# the following is a complete title for the image (not set)

title = "SLR Eigenvalue estimates with " 
title = title + str (nt) + " simulations. \n"
title = title + "N=" + str(fm.N) + " with "
if (fm.K0 == 1):
    title = title + str(fm.K0) + " factor"
else:
    title = title + str(fm.K0) + " factors"

if (len (fm.KS) >= 2):
    if ( fm.K1 > 0 ):
        title = title + ", " + str(fm.K1) + " countries"
    
    if (len (fm.KS) >= 3):
        title = title + ", " + str(fm.K2) + " industries"
    #@ if
    if (len (fm.KS) >= 4):
        title = title + ", " + str (sum (fm.KS[3:])) + " Other"
    #@ if

    title = title + "."

#@ if
#plt.title(title)

plt.savefig ("img/SLReigenvalue_alt_" + id_str, 
    transparent=True, dpi=256)

plt.show (block=False)


# RIGHT PANEL slide 20 with different ylims

# pool = CPU (processes=cpus_)

# seed values for sample covariance matrices
seeds = range (1,1+nt)

# T = 250 and T = 1000 experiments
X1 = zip (repeat (fm), repeat (250), seeds)
X2 = zip (repeat (fm), repeat (1000), seeds)

# run forcast nt times using 4 threads
# WR1 = pool.starmap (eigen, X1)
# WR2 = pool.starmap (eigen, X2)
WR1 = list(starmap (eigen, X1))
WR2 = list(starmap (eigen, X2))

# pool.close()
# pool.join()

S, L, D = fm.covariances()
V = S + L + D
    
# compute eigenvalues of V, L and M = L + S

V_val = eigsh (V, 1, which='LA', 
    maxiter=int (1e9),
    return_eigenvectors=False)[0]
L_val = eigsh (L, 1, which='LA', 
    maxiter=int (1e9),
    return_eigenvectors=False)[0]
M_val = eigsh (L+S, 1, which='LA', 
    maxiter=int (1e9),
    return_eigenvectors=False)[0]

# rescale to ann. vols. %

WR1 = np.sqrt (np.asarray (WR1) / fm.N * 256) * 100
WR2 = np.sqrt (np.asarray (WR2) / fm.N * 256) * 100
L_val = np.sqrt (L_val / fm.N * 256) * 100
V_val = np.sqrt (V_val / fm.N * 256) * 100
M_val = np.sqrt (M_val / fm.N * 256) * 100

# remove these to keep same scale
ymin = np.min ((WR1, WR2))
ymin = 0.95*np.min ((ymin, V_val, M_val, L_val))
ymax = np.max ((WR1, WR2))
ymax = 1.05*np.max ((ymax, V_val, M_val, L_val))

# make the plots

id_str = ''.join('{}{}'.format(key, val) 
    for key, val in sorted(signature.items()))

fig_w = 7
fig_h = 6

plt.figure ('PCA' + id_str, figsize=(fig_w,fig_h), dpi=100)

bp = plt.boxplot ([WR1, WR2], 
    notch = True, widths = [0.2,0.2],
    patch_artist=True)

violet = "#462066"
deluge = "#7C71AD"  
bamboo = "#DC5C05"
orange = "#FF9000"
oyster = '#978B7D'
yellow = '#FFAC00'
downy = "#6EC5B8"
coral = '#C7BAA7'

lw = 2.75

for i in range (2):
    setp(bp['fliers'][i], marker = 'o',
        markerfacecolor=bamboo, markeredgecolor=bamboo)

setp (bp['boxes'][0], edgecolor=violet, fill=False, linewidth=lw)
setp (bp['medians'][0], color=violet, linewidth=lw)
setp (bp['whiskers'][0], color=violet, linewidth=lw)
setp (bp['whiskers'][1], color=violet, linewidth=lw)
setp (bp['caps'][0], color=violet, linewidth=lw)
setp (bp['caps'][1], color=violet, linewidth=lw)

setp (bp['boxes'][1], edgecolor=violet, fill=False, linewidth=lw)
setp (bp['medians'][1], color=violet, linewidth=lw)
setp (bp['whiskers'][2], color=violet, linewidth=lw)
setp (bp['whiskers'][3], color=violet, linewidth=lw)
setp (bp['caps'][2], color=violet, linewidth=lw)
setp (bp['caps'][3], color=violet, linewidth=lw)


plt.plot ([1,2], [V_val, V_val], 
    linestyle='None',
    color = deluge,
    markeredgecolor = deluge,
    marker="s", 
    markersize=9)

plt.plot ([1,2], [M_val, M_val], 
    linestyle='None',
    color = orange,
    markeredgecolor = downy,
    marker="o", 
    markersize=10)

plt.plot ([1,2], [L_val, L_val], 
    linestyle='None',
    color = downy,
    markeredgecolor = downy,
    marker="^", 
    markersize=11)

# legend

vleg = mlines.Line2D([], [], 
                    color=deluge, 
                    markeredgecolor=deluge,
                    marker='s',
                    linestyle = 'None',
                    markersize=9, 
                    label='$\\lambda_1 (\Sigma)$')

mleg = mlines.Line2D([], [], 
                    color = orange,
                    markeredgecolor = downy,
                    marker='o',
                    linestyle = 'None',
                    markersize=10, 
                    label='$\\lambda_1 (\Lambda)$')

lleg = mlines.Line2D([], [], 
                    color=downy, 
                    markeredgecolor = downy,
                    marker='^',
                    linestyle = 'None',
                    markersize=11, 
                    label='${\\lambda_1 (L)}$')

legend = plt.legend(handles=[vleg, mleg, lleg], 
    numpoints = 1, fancybox=True, framealpha=0.25)


plt.ylim ([ymin, ymax])

plt.ylabel ('Ann. Volatility (%)',
    fontsize=16, color='black')
plt.yticks (fontsize=13)
plt.xticks ([1, 2], ['T = 250', 'T = 1000'], y = -0.01,
    fontsize=16, color='black')

plt.title ('PCA (N = ' + str (fm.N) + ')', 
    y = 1.01, fontsize=18)

# the following is a complete title for the image (not set)

title = "PCA Eigenvalue estimated with " 
title = title + str (nt) + " simulations. \n"
title = title + "N=" + str(fm.N) + " with "
if (fm.K0 == 1):
    title = title + str(fm.K0) + " factor"
else:
    title = title + str(fm.K0) + " factors"

if (len (fm.KS) >= 2):
    if ( fm.K1 > 0 ):
        title = title + ", " + str(fm.K1) + " countries"
    
    if (len (fm.KS) >= 3):
        title = title + ", " + str(fm.K2) + " industries"
    #@ if
    if (len (fm.KS) >= 4):
        title = title + ", " + str (sum (fm.KS[3:])) + " Other"
    #@ if

    title = title + "."

#@ if
#plt.title(title)


plt.savefig ("img/PCAeigenvalue_alt2_" + id_str, 
    transparent=True, dpi=256)

plt.show (block=False)







