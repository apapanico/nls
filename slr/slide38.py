import matplotlib.pyplot as plt
import numpy as np
import portfolios as portfolio 
import factor
import ADMM

from importlib import reload
# reload any changes
reload (factor)
reload (ADMM)
reload (portfolio)

from scipy.sparse.linalg import eigsh
from multiprocessing import Pool as CPU

from itertools import repeat
from pylab import setp
from factor import fm_
from ADMM import SLR


def forecast_PCA (fm, T, K, seed=1):

    # generate returns
    R = fm.R (T=T, seed=seed)
    # sample covariance 
    # fix me: use the T x T matrix when T < N
    W = R.T.dot (R) / T

    vals, vecs = eigsh (W, K, which='LA', maxiter= int (1e9))
    dels = np.diag (W-vecs.dot (np.diag (vals)).dot (vecs.T))

    L_pca = vecs.dot (np.diag (vals)).dot (vecs.T)
    D_pca = np.diag (dels)
    V_pca = L_pca + D_pca

    # compute the optimized portfolio based on PCA estimate
    w = portfolio.longonly_minvar (fm, V_pca) 
    
    S, L, D = fm.covariances()
    V = S + L + D
    
    # compute the forecast ratios
    RV = w.T.dot (V_pca).dot (w) / w.T.dot (V).dot (w)
    RL = w.T.dot (L_pca).dot (w) / w.T.dot (L+S).dot (w)
    RD = w.T.dot (D_pca).dot (w) / w.T.dot (D).dot (w)

    return (RV, RL, RD)
#@ def

def forecast_SLR (fm, T, lam, eta, seed=1):

    S, L, D = fm.covariances()
    V = S + L + D
    
    R = fm.R (T, seed=seed)
    W = R.T.dot (R) / T

    # decompose as L + S
    st1 = SLR (W, lam = lam, gam = 6 / fm.N)
    # further decompose S as S + D
    st2 = SLR (st1['S'], lam = eta, gam = 0.0) 
        
    L_SLR = st1['L']
    D_SLR = st2['S']
    S_SLR = st1['S'] - D_SLR
    
    V_SLR = L_SLR + S_SLR + D_SLR
    
    # number of sparse factors
    K = sum (fm.KS[1:])

    # compute the optimized portfolio based on SLR estimate
    w = portfolio.longonly_minvar (fm, V_SLR) 

    # compute the forecast ratios
    RV = w.T.dot (V_SLR).dot (w) / w.T.dot (V).dot (w)
    RL = w.T.dot (L_SLR+S_SLR).dot (w) / w.T.dot (L+S).dot (w)
    RD = w.T.dot (D_SLR).dot (w) / w.T.dot (D).dot (w)
    
    return (RV, RL, RD, st1['rank'], st2['rank'])
#@ def


# LEFT PANEL
# number of simulations to run (number of sample cov matrices)
nt = 100
N = 128
# number of cpus to use in parallel
cpus_ = 4

# factor model
signature = dict (N=N, K0 = 4, K1=16, K2=8, seed=1)
fm = fm_(**signature)

# number of observations 
T = 250

pool = CPU (processes=cpus_)

# paramters for lam and eta controls the # of factors extracted
lam = 2e-4 * np.sqrt (N/T)
eta = 2e-4 * np.sqrt (N/T)
seeds = range (1,1+nt)

X = zip (repeat(fm), repeat (T), repeat (lam), repeat (eta),seeds)

# run forcast nt times using cpu_ threads
Y = pool.starmap (forecast_SLR, X)

pool.close()
pool.join()
    
# unpack the variance forecasts
VR, LR, DR, Lrank, Srank = zip (*Y)

# make the plots

deluge = "#7C71AD"
bamboo = "#DC5C05"
orange = "#FF9000"
oyster = '#978B7D'
yellow = '#FFAC00'
downy = "#6EC5B8"
coral = '#C7BAA7'

fig_w = 7
fig_h = 6

id_str = ''.join('{}{}'.format(key, val) 
    for key, val in sorted(signature.items()))

plt.figure ('SLR' + id_str, figsize=(fig_w,fig_h), dpi=100)

# multiple box plots on one figure
bp = plt.boxplot ([VR, LR, DR], 
    notch = True, widths = [0.2,0.2, 0.2],
    patch_artist=True)

lw = 2.75

for i in range (3):
    setp(bp['fliers'][i], marker = 'o',
        markerfacecolor=bamboo, markeredgecolor=bamboo)

setp (bp['boxes'][0], facecolor=deluge, 
    edgecolor=downy, linewidth=lw)
setp (bp['medians'][0], color=downy, linewidth=4)
setp (bp['whiskers'][0], color=deluge, linewidth=lw)
setp (bp['whiskers'][1], color=deluge, linewidth=lw)
setp (bp['caps'][0], color=deluge, linewidth=lw)
setp (bp['caps'][1], color=deluge, linewidth=lw)


setp (bp['boxes'][1], edgecolor=downy, 
    facecolor=orange, linewidth=lw)
setp (bp['medians'][1], color=downy, linewidth=4)
setp (bp['whiskers'][2], color=orange, linewidth=lw)
setp (bp['whiskers'][3], color=orange, linewidth=lw)
setp (bp['caps'][2], color=orange, linewidth=lw)
setp (bp['caps'][3], color=orange, linewidth=lw)


setp (bp['boxes'][2], edgecolor=orange, 
    facecolor=oyster, linewidth=lw)
setp( bp['medians'][2], color=orange, linewidth=4)
setp (bp['whiskers'][4], color=oyster, linewidth=lw)
setp (bp['whiskers'][5], color=oyster, linewidth=lw)
setp (bp['caps'][5], color=oyster, linewidth=lw)
setp (bp['caps'][5], color=oyster, linewidth=lw)

plt.ylabel ('Forecast Ratio ' + 
    '$\\mathscr{R}_{\cdot} ( w \\,|\\, \\theta)$',
    fontsize=16, color='black')
plt.yticks (fontsize=13)
plt.xticks([1, 2, 3], ['Portfolio $(\\Sigma)$', 
    'Factor $(S+L)$', 'Specific $(\\Delta)$'],
    y = -0.01, fontsize=16, color='black')

plt.plot ([0,1,2,3,4], [1.0, 1.0, 1.0, 1.0, 1.0],     
    linestyle=':',
    color = coral,
    linewidth=2)

#ymin = 0.35 
#ymax = 1.10

#plt.ylim ([ymin, ymax])

# short title
plt.title ('SLR', y = 1.01,
    fontsize=18, color='black')
#plt.title ('SLR (N = ' + str (fm.N) + ')', y = 1.01,
#    fontsize=18, color='black')

# the following is a complete title for the image (not set)
title = "SLR Forecast Ratios " 
title = title +  "(OPT): " 
title = title + str (nt) + " simulations. \n"
title = title + "N=" + str(fm.N) + ", "
title = title + "T=" + str(T) + " and "
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

plt.savefig ('img/SLRforecast_OPT_' + id_str, 
    transparent=True, dpi=256)

plt.show (block=False)

print ('SLR PFR mean = ' + str (np.mean (VR)))
print ('SLR FFR mean = ' + str (np.mean (LR)))
print ('SLR SFR mean = ' + str (np.mean (DR)))
print ('Broad factors = ' + str (np.mean (Lrank)))
print ('Sparse factors = ' + str (np.mean (Srank)))


# RIGHT PANEL

pool = CPU (processes=cpus_)

# number of principal component to extract
K = sum (fm.KS) 

# seeds for the sample covariance matrices
seeds = range (1,1+nt)

X = zip (repeat (fm), repeat (T), repeat (K), seeds)

# run forcast nt times using cpu_ threads
Y = pool.starmap (forecast_PCA, X)

pool.close()
pool.join()
    
# unpack the variance forecasts
VR, LR, DR = zip (*Y)

# make the plots

deluge = "#7C71AD"
bamboo = "#DC5C05"
orange = "#FF9000"
oyster = '#978B7D'
yellow = '#FFAC00'
downy = "#6EC5B8"
coral = '#C7BAA7'

fig_w = 7
fig_h = 6

id_str = ''.join('{}{}'.format(key, val) 
    for key, val in sorted(signature.items()))

plt.figure ('PCA' + id_str, figsize=(fig_w,fig_h), dpi=100)

# multiple box plots on one figure
bp = plt.boxplot ([VR, LR, DR], 
    notch = True, widths = [0.2,0.2, 0.2],
    patch_artist=True)

lw = 2.75

for i in range (3):
    setp(bp['fliers'][i], marker = 'o',
        markerfacecolor=bamboo, markeredgecolor=bamboo)

setp (bp['boxes'][0], facecolor=deluge, 
    edgecolor=downy, linewidth=lw)
setp (bp['medians'][0], color=downy, linewidth=4)
setp (bp['whiskers'][0], color=deluge, linewidth=lw)
setp (bp['whiskers'][1], color=deluge, linewidth=lw)
setp (bp['caps'][0], color=deluge, linewidth=lw)
setp (bp['caps'][1], color=deluge, linewidth=lw)


setp (bp['boxes'][1], edgecolor=downy, 
    facecolor=orange, linewidth=lw)
setp (bp['medians'][1], color=downy, linewidth=4)
setp (bp['whiskers'][2], color=orange, linewidth=lw)
setp (bp['whiskers'][3], color=orange, linewidth=lw)
setp (bp['caps'][2], color=orange, linewidth=lw)
setp (bp['caps'][3], color=orange, linewidth=lw)


setp (bp['boxes'][2], edgecolor=orange, 
    facecolor=oyster, linewidth=lw)
setp( bp['medians'][2], color=orange, linewidth=4)
setp (bp['whiskers'][4], color=oyster, linewidth=lw)
setp (bp['whiskers'][5], color=oyster, linewidth=lw)
setp (bp['caps'][5], color=oyster, linewidth=lw)
setp (bp['caps'][5], color=oyster, linewidth=lw)

plt.ylabel ('Forecast Ratio ' + 
    '$\\mathscr{R}_{\cdot} ( w \\,|\\, \\theta)$',
    fontsize=16, color='black')
plt.yticks (fontsize=13)
plt.xticks([1, 2, 3], ['Portfolio $(\\Sigma)$', 
    'Factor $(S+L)$', 'Specific $(\\Delta)$'],
    y = -0.01, fontsize=16, color='black')

plt.plot ([0,1,2,3,4], [1.0, 1.0, 1.0, 1.0, 1.0],     
    linestyle=':',
    color = coral,
    linewidth=2)

#ymin = 0.35 
#ymax = 1.10 

#plt.ylim ([ymin, ymax])

# short title
# plt.title ('PCA (N = ' + str (fm.N) + ')', y = 1.01,
#    fontsize=18, color='black')
plt.title ('PCA', y = 1.01,
    fontsize=18, color='black')

# the following is a complete title for the image (not set)
title = "PCA Forecast Ratios " 
title = title +  "(OPT): " 
title = title + str (K) + " PCs, "   
title = title + str (nt) + " simulations. \n"
title = title + "N=" + str(fm.N) + ", "
title = title + "T=" + str(T) + " and "
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

plt.savefig ('img/PCAforecast_OPT_' + id_str, 
    transparent=True, dpi=256)

plt.show (block=False)

print ('PCA PFR mean = ' + str (np.mean (VR)))
print ('PCA FFR mean = ' + str (np.mean (LR)))
print ('PCA SFR mean = ' + str (np.mean (DR)))
print ('PCA receives exact number of factors to recover')

