
import numpy as np
import click

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from scipy.sparse.linalg import eigsh
from collections import defaultdict
from pylab import setp

uniform = np.random.uniform 
eigvals = np.linalg.eigvals

### Demonstrate the downward/upward bias for sample cov matrix under various covariance models
### There are 2 sources of randomness in this code: 
### 1. Seed for generate_simple_returns 
### 2. Seed for generate_uniform_volatilities

def generate_simple_returns(vols,T,seed=None):
   
    # vols - list of variances for each security
    # T    - number of samples
    # seed - seed of random number generator
    
    rng  = np.random.RandomState (seed)
    N    = len(vols)
    vcov = np.diag(vols)
    
    R = rng.multivariate_normal (np.zeros(N), vcov, T) 
    
    return R
#@def

def generate_uniform_volatilities(N,seed=None):
    
    # N - number of securities
    tdays = 256 
    rng  = np.random.RandomState(seed)
    vols = rng.uniform(((5/100)**2)/tdays,((30/100)**2)/tdays,N).tolist()   # assume 5-30% fluctuation a year
    
    return vols
 #@def
 
def cov_mat_sample(R):
    
    T = R.shape[0]
    vcov = R.T.dot (R) / T
    
    return vcov
#@def

def cov_mat_nls(R):
    return "not implemented yet"
#@def

def cov_mat_cv(R):
    return "not implemented yet"
#@def

def eig(A, k, return_eigenvectors=False):
    
    #extract eigenvalues with sparse lin algebra
    
    # A - matrix
    # k - number of eigenvalues to be extracted (<A.shape[1])
        
    result = eigsh(A, k, which='LA', maxiter=int(1e9),
                   return_eigenvectors=return_eigenvectors)
    if return_eigenvectors:
        vals, vecs = result
        return vals, vecs
    else:
        vals = result
        if k == 1:
            vals = vals[0]
        return vals
#@def
    
def simulate_eigs(simulations=1000,N=100,T=1000,seed=1):
    
    # this function calculates eigenvalues for every simulated matrix 
    # and stores the result in a hashtable 
    
    seeds        = range(seed, seed + simulations)
    eigenvalues  = defaultdict(list)
    vols         = np.sort(generate_uniform_volatilities(N,seed=seed))[::-1]

    for i in range(simulations):
        R    = generate_simple_returns(vols,T,seed=seeds[i])
        C_sample    = cov_mat_sample(R)
        #C_nls      = 
        #C_cv       = 
        eigenvalues['sample'+'_'+\
                    'T='+str(T)+'_'+\
                    'N='+str(N)].append(np.sort(eigvals(C_sample))[::-1])
    #@for
        
    return eigenvalues,vols
#@def
    
def read_inquiry(inq):
    # inq = inquiry of the form (estimator,T,N)
    return inq[0]+'_T='+str(inq[1])+'_N='+str(inq[2])
#@def

def box_plot_eigs(simResult,T=1000,N=100,estimators=['sample'],qtile=5):
    
    # function plots an inquiry of the form (estimator,T,N) with all nonzero elements
    # it compares the distribution of each eigenvalue with the true eigenvalue
    
    # simresult  = output from simulate_eigs
    # estimators = estimator used to find the covariance matrix 
    # T,N        = number of samples and number of stocks
    # qtile      = quantile of eigenvalues which to display (for visualization purposes)
    
    # convert daily variances into annualized std in % 
    
    tdays = 256 
    sd_annual = lambda x: np.sqrt(x*tdays)*100
    
    true_eigs = sd_annual(simResult[1])
        
    for i in estimators:
        inq = read_inquiry((i,T,N))
        est_eigs  = np.vstack(simResult[0][inq])
        est_eigs  = sd_annual(est_eigs)
    
        # choose eigenvalues on qtile quantiles
        xcoord      = np.rint(np.linspace(0,N-1,qtile)).astype(int)
        box_medians = np.median(est_eigs,0)
        
        # plotting part
        
        violet = "#462066"
        bamboo = "#DC5C05"
        orange = "#FF9000"
        
        fig_w = 7
        fig_h = 6
        lw = 1.5
    
        plt.figure(figsize=(fig_w, fig_h), dpi=100)
    
        bp = plt.boxplot(est_eigs[:,xcoord],
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
        
        plt.plot(list(range(1,qtile+1)),
                 true_eigs[xcoord],
                 marker="o",
                 markerfacecolor=bamboo, 
                 markeredgecolor=bamboo,
                 markersize=3,
                 zorder=2)
        
        # draw titles, axes and etc.
        
        plt.title('N = ' + str(N)+', T= ' + str(T), y=1.01,
                  fontsize=18, color='black')
        plt.xticks(list(range(1,qtile+1)),
                   ['${\\lambda_{'+ str(i) + '}(\hat{\Sigma})}$' for i in xcoord+1])
        plt.ylabel('Ann.Volatility(%)')
        
        # show the difference between true eigenvalues and centers of boxplots
     
        plt.fill_between(list(range(1,qtile+1)), 
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

        legend = plt.legend(handles=[vleg,mleg],
                            numpoints=1, fancybox=True, framealpha=0.25)
        
        plt.show()
        #plt.clf()
        
    #@for
    
#@def

@click.command()
@click.option('--n', type=int, default=10)
@click.option('--t', type=int, default=100)
@click.option('--simulations', type=int, default=1000)
@click.option('--seed', type=int, default=1)
@click.option('--qtile', type=int, default=5)
@click.option('--estimators', type=list, default=['sample'])

def main(n,t,simulations,seed,qtile,estimators):
    simResult = simulate_eigs(simulations=simulations,N=n,T=t,seed=seed)
    box_plot_eigs(simResult,estimators=estimators,qtile=qtile,T=t,N=n)  
#@main

if __name__ == "__main__":
    main()
