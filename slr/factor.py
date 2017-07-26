import numpy as np

# some numpy functions and libraries 
multivariate_normal = np.random.multivariate_normal 
normal = np.random.normal
choice = np.random.choice 
uniform = np.random.uniform 

# see below for fm_=factor_model

###################################################
#                                                 #
# Generates a factor model of the form            #
# R = psi Y + phi X + eps                         #
#                                                 # 
# Y - K0 by N matrix of global factor exposures   #
# X - (K1 + K2 + ...) by N matrix of sparse       #
#     factor exposures.                           #
# psi - global factor returns                     #
# phi - sparse factor returns                     #
# eps - idiosyncratic returns                     #
#                                                 #
# Instantiated as fm = factor_model(N = 32, ...)  #
# or fm = fm_(N = 32, ...) for shorthand. Every   #
# argument has a default value and so is not      #
# required to be specified.                       #
#                                                 #
# @N - total number of securities                 #
# @K0 - number of global (broad) factors          #
# @K1 - number of sparse factors of type 1        #
# ... K2, K3, K4 and so on.                       #
# @vol0 - global factor volatilities              #
# @vol1 - type 1 sparse factor volatilities       #
# ... vol2, vol3 and so on.                       #
# @vols - idiosyncratic volatilities              #
#                                                 #
###################################################

class factor_model:

    machine_tol = 1e-15 
    num_tdays = 256
    pavol2var = lambda x: (x/100)**2 / factor_model.num_tdays 

    # GEM 3 ann. volatilities in %
    country_vols_dev = [14.42, 14.11, 11.56, 9.26, 13.42, 14.00, 
        11.37, 11.69, 26.56, 20.67, 16.01, 17.82, 12.65, 18.42, 
        11.88, 13.22, 16.30, 13.90, 19.29, 12.04, 13.80, 10.74, 
        10.44, 10.50]

    country_vols_emg = [19.46, 12.16, 28.44, 18.27, 19.77, 24.16,
        23.58, 24.55, 27.97, 30.56, 23.75, 14.15, 15.54, 17.13, 
        24.14, 22.15, 11.49, 41.49, 13.56,  24.50, 26.36, 37.74]

    industry_vols = [23.19, 8.81, 15.06, 5.15, 4.89, 10.27, 22.64,
        8.33, 3.39, 4.17, 4.59, 12.90, 6.86, 4.65, 5.76, 5.36, 
        6.86, 5.44, 4.03, 7.12, 7.02, 16.14, 6.62, 5.43, 5.27, 
        6.33, 5.68, 16.89, 7.01, 11.13, 6.55, 15.13, 6.03, 5.69]

    def __init__ (self, **kwargs):
    
        dict_of_defaults = dict (N=32, K0=0, K1=0, seed = None)
        keys = kwargs.keys()
    
        # notify user of defaults used
        for key, val in dict_of_defaults.items():
            if key not in keys:
                print ("Defaulting %s to %i." % (key,val))
            #@ if
        #@ for 

        # overwrite defaults with new arguments
        dict_of_defaults.update (kwargs) 
        kwargs = dict_of_defaults
        keys = kwargs.keys()
    
        # set supplied arguments to attributes
        # also parse the factor types: K0, K1, K2, ...
        # the list KS must have at least K0 and K1 
        self.KS = []
        for key, val in kwargs.items():
            setattr (self, key, val)
            
            label = 'K' + str ( len (self.KS) ) 
            if label in keys:
                self.KS.append ( kwargs[label] )
            #@ if
        #@ for

        # set the random numbers seed for self
        self.signature = [self.seed, self.N] + self.KS 
        if self.seed is not None:
            np.random.seed (self.signature)

        # generate volatilities unless provided
        for i in range (len (self.KS)):
            # generate even if provided to keep consistent
            # random number generation 
            vols = self.generate_volatilities (i, self.KS[i]) 
            
            # if not supplies assign 
            key = 'vol' + str (i)
            if ( key not in keys ):
               setattr (self, key, vols)  
            #@ if
        #@ for

        # generate idiosyncratic volatilities
        if ('vols' not in keys):
            self.vols = self.generate_volatilities (-1, self.N) 
        #@ if
 
        # generate exposures
        self.Y = self.generate_global_exposures()
        self.X = self.generate_sparse_exposures()
        
        if self.seed is not None:
            self.seed_return_generating_process()
    #@ __init__


    ######################################################
    # Create a random number generator for each of the 3 #
    # return streams: phi, psi, eps                      #
    #                                                    #
    # @seed - random number generator seed               #
    # can be either an int or a 3-tuple                  #
    #                                                    #
    ######################################################
    def seed_return_generating_process (self, seed = 0):
        
        seed = (np.int_ ([1,1,1]) + np.int_ (seed)).tolist()
        
        # set up the random number generators for the returns
        # len (rngs)  must be <= len (self.signature) for now
        ret = ['psi', 'phi', 'eps']

        sign = self.signature
        sign = [0,0,0] + sign

        # see each return type in rngs with a unique seed
        for i,r in enumerate (ret):
            sign[i] += seed[i]
            rng = r + '_rng'
            setattr (self, rng, np.random.RandomState (sign))
        #@ for
    #@ def


    ###################################################
    #                                                 #
    # Generates T realization of the returns          #
    # [ psi Y, phi X, eps ]                           #
    #                                                 #
    # @T - number of observations (int > 0)           #
    # @seed - random number generator seed            #
    #                                                 #
    ###################################################
    def generate_returns (self, T, seed = None):

        if (seed is not None):
            self.seed_return_generating_process (seed)
        #@ if

        psi = self.generate_global_factor_returns (T)
        phi = self.generate_sparse_factor_returns (T)
        eps = self.generate_idiosyncratic_returns (T)

        return [psi, phi, eps]


    ####################################################
    # Computes T realizations of returns to securities #
    # R = psi Y + phi X + eps                          #
    #                                                  #
    # @T - number of observations                      #
    #                                                  #
    ####################################################
    def R (self, T, seed = None):

        psi, phi, eps = self.generate_returns (T, seed)

        return psi.dot (self.Y) + phi.dot (self.X) + eps
    #@ def


    ####################################################
    #                                                  #
    # Construct data covariance matrices based on T    #
    # T observations. T = infinity indicated the exact #
    # covariance is sought.                            #
    #                                                  #  
    # @T - number of observations                      #
    #                                                  #
    # *use SLD for shorthand                           #
    ####################################################
    def covariances (self, T = np.inf, seed = None):
    
        # population covariances
        if np.isinf (T):
      
            if self.K0 <= 0:
                L = np.zeros ( (self.N, self.N) )
            else:
                fvars = list (map (fm_.pavol2var, self.vol0))
                L = self.Y.T.dot ( np.diag (fvars) )
                L = L.dot (self.Y)
            #@ if 
        
            if self.K1 <= 0:
                S = np.zeros ( (self.N, self.N) )
            else:
                population_sizes = self.KS[1:]
                mean = np.zeros ( np.sum (population_sizes) )
            
                fvars = self.get_sparse_factor_volatilities()
                fvars = list (map (fm_.pavol2var, fvars))

                S = self.X.T.dot ( np.diag (fvars) )
                S = S.dot (self.X)
          
                # sparsity = (S == 0).sum() / (self.N**2)
                # print ('Measure of sparsity: ' + str (sparsity))
            #@ if
            svars = list (map (fm_.pavol2var, self.vols))
            D = np.diag (svars)
        else: # sample covariance
            psi, phi, eps = self.generate_returns (T, seed)

            RY = psi.dot (self.Y)
            L = RY.T.dot (RY) / T
            
            RX = phi.dot (self.X)
            S = RX.T.dot (RX) / T
            
            D = eps.T.dot (eps) / T 
        #@ if

        return [S, L, D]
    #@ def


    def covariance (self, T = np.inf, seed = None):
        S, L, D = self.covariances (T, seed)
        return S + L + D
    #@ def
    
  
    ########################################################
    # Methods below are in a sense private (used in init)  #
    ########################################################

    def generate_global_exposures (self):
    
        # check if there are global factors
        if self.K0 <= 0:
            return np.zeros (1) # or something else?
      
        # market factor
        Y0 = np.reshape (np.ones (self.N), (1, self.N))
        #Y0 = normal (1.0, 0.25, self.N)
        #Y0 = Y0 * np.sqrt (self.N) / np.linalg.norm (Y0)

        # number of remaining factors
        K = self.K0 - 1

        if K <= 0:
            return np.reshape (Y0, (1,self.N))
    
        # style factors
        mean = np.zeros (K) 
        vcov = np.diag (np.ones (K))
    
        Y1 = multivariate_normal (mean, vcov, self.N).T
    
        return np.vstack ((Y0, Y1))
    #@ def 


    def generate_sparse_exposures (self):
   
        # if no sparse factors are present return
        if self.K1 <= 0: 
            return np.zeros (1) # or something else?

        # enumerates numbers of factors of each type
        population_sizes = self.KS[1:]
        # counts size of factor 
        sizes = np.ones (self.N) / self.N

        # iterate over each type of sparse factor
        z = []
        for K in population_sizes:
            # matrix of exposures to this sparse factor
            Z = np.zeros ( (K, self.N) )

            # order the indices in proportion to populations
            indices = choice (self.N,self.N,p=sizes,replace=False)
            indices = indices.tolist()

            # initialize initial assignments to the K factors at
            # the end each factor will have exactly 1 security.
            init = choice (indices, K, replace=False)
            for i in range (K):
                Z[i, init[i]] = 1
                indices.remove ( init[i] )
            #@ for 

            for j in indices:
                # assignment probabilities
                prob = np.sum (Z, 1)
                # pick uniformly with prob from unselected factors
                prob = prob/prob.sum()

                # choose a sparse factor in proportion to size
                i = choice (K, 1, p = prob)[0]
                # add security to this sparse factor
                Z[i,j] = 1
            #@ for
      
            # for each security store the size of the population
            # of the factor it was assigned to.
            for j in range (self.N):
                i = Z[ : , j].tolist().index(1)
                sizes[j] = np.sum ( Z [i, ] )
            #@ for
            sizes = sizes / np.sum (sizes)

            # store the set (type) of sparse factors 
            z.append (Z)
        #@ for
    
        return np.vstack (z)
    #@ def


    def generate_volatilities (self, t, K):
  
        if K <= 0:
            return [] # or something else?

        # idiosyncratic volatilities
        if t < 0:
            vols = uniform (10, 60, K).tolist()
        # global factor volatilities
        elif t == 0:
            vols = []
            # generate style factor volatilities
            if (K >= 2):
                vols = uniform (1, 4, K-2).tolist() 
                # add beta style factor
                vols.insert (0, 8.0)
            # add market factor volatility
            vols.insert (0, 16.0)
        # sparse factor volatilities (scale with 1/t)
        elif t == 1:
            vols = fm_.country_vols_dev + fm_.country_vols_emg
            vols = choice (vols, K, replace=False).tolist()
        elif t == 2:
            vols = fm_.industry_vols
            vols = choice (vols, K, replace=False).tolist()
        else:
            cent = 20 / np.sqrt (t)
            mdev = 10 / np.log (1+t) 
            vols = uniform (cent-mdev, cent+mdev, K).tolist()
        return vols
    #@ def

    
    def generate_global_factor_returns (self, T):
    
        if self.K0 <= 0:
            return np.zeros (1) # force dimension to T x K0?

        mean = np.zeros (self.K0)
        vcov = np.diag (list (map (fm_.pavol2var, self.vol0)))
        
        rng = self.psi_rng

        return rng.multivariate_normal (mean, vcov, T)
    #@ def


    def get_sparse_factor_volatilities (self):
        
        population_sizes = self.KS[1:]
        sfvols = []
        
        for t in range (len (population_sizes)):
            vols = getattr (self, 'vol' + str (t+1))
            sfvols.extend (vols)

        return sfvols
    #@ def


    def generate_sparse_factor_returns (self, T):
 
        if self.K1 <= 0:
            return np.zeros (1) # force dimension to T x K1?
    
        population_sizes = self.KS[1:]
        mean = np.zeros ( np.sum (population_sizes) )
    
        fvars = self.get_sparse_factor_volatilities()
        fvars = list (map (fm_.pavol2var, fvars))

        rng = self.phi_rng
        
        return rng.multivariate_normal (mean, np.diag (fvars), T)
    #@ def
  

    def generate_idiosyncratic_returns (self, T):
    
        if (self.vols is None):
            return np.zeros (1) # force dimension to T x N?
 
        means = np.zeros (self.N)
        svols = np.sqrt (list (map (fm_.pavol2var, self.vols)))
        
        rng = self.eps_rng

        return rng.normal (means, svols, (T,self.N))
    #@ def

    #####################################
    # aliases                           #
    #####################################
    srgp = seed_return_generating_process
    
    SLD = covariances
    cov = V = covariance
    gr = generate_returns
    
    psi = generate_global_factor_returns 
    phi = generate_sparse_factor_returns
    eps = generate_idiosyncratic_returns 

#@ class
fm_ = factor_model


   
  

