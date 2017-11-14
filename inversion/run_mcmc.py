# -*- coding: utf-8 -*-
"""
A Python 2.7 implementation of the DREAMzs MCMC sampler (Vrugt et al., 2009, 
Laloy and Vrugt, 2012) tailored to the geostatistical inverse problems considered in 
Laloy et al. (2017).  This DREAMzs implementation is based on the 2013 DREAMzs Matlab code 
(version 1.5, licensed under GPL3) written by Jasper Vrugt (FYI: a more recent Matlab code
with many more options is available at http://faculty.sites.uci.edu/jasper/). 

Version 0.0 - October 2016. Probably a bit non-pythonic coding and not optimized for 
speed, but the does the job and the forward model evaluations can be performed in 
parallel on several cores.

@author: Eric Laloy <elaloy@sckcen.be>

Please drop me an email if you have any question and/or if you find a bug in this
program. 

Also, if you find this code useful please make sure to cite the paper for which it 
has been developed (Laloy et al., 2017).

===
Copyright (C) 2017  Eric Laloy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ===                               

References:
    
Laloy et al., AWR (2017)
    
Laloy, E., Vrugt, J.A., High-dimensional posterior exploration of hydrologic models      
    using multiple-try DREAMzs and high-performance computing, Water Resources Research, 
    48, W01526, doi:10.1029/2011WR010608, 2012.
    
ter Braak, C.J.F., Vrugt, J.A., Differential Evolution Markov Chain with snooker updater 
    and fewer chains, Statistics and Computing, 18, 435–446, doi:10.1007/s11222-008-9104-9,
	2008.
    
Vrugt, J. A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and J.M. Hyman,
    Accelerating Markov chain Monte Carlo simulation by differential evolution with
    self-adaptive randomized subspace sampling, International Journal of Nonlinear Sciences
    and Numerical Simulation, 10(3), 273-290, 2009.                                         
                                                                                                                                                                                                       
"""

import os
import time

main_dir='/home/elaloy/VAE_MCMC/Inversion' # Set the working directory

os.chdir(main_dir)

import numpy as np
import shutil
import mcmc

#% Set rng_seed and case study
rng_seed=123 # np.random.seed(np.floor(time.time()).astype('int'))

CaseStudy=2
 
if  CaseStudy==0: #100-d correlated gaussian (case study 2 in DREAMzs Matlab code)
    seq=3
    steps=5000
    ndraw=seq*100000
    thin=10
    jr_scale=1.0
    nCR=3
    Prior='LHS' # Uniform prior with initial points sampled by LHS 
    # (note that here the model returns the posterior density directly)
    DoParallel=False
    MakeNewDir=False

if  CaseStudy==1: #10-d bimodal distribution (case study 3 in DREAMzs Matlab code)
    seq=5
    ndraw=seq*40000
    thin=10
    steps=np.int32(ndraw/(20.0*seq))
    jr_scale=1.0
    nCR=3
    Prior='COV' # Uniform prior with initial points sampled from a standard normal distribution 
    # (note that here the model returns the posterior density directly)
    DoParallel=False
    MakeNewDir=False
    
if  CaseStudy==2: # 2D steady-state flow problem
    seq=8
    ndraw=seq*25000
    thin=1
    steps=100
    jr_scale=1
    nCR=10
    Prior='COV' # See explanations in paper
    DoParallel=True
    MakeNewDir=True
    # Note that under Windows, DoParallel = True does not work with MakeNewDir = True (I don't know why)
    # So that if using Windows, one has to copy the 1 to seq modflow folders manually first, and then run 
    # in parallel with DoParllel = True and MakeNewDir = False 

if  CaseStudy==3: # 3D transient hydraulic tomography problem
    seq=8
    ndraw=seq*25000
    thin=1
    steps=100
    jr_scale=1
    nCR=10
    Prior='COV' # See explanations in paper
    DoParallel=True
    MakeNewDir=True
    # See above warning for Windows users
    
if MakeNewDir==True: # Creat the 1 to seq modflow files for multi-core computing
    src_dir=main_dir+'/'+'modflow_'+str(CaseStudy)+'D'
    for i in range(1,seq+1):
        dst_dir=main_dir+'/'+'modflow_'+str(i)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir,dst_dir)

#% Run the DREAMzs algorithm
if __name__ == '__main__':
    
    start_time = time.time()
    
    q=mcmc.Sampler(CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,Prior=Prior,parallel_jobs=seq,steps=steps,
                   parallelUpdate = 0.9,pCR=True,thin=thin,nCR=nCR,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Reflect',
                   lik_sigma_est=False,DoParallel=DoParallel,jr_scale=jr_scale,rng_seed=rng_seed)
    
    print("Iterating")
    
    tmpFilePath=None # None or: main_dir+'\out_tmp.pkl'
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar = q.sample(RestartFilePath=tmpFilePath)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
    
#%%