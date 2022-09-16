# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:48:55 2022

@author: ronni
"""

import h5py
import numpy as np
#import scipy.io as sci
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import time

C=38                  #4 outputs
mb_size=192
num_epochs = 45
sample=0
testno=0
testno_V=0
filename="CNNnew120"
print(filename)
b = {}
n= {}

def output (X,C,mini_batch_size,train_no,test_no,sample):

    Y_train=np.zeros((C,1,train_no))
    Y_test=np.zeros((C,1,test_no))
    Y_train[sample,:,:]=1
    Y_test[sample,:,:]=1
    return Y_train,Y_test

def minibatch_init(X,mb_size,train_size=0.9,use_size=1):
    count=0
    lis={}
    leent=math.floor(X.shape[0]/2)
    Y=np.zeros((leent,3))
    a=0
    count=0
    for c in range (0,leent):
        Y[c,:]=X[2*c,:]

    while(a<Y.shape[0]-mb_size):
        if(Y[a,1]>0.1 and Y[a+mb_size,1]>0.1):
            lis['M'+str(count)]=a
            a=a+mb_size
            count=count +1
        elif(Y[a,1]<0.1 and Y[a+1,1]>0.1):
            a=a+300
        else:
            a=a+1
    num_use_minibatches = math.floor(use_size*count)
    train_no=math.floor(train_size*num_use_minibatches)
    test_no=num_use_minibatches-train_no
    mini_batches_test=np.zeros((3,mb_size,test_no))
    mini_batches_train=np.zeros((3,mb_size,train_no))
    per = list(np.random.permutation(count))
    
    for k in range(0, train_no):
        Q=lis['M'+str(per[k])]
        mini_batches_train[:,:,k] = Y.T[:,Q:Q+mb_size]
    
    for k in range(train_no, num_use_minibatches):
        Q=lis['M'+str(per[k])]
        mini_batches_test[:,:,k-train_no] = Y.T[:,Q:Q+mb_size]
    
    return mini_batches_train,mini_batches_test,train_no,test_no
                   
def loaddata(M,C,mb_size,meas,testno,testno_V,sample):
    d=None
    d={}
    rett=None
    rett={}
    
    for k in range(1,meas+1):
        with h5py.File(M+str(k)+'.h5', 'r') as f:
            d[M+'{0}'.format(k)] = f['/'+M+str(k)].value
        
        d['mbtrain'+str(k)],d['mbtest'+str(k)],train_C,test_C=minibatch_init(d[M+str(k)],mb_size)
        d['Ytrain'+str(k)],d['Ytest'+str(k)]=output(d[M+str(k)],C,mb_size,train_C,test_C,sample)            

    if meas == 2:
        rett['Ytrain']=d['Ytrain1']
        rett['Ytest']=d['Ytest1']
        rett['mbtrain']=d['mbtrain1']
        rett['mbtest']=d['mbtest1']
    elif meas == 3:
        rett['Ytrain']=np.concatenate((d['Ytrain1'],d['Ytrain2']),axis=2)
        rett['Ytest']=np.concatenate((d['Ytest1'],d['Ytest2']),axis=2)
        rett['mbtrain']=np.concatenate((d['mbtrain1'],d['mbtrain2']),axis=2)
        rett['mbtest']=np.concatenate((d['mbtest1'],d['mbtest2']),axis=2)
    elif meas == 4:
        rett['Ytrain']=np.concatenate((d['Ytrain1'],d['Ytrain2'],d['Ytrain3']),axis=2)
        rett['Ytest']=np.concatenate((d['Ytest1'],d['Ytest2'],d['Ytest3']),axis=2)
        rett['mbtrain']=np.concatenate((d['mbtrain1'],d['mbtrain2'],d['mbtrain3']),axis=2)
        rett['mbtest']=np.concatenate((d['mbtest1'],d['mbtest2'],d['mbtest3']),axis=2)
    elif meas == 5:
        rett['Ytrain']=np.concatenate((d['Ytrain1'],d['Ytrain2'],d['Ytrain3'],d['Ytrain4']),axis=2)
        rett['Ytest']=np.concatenate((d['Ytest1'],d['Ytest2'],d['Ytest3'],d['Ytest4']),axis=2)
        rett['mbtrain']=np.concatenate((d['mbtrain1'],d['mbtrain2'],d['mbtrain3'],d['mbtrain4']),axis=2)
        rett['mbtest']=np.concatenate((d['mbtest1'],d['mbtest2'],d['mbtest3'],d['mbtest4']),axis=2)
    else:
        print('error')
    
    rett['Ytest_V']=np.concatenate((d['Ytrain'+str(meas)],d['Ytest'+str(meas)]),axis=2)
    rett['mbtest_V']=np.concatenate((d['mbtrain'+str(meas)],d['mbtest'+str(meas)]),axis=2)
    sample=sample+1
    testno=rett['Ytest'].shape[2]+testno
    testno_V=rett['Ytest_V'].shape[2]+testno_V

    return rett,testno,testno,testno_V,testno_V,sample

n['TV0']=n['T0']=0
nonf,n['T1'],testno,n['TV1'],testno_V,sample=loaddata('nonf',C,mb_size,4,testno,testno_V,sample)
lvf,n['T2'],testno,n['TV2'],testno_V,sample=loaddata('lvf',C,mb_size,4,testno,testno_V,sample)
hvf,n['T3'],testno,n['TV3'],testno_V,sample=loaddata('hvf',C,mb_size,4,testno,testno_V,sample)
lcf,n['T4'],testno,n['TV4'],testno_V,sample=loaddata('lcf',C,mb_size,4,testno,testno_V,sample)
hcf,n['T5'],testno,n['TV5'],testno_V,sample=loaddata('hcf',C,mb_size,4,testno,testno_V,sample)
hvhcf,n['T6'],testno,n['TV6'],testno_V,sample=loaddata('hvhcf',C,mb_size,4,testno,testno_V,sample)
hvlcf,n['T7'],testno,n['TV7'],testno_V,sample=loaddata('hvlcf',C,mb_size,4,testno,testno_V,sample)
lvhcf,n['T8'],testno,n['TV8'],testno_V,sample=loaddata('lvhcf',C,mb_size,4,testno,testno_V,sample)
lvlcf,n['T9'],testno,n['TV9'],testno_V,sample=loaddata('lvlcf',C,mb_size,4,testno,testno_V,sample)

mag,n['T10'],testno,n['TV10'],testno_V,sample=loaddata('mag',C,mb_size,4,testno,testno_V,sample)
lvmag,n['T11'],testno,n['TV11'],testno_V,sample=loaddata('lvmag',C,mb_size,4,testno,testno_V,sample)
hvmag,n['T12'],testno,n['TV12'],testno_V,sample=loaddata('hvmag',C,mb_size,4,testno,testno_V,sample)
lcmag,n['T13'],testno,n['TV13'],testno_V,sample=loaddata('lcmag',C,mb_size,4,testno,testno_V,sample)
hcmag,n['T14'],testno,n['TV14'],testno_V,sample=loaddata('hcmag',C,mb_size,4,testno,testno_V,sample)

heatD,n['T15'],testno,n['TV15'],testno_V,sample=loaddata('heatD',C,mb_size,4,testno,testno_V,sample)
lvoheatD,n['T16'],testno,n['TV16'],testno_V,sample=loaddata('lvoheatD',C,mb_size,4,testno,testno_V,sample)
hvoheatD,n['T17'],testno,n['TV17'],testno_V,sample=loaddata('hvoheatD',C,mb_size,4,testno,testno_V,sample)
lcoheatD,n['T18'],testno,n['TV18'],testno_V,sample=loaddata('lcoheatD',C,mb_size,4,testno,testno_V,sample)
hcoheatD,n['T19'],testno,n['TV19'],testno_V,sample=loaddata('hcoheatD',C,mb_size,4,testno,testno_V,sample)

heatmos,n['T20'],testno,n['TV20'],testno_V,sample=loaddata('heatmos',C,mb_size,4,testno,testno_V,sample)
lvoheatmos,n['T21'],testno,n['TV21'],testno_V,sample=loaddata('lvoheatmos',C,mb_size,4,testno,testno_V,sample)
hvoheatmos,n['T22'],testno,n['TV22'],testno_V,sample=loaddata('hvoheatmos',C,mb_size,4,testno,testno_V,sample)
lcoheatmos,n['T23'],testno,n['TV23'],testno_V,sample=loaddata('lcoheatmos',C,mb_size,4,testno,testno_V,sample)
hcoheatmos,n['T24'],testno,n['TV24'],testno_V,sample=loaddata('hcoheatmos',C,mb_size,4,testno,testno_V,sample)

cap,n['T25'],testno,n['TV25'],testno_V,sample=loaddata('cap',C,mb_size,4,testno,testno_V,sample)
lvcap,n['T26'],testno,n['TV26'],testno_V,sample=loaddata('lvcap',C,mb_size,4,testno,testno_V,sample)
hvcap,n['T27'],testno,n['TV27'],testno_V,sample=loaddata('hvcap',C,mb_size,4,testno,testno_V,sample)
lccap,n['T28'],testno,n['TV28'],testno_V,sample=loaddata('lccap',C,mb_size,4,testno,testno_V,sample)
hccap,n['T29'],testno,n['TV29'],testno_V,sample=loaddata('hccap',C,mb_size,4,testno,testno_V,sample)

ocf,n['T30'],testno,n['TV30'],testno_V,sample=loaddata('ocf',C,mb_size,4,testno,testno_V,sample)
scf,n['T31'],testno,n['TV31'],testno_V,sample=loaddata('scf',C,mb_size,4,testno,testno_V,sample)
ocD,n['T32'],testno,n['TV32'],testno_V,sample=loaddata('ocD',C,mb_size,4,testno,testno_V,sample)
scD,n['T33'],testno,n['TV33'],testno_V,sample=loaddata('scD',C,mb_size,4,testno,testno_V,sample)
ocL,n['T34'],testno,n['TV34'],testno_V,sample=loaddata('ocL',C,mb_size,4,testno,testno_V,sample)
scL,n['T35'],testno,n['TV35'],testno_V,sample=loaddata('scL',C,mb_size,4,testno,testno_V,sample)
ocmos,n['T36'],testno,n['TV36'],testno_V,sample=loaddata('ocmos',C,mb_size,4,testno,testno_V,sample)
scmos,n['T37'],testno,n['TV37'],testno_V,sample=loaddata('scmos',C,mb_size,4,testno,testno_V,sample)
occ,n['T38'],testno,n['TV38'],testno_V,sample=loaddata('occ',C,mb_size,4,testno,testno_V,sample)

#labels
n['L0']='OverAll'
n['L1']='No Fault'
n['L2']='Low Voltage'
n['L3']='High Voltage'
n['L4']='Low Current'
n['L5']='High Current'
n['L6']='High Voltage High Current'
n['L7']='High Voltage Low Current'
n['L8']='Low Voltage High Current'
n['L9']='Low Voltage Low Current'
n['L10']='Magnetic Fault '
n['L11']='Low Voltage Magnetic'
n['L12']='High Voltage Magnetic'
n['L13']='Low Current Magnetic'
n['L14']='High Current Magnetic'
n['L15']='Heat Diode Fault'
n['L16']='Low Voltage Heat Diode'
n['L17']='High Voltage Heat Diode'
n['L18']='Low Current Heat Diode'
n['L19']='High Current Heat Diode'
n['L20']='Heat Mosfet Fault'
n['L21']='Low Voltage Heat Mosfet'
n['L22']='High Voltage Heat Mosfet'
n['L23']='Low Current Heat Mosfet'
n['L24']='High Current Heat Mosfet'
n['L25']='Capacitor resistance'
n['L26']='Low Voltage Capacitor resistance'
n['L27']='High Voltage Capacitor resistance'
n['L28']='Low Current Capacitor resistance'
n['L29']='High Current Capacitor resistance'
n['L30']='Open Circuit Resitance'
n['L31']='Short Circuit Resistance'
n['L32']='Open Circuit Diode'
n['L33']='Short Circuit Diode'
n['L34']='Open Circuit Inductor'
n['L35']='Short Circuit Inductor'
n['L36']='Open Circuit Mosfet'
n['L37']='Short Circuit Mosfet'
n['L38']='Open Circuit Capacitor'


#
# concatenate all output train data
Ytrain=np.concatenate((nonf['Ytrain'],lvf['Ytrain'],hvf['Ytrain'],lcf['Ytrain'],
    hcf['Ytrain'], hvhcf['Ytrain'],hvlcf['Ytrain'],lvhcf['Ytrain'],lvlcf['Ytrain'],
    mag['Ytrain'],lvmag['Ytrain'],hvmag['Ytrain'],lcmag['Ytrain'],hcmag['Ytrain'],
    heatD['Ytrain'],lvoheatD['Ytrain'],hvoheatD['Ytrain'],lcoheatD['Ytrain'],hcoheatD['Ytrain'],
    heatmos['Ytrain'],lvoheatmos['Ytrain'],hvoheatmos['Ytrain'],lcoheatmos['Ytrain'],hcoheatmos['Ytrain'],
    cap['Ytrain'],lvcap['Ytrain'],hvcap['Ytrain'],lccap['Ytrain'],hccap['Ytrain'],
    ocf['Ytrain'],scf['Ytrain'],ocD['Ytrain'],scD['Ytrain'],ocL['Ytrain'],
    scL['Ytrain'],ocmos['Ytrain'],scmos['Ytrain'],occ['Ytrain']),axis=2)
    
nonf['Ytrain']=lvf['Ytrain']=hvf['Ytrain']=lcf['Ytrain']=hcf['Ytrain']= hvhcf['Ytrain']=hvlcf['Ytrain']=lvhcf['Ytrain']=lvlcf['Ytrain']=None
mag['Ytrain']=lvmag['Ytrain']=hvmag['Ytrain']=lcmag['Ytrain']=hcmag['Ytrain']=heatD['Ytrain']=lvoheatD['Ytrain']=hvoheatD['Ytrain']=lcoheatD['Ytrain']=hcoheatD['Ytrain']=None
heatmos['Ytrain']=lvoheatmos['Ytrain']=hvoheatmos['Ytrain']=lcoheatmos['Ytrain']=hcoheatmos['Ytrain']=cap['Ytrain']=lvcap['Ytrain']=hvcap['Ytrain']=lccap['Ytrain']=hccap['Ytrain']=None
ocf['Ytrain']=scf['Ytrain']=ocD['Ytrain']=scD['Ytrain']=ocL['Ytrain']=scL['Ytrain']=ocmos['Ytrain']=scmos['Ytrain']=occ['Ytrain']=None
#
# concatenate all output test data
Ytest=np.concatenate((nonf['Ytest'],lvf['Ytest'],hvf['Ytest'],lcf['Ytest'],
    hcf['Ytest'], hvhcf['Ytest'],hvlcf['Ytest'],lvhcf['Ytest'],lvlcf['Ytest'],
    mag['Ytest'],lvmag['Ytest'],hvmag['Ytest'],lcmag['Ytest'],hcmag['Ytest'],
    heatD['Ytest'],lvoheatD['Ytest'],hvoheatD['Ytest'],lcoheatD['Ytest'],hcoheatD['Ytest'],
    heatmos['Ytest'],lvoheatmos['Ytest'],hvoheatmos['Ytest'],lcoheatmos['Ytest'],hcoheatmos['Ytest'],
    cap['Ytest'],lvcap['Ytest'],hvcap['Ytest'],lccap['Ytest'],hccap['Ytest'],
    ocf['Ytest'],scf['Ytest'],ocD['Ytest'],scD['Ytest'],ocL['Ytest'],
    scL['Ytest'],ocmos['Ytest'],scmos['Ytest'],occ['Ytest']),axis=2)
    
nonf['Ytest']=lvf['Ytest']=hvf['Ytest']=lcf['Ytest']=hcf['Ytest']= hvhcf['Ytest']=hvlcf['Ytest']=lvhcf['Ytest']=lvlcf['Ytest']=None
mag['Ytest']=lvmag['Ytest']=hvmag['Ytest']=lcmag['Ytest']=hcmag['Ytest']=heatD['Ytest']=lvoheatD['Ytest']=hvoheatD['Ytest']=lcoheatD['Ytest']=hcoheatD['Ytest']=None
heatmos['Ytest']=lvoheatmos['Ytest']=hvoheatmos['Ytest']=lcoheatmos['Ytest']=hcoheatmos['Ytest']=cap['Ytest']=lvcap['Ytest']=hvcap['Ytest']=lccap['Ytest']=hccap['Ytest']=None
ocf['Ytest']=scf['Ytest']=ocD['Ytest']=scD['Ytest']=ocL['Ytest']=scL['Ytest']=ocmos['Ytest']=scmos['Ytest']=occ['Ytest']=None


#
# concatenate all batch train data
mbtrain=np.concatenate((nonf['mbtrain'],lvf['mbtrain'],hvf['mbtrain'],lcf['mbtrain'],
    hcf['mbtrain'], hvhcf['mbtrain'],hvlcf['mbtrain'],lvhcf['mbtrain'],lvlcf['mbtrain'],
    mag['mbtrain'],lvmag['mbtrain'],hvmag['mbtrain'],lcmag['mbtrain'],hcmag['mbtrain'],
    heatD['mbtrain'],lvoheatD['mbtrain'],hvoheatD['mbtrain'],lcoheatD['mbtrain'],hcoheatD['mbtrain'],
    heatmos['mbtrain'],lvoheatmos['mbtrain'],hvoheatmos['mbtrain'],lcoheatmos['mbtrain'],hcoheatmos['mbtrain'],
    cap['mbtrain'],lvcap['mbtrain'],hvcap['mbtrain'],lccap['mbtrain'],hccap['mbtrain'],
    ocf['mbtrain'],scf['mbtrain'],ocD['mbtrain'],scD['mbtrain'],ocL['mbtrain'],
    scL['mbtrain'],ocmos['mbtrain'],scmos['mbtrain'],occ['mbtrain']),axis=2)
    
nonf['mbtrain']=lvf['mbtrain']=hvf['mbtrain']=lcf['mbtrain']=hcf['mbtrain']= hvhcf['mbtrain']=hvlcf['mbtrain']=lvhcf['mbtrain']=lvlcf['mbtrain']=None
mag['mbtrain']=lvmag['mbtrain']=hvmag['mbtrain']=lcmag['mbtrain']=hcmag['mbtrain']=heatD['mbtrain']=lvoheatD['mbtrain']=hvoheatD['mbtrain']=lcoheatD['mbtrain']=hcoheatD['mbtrain']=None
heatmos['mbtrain']=lvoheatmos['mbtrain']=hvoheatmos['mbtrain']=lcoheatmos['mbtrain']=hcoheatmos['mbtrain']=cap['mbtrain']=lvcap['mbtrain']=hvcap['mbtrain']=lccap['mbtrain']=hccap['mbtrain']=None
ocf['mbtrain']=scf['mbtrain']=ocD['mbtrain']=scD['mbtrain']=ocL['mbtrain']=scL['mbtrain']=ocmos['mbtrain']=scmos['mbtrain']=occ['mbtrain']=None


#
# concatenate all batch test data
mbtest=np.concatenate((nonf['mbtest'],lvf['mbtest'],hvf['mbtest'],lcf['mbtest'],
    hcf['mbtest'], hvhcf['mbtest'],hvlcf['mbtest'],lvhcf['mbtest'],lvlcf['mbtest'],
    mag['mbtest'],lvmag['mbtest'],hvmag['mbtest'],lcmag['mbtest'],hcmag['mbtest'],
    heatD['mbtest'],lvoheatD['mbtest'],hvoheatD['mbtest'],lcoheatD['mbtest'],hcoheatD['mbtest'],
    heatmos['mbtest'],lvoheatmos['mbtest'],hvoheatmos['mbtest'],lcoheatmos['mbtest'],hcoheatmos['mbtest'],
    cap['mbtest'],lvcap['mbtest'],hvcap['mbtest'],lccap['mbtest'],hccap['mbtest'],
    ocf['mbtest'],scf['mbtest'],ocD['mbtest'],scD['mbtest'],ocL['mbtest'],
    scL['mbtest'],ocmos['mbtest'],scmos['mbtest'],occ['mbtest']),axis=2)
    
nonf['mbtest']=lvf['mbtest']=hvf['mbtest']=lcf['mbtest']=hcf['mbtest']= hvhcf['mbtest']=hvlcf['mbtest']=lvhcf['mbtest']=lvlcf['mbtest']=None
mag['mbtest']=lvmag['mbtest']=hvmag['mbtest']=lcmag['mbtest']=hcmag['mbtest']=heatD['mbtest']=lvoheatD['mbtest']=hvoheatD['mbtest']=lcoheatD['mbtest']=hcoheatD['mbtest']=None
heatmos['mbtest']=lvoheatmos['mbtest']=hvoheatmos['mbtest']=lcoheatmos['mbtest']=hcoheatmos['mbtest']=cap['mbtest']=lvcap['mbtest']=hvcap['mbtest']=lccap['mbtest']=hccap['mbtest']=None
ocf['mbtest']=scf['mbtest']=ocD['mbtest']=scD['mbtest']=ocL['mbtest']=scL['mbtest']=ocmos['mbtest']=scmos['mbtest']=occ['mbtest']=None


#
# concatenate all validation batch data
mbtest_V=np.concatenate((nonf['mbtest_V'],lvf['mbtest_V'],hvf['mbtest_V'],lcf['mbtest_V'],
    hcf['mbtest_V'], hvhcf['mbtest_V'],hvlcf['mbtest_V'],lvhcf['mbtest_V'],lvlcf['mbtest_V'],
    mag['mbtest_V'],lvmag['mbtest_V'],hvmag['mbtest_V'],lcmag['mbtest_V'],hcmag['mbtest_V'],
    heatD['mbtest_V'],lvoheatD['mbtest_V'],hvoheatD['mbtest_V'],lcoheatD['mbtest_V'],hcoheatD['mbtest_V'],
    heatmos['mbtest_V'],lvoheatmos['mbtest_V'],hvoheatmos['mbtest_V'],lcoheatmos['mbtest_V'],hcoheatmos['mbtest_V'],
    cap['mbtest_V'],lvcap['mbtest_V'],hvcap['mbtest_V'],lccap['mbtest_V'],hccap['mbtest_V'],
    ocf['mbtest_V'],scf['mbtest_V'],ocD['mbtest_V'],scD['mbtest_V'],ocL['mbtest_V'],
    scL['mbtest_V'],ocmos['mbtest_V'],scmos['mbtest_V'],occ['mbtest_V']),axis=2)
    
nonf['mbtest_V']=lvf['mbtest_V']=hvf['mbtest_V']=lcf['mbtest_V']=hcf['mbtest_V']= hvhcf['mbtest_V']=hvlcf['mbtest_V']=lvhcf['mbtest_V']=lvlcf['mbtest_V']=None
mag['mbtest_V']=lvmag['mbtest_V']=hvmag['mbtest_V']=lcmag['mbtest_V']=hcmag['mbtest_V']=heatD['mbtest_V']=lvoheatD['mbtest_V']=hvoheatD['mbtest_V']=lcoheatD['mbtest_V']=hcoheatD['mbtest_V']=None
heatmos['mbtest_V']=lvoheatmos['mbtest_V']=hvoheatmos['mbtest_V']=lcoheatmos['mbtest_V']=hcoheatmos['mbtest_V']=cap['mbtest_V']=lvcap['mbtest_V']=hvcap['mbtest_V']=lccap['mbtest_V']=hccap['mbtest_V']=None
ocf['mbtest_V']=scf['mbtest_V']=ocD['mbtest_V']=scD['mbtest_V']=ocL['mbtest_V']=scL['mbtest_V']=ocmos['mbtest_V']=scmos['mbtest_V']=occ['mbtest_V']=None

#
# concatenate all validation output data
Ytest_V=np.concatenate((nonf['Ytest_V'],lvf['Ytest_V'],hvf['Ytest_V'],lcf['Ytest_V'],
    hcf['Ytest_V'], hvhcf['Ytest_V'],hvlcf['Ytest_V'],lvhcf['Ytest_V'],lvlcf['Ytest_V'],
    mag['Ytest_V'],lvmag['Ytest_V'],hvmag['Ytest_V'],lcmag['Ytest_V'],hcmag['Ytest_V'],
    heatD['Ytest_V'],lvoheatD['Ytest_V'],hvoheatD['Ytest_V'],lcoheatD['Ytest_V'],hcoheatD['Ytest_V'],
    heatmos['Ytest_V'],lvoheatmos['Ytest_V'],hvoheatmos['Ytest_V'],lcoheatmos['Ytest_V'],hcoheatmos['Ytest_V'],
    cap['Ytest_V'],lvcap['Ytest_V'],hvcap['Ytest_V'],lccap['Ytest_V'],hccap['Ytest_V'],
    ocf['Ytest_V'],scf['Ytest_V'],ocD['Ytest_V'],scD['Ytest_V'],ocL['Ytest_V'],
    scL['Ytest_V'],ocmos['Ytest_V'],scmos['Ytest_V'],occ['Ytest_V']),axis=2)
    
nonf=lvf=hvf=lcf=hcf=hvhcf=hvlcf=lvhcf=lvlcf=None
mag=lvmag=hvmag=lcmag=hcmag=heatD=lvoheatD=hvoheatD=lcoheatD=hcoheatD=None
heatmos=lvoheatmos=hvoheatmos['Ytest_V']=lcoheatmos['Ytest_V']=hcoheatmos['Ytest_V']=cap['Ytest_V']=lvcap['Ytest_V']=hvcap['Ytest_V']=lccap['Ytest_V']=hccap['Ytest_V']=None
ocf=scf=ocD=scD=ocL=scL=ocmos=scmos=occ=None


b['Ytrain']='Ytrain shape: '+str(Ytrain.shape)
b['mbtrain']='mbtrain shape: '+str(mbtrain.shape)
b['Ytest']='Ytest shape: '+str(Ytest.shape)
b['mbtest']='mbtest shape: '+str(mbtest.shape)
b['Ytest_V']='Yvalidation shape: '+str(Ytest_V.shape)
b['mbtest_V']='mbvalidation shape: '+str(mbtest_V.shape)
print('Ytrain: ',Ytrain.shape,' Ytest: ',Ytest.shape, ' Yvali: ',Ytest_V.shape)
print('mbtrain: ',mbtrain.shape,' mbtest: ',mbtest.shape, ' mbvali: ',mbtest_V.shape)
tsec=time.time()

#
# create placeholders
X = tf.placeholder(tf.float32, [None,3, mb_size,1], name="X")
Y = tf.placeholder(tf.float32, [C, None], name="Y")
keep_prob=tf.placeholder(tf.float32)
learnrate=tf.placeholder(tf.float32)
numb=30

#
# initialize parameters
W1 = tf.get_variable("W1", [1, 20, 1, 5], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [5], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [3, 9, 5, 7], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [7], initializer = tf.zeros_initializer())
W3 = tf.get_variable("W3", [1, 8,7,9], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable("b3", [9], initializer = tf.zeros_initializer())
W4 = tf.get_variable("W4", [1, 9,9,11], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable("b4", [11], initializer = tf.zeros_initializer())
W5 = tf.get_variable("W5", [1, 8,11,13], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b5 = tf.get_variable("b5", [13], initializer = tf.zeros_initializer())
W6 = tf.get_variable("W6", [1, 8,13,15], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b6 = tf.get_variable("b6", [15], initializer = tf.zeros_initializer())
#
# forward propagation
Z1a = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'VALID')
#Z1b = tf.nn.bias_add(Z1a, b1)
P1 = tf.nn.elu(Z1a)
b['W1']='W1 shape: '+str(W1.shape)
b['P1']='P1 shape: '+str(P1.shape)

A1=tf.nn.max_pool(P1, ksize = [1,1,3,1], strides = [1,1,1,1], padding = 'VALID')

b['A1']='A1 shape: ' + str(A1.shape)

Z2a = tf.nn.conv2d(A1, W2, strides = [1,1,2,1], padding = 'VALID')
#Z2b = tf.nn.bias_add(Z2a, b2)
P2 = tf.nn.elu(Z2a)
b['W2']='W2 shape: '+str(W2.shape)
b['P2']='P2 shape: '+str(P2.shape)

Z3a = tf.nn.conv2d(P2, W3, strides = [1,1,1,1], padding = 'VALID')
#Z3b = tf.nn.bias_add(Z3a, b3)
P3 = tf.nn.elu(Z3a)
b['W3']='W3 shape: '+str(W3.shape)
b['P3']='P3 shape: '+str(P3.shape)

Z4a = tf.nn.conv2d(P3, W4, strides = [1,1,2,1], padding = 'VALID')
#Z4b = tf.nn.bias_add(Z4a, b4)
P4 = tf.nn.elu(Z4a)
b['W4']='W4 shape: '+str(W4.shape)
b['P4']='P4 shape: '+str(P4.shape)

Z5a = tf.nn.conv2d(P4, W5, strides = [1,1,2,1], padding = 'VALID')
#Z5b = tf.nn.bias_add(Z5a, b5)
P5 = tf.nn.elu(Z5a)
b['W5']='W5 shape: '+str(W5.shape)
b['P5']='P5 shape: '+str(P5.shape)

Z6a = tf.nn.conv2d(P5, W6, strides = [1,1,1,1], padding = 'VALID')
#Z6b = tf.nn.bias_add(Z6a, b6)
P6 = tf.nn.elu(Z6a)
b['W6']='W6 shape: '+str(W6.shape)
b['P6']='P6 shape: '+str(P6.shape)

#fully connected layers
P7 = tf.contrib.layers.flatten(P6)
b['P7']='flattened shape: '+str(P7.shape)

#P8d = tf.nn.dropout(P7, keep_prob)
P8 = tf.contrib.layers.fully_connected(P7, C, activation_fn=None)
b['D8']='dropout 8: '+str(100)
b['P8']='P8 shape: '+str(P8.shape)
SOFTmax = tf.nn.softmax(P8)
#
#cost function
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)       #regularization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = P8, labels = tf.transpose(Y)))
#+ 0.06 * sum(reg_variables)

optimizer = tf.train.AdamOptimizer(learning_rate = learnrate).minimize(cost)
costs = [] #keep track of cost

# to find classifier performance
accuracy = 0
correct_prediction = tf.equal(tf.argmax(SOFTmax, 1), tf.argmax(tf.transpose(Y), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initializer
init = tf.global_variables_initializer()

def random_mini_batches(mbtrain,Ytrain,mb_size,per,ruuns,numb,num_runs,rema,C):
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    if(ruuns==num_runs):
        mini_batch_X=np.zeros((rema,3,mb_size,1))
        mini_batch_Y=np.zeros((C,rema))
        for h in range(0,rema):
            mini_batch_X[h,:,:,:] = np.expand_dims(np.expand_dims(mbtrain[:,:,per[numb*ruuns+h]],axis=0), axis = 3)
            mini_batch_Y[:,h] =np.squeeze(Ytrain[:,:,per[numb*ruuns+h]],axis=1) 
    else:
        mini_batch_X=np.zeros((numb,3,mb_size,1))
        mini_batch_Y=np.zeros((C,numb))
        for h in range(0,numb):
            mini_batch_X[h,:,:,:] = np.expand_dims(np.expand_dims(mbtrain[:,:,per[numb*ruuns+h]],axis=0), axis = 3)
            mini_batch_Y[:,h] = np.squeeze(Ytrain[:,:,per[numb*ruuns+h]],axis=1)
    return mini_batch_X,mini_batch_Y

# run
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = mbtrain.shape[2]
        num_runs=math.floor(mbtrain.shape[2]/numb)-1
        per = list(np.random.permutation(num_minibatches))
        rema=num_minibatches%numb+numb
        
        for ruuns in range(0,num_runs):
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            mb_X,mb_Y=random_mini_batches(mbtrain,Ytrain,mb_size,per,ruuns,numb,num_runs,rema,C)
            _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: mb_X[:,:,:,:], Y: mb_Y[:,:],keep_prob : 0.95, learnrate: 0.0003*0.9**epoch})
            minibatch_cost += numb*temp_cost / num_minibatches       

        mb_X,mb_Y=random_mini_batches(mbtrain,Ytrain,mb_size,per,num_runs,numb,num_runs,rema,C)
        _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X: mb_X[:,:,:,:], Y: mb_Y[:,:],keep_prob : 0.95, learnrate: 0.0003*0.9**epoch})
        minibatch_cost += rema*temp_cost/num_minibatches
        # Print the cost every 50 epoch
        n['cost'+str(epoch)]=minibatch_cost
        b['cost'+str(epoch+1)]="Cost after epoch %i: %f" % (epoch+1, minibatch_cost)
        print(b['cost'+str(epoch+1)])
        #if epoch % 10 == 9:

    b['telapsed']='time elapsed during training: '+str(time.time()-tsec)+"\n"

    #save matrixes
    SW1,SW2,SW3,SW4,SW5,SW6,Sb1,Sb2,Sb3,Sb4,Sb5,Sb6=sess.run([W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6])

    SW7=sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'fully_connected/weights')])
    Sb7=sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'fully_connected/biases')])

    parameters = {"W1": SW1,
                  "W2": SW2,
                  "W3": SW3,
                  "W4": SW4,
                  "W5": SW5,
                  "W6": SW6,
                  "b7": Sb7,
                  "W7": SW7}
    #sci.savemat('mydata',parameters)
    np.save('output' + filename,parameters)
    minibatches=None    #remove data
    mbtrain=None        #remove more data

      #test accuracy
    A = np.expand_dims(np.swapaxes(np.swapaxes(mbtest,2,0),1,2), axis = 3)
    B =np.squeeze(np.swapaxes(Ytest,1,2),axis=2)
    Ytest=None
    mbtest=None

    """
    b['F0']='Test '+n['L0']+" Accuracy: "+str(accuracy.eval(feed_dict={X: A[:,:,:,:], Y: B[:,:], keep_prob : 1.0,keep_prob2 : 1.0,learnrate: 0}))
    for j in range (0,C):
        b['F'+str(j+1)]='Test '+n['L'+str(j+1)]+" Accuracy:"+str(accuracy.eval(feed_dict={X: A[n['T'+str(j)]:n['T'+str(j+1)],:,:,:], Y: B[:,n['T'+str(j)]:n['T'+str(j+1)]], keep_prob : 1.0,keep_prob2 : 1.0,learnrate: 0}))
    """
    
    MT=100      #test measurements at time
    T1=0
    T2=n['T'+str(C)]
    acc=np.zeros((1,T2-T1))
    L=math.floor((T2-T1)/MT)-1
    for m in range(0,L):
        acc[:,m*MT:(1+m)*MT]=accuracy.eval(feed_dict={X: A[T1:T1+MT,:,:,:], Y: B[:,T1:T1+MT], keep_prob : 1.0,learnrate: 0})
        T1=T1+MT
    acc[:,T1:T2]=accuracy.eval(feed_dict={X: A[T1:T2,:,:,:], Y: B[:,T1:T2], keep_prob : 1.0,learnrate: 0})
    b['T0'] = 'Test '+n['L0']+" Accuracy: "+ str(np.mean(acc))
    acc=None
    
    for j in range (0,C):
        T1=n['T'+str(j)]
        T2=n['T'+str(j+1)]
        acc=np.zeros((1,T2-T1))
        L=math.floor((T2-T1)/MT)
        if((T2-T1)%MT==0):
            L=L-1
        for m in range(0,L):
            acc[:,m*MT:(1+m)*MT]=accuracy.eval(feed_dict={X: A[T1:T1+MT,:,:,:], Y: B[:,T1:T1+MT], keep_prob : 1.0,learnrate: 0})
            T1=T1+MT
        acc[:,L*MT:T2-n['T'+str(j)]]=accuracy.eval(feed_dict={X: A[T1:T2,:,:,:], Y: B[:,T1:T2], keep_prob : 1.0,learnrate: 0})
        b['T'+str(j+1)] = 'Test '+n['L'+str(j+1)]+" Accuracy: "+ str(np.mean(acc))
        acc=None

    #validation accuracy
    A = np.expand_dims(np.swapaxes(np.swapaxes(mbtest_V,2,0),1,2), axis = 3)
    B = np.squeeze(np.swapaxes(Ytest_V,1,2),axis=2)
    Ytest_V=None
    mbtest_V=None

    """
    b['G0']='Validation '+n['L0']+" Accuracy: "+str(accuracy.eval(feed_dict={X: A[:,:,:,:], Y: B[:,:], keep_prob : 1.0,keep_prob2 : 1.0,learnrate: 0}))
    for j in range (0,C):
        b['G'+str(j+1)]='Validation '+n['L'+str(j+1)]+" Accuracy: "+str(accuracy.eval(feed_dict={X: A[n['TV'+str(j)]:n['TV'+str(j+1)],:,:,:], Y: B[:,n['TV'+str(j)]:n['TV'+str(j+1)]], keep_prob : 1.0,keep_prob2 : 1.0,learnrate: 0}))
    """

    MT=100      #test measurements at time
    T1=0
    T2=n['TV'+str(C)]
    acc=np.zeros((1,T2-T1))
    L=math.floor((T2-T1)/MT)
    if((T2-T1)%MT==0):
        L=L-1    
    for m in range(0,L):
        acc[:,m*MT:(1+m)*MT]=accuracy.eval(feed_dict={X: A[T1:T1+MT,:,:,:], Y: B[:,T1:T1+MT], keep_prob : 1.0,learnrate: 0})
        T1=T1+MT
    acc[:,T1:T2]=accuracy.eval(feed_dict={X: A[T1:T2,:,:,:], Y: B[:,T1:T2], keep_prob : 1.0,learnrate: 0})
    b['V0'] = 'Validation '+n['L0']+" Accuracy: "+ str(np.mean(acc))
    acc=None
    
    for j in range (0,C):
        T1=n['TV'+str(j)]
        T2=n['TV'+str(j+1)]
        acc=np.zeros((1,T2-T1))
        L=math.floor((T2-T1)/MT)-1
        for m in range(0,L):
            acc[:,m*MT:(1+m)*MT]=accuracy.eval(feed_dict={X: A[T1:T1+MT,:,:,:], Y: B[:,T1:T1+MT], keep_prob : 1.0,learnrate: 0})
            T1=T1+MT
        acc[:,L*MT:T2-n['TV'+str(j)]]=accuracy.eval(feed_dict={X: A[T1:T2,:,:,:], Y: B[:,T1:T2], keep_prob : 1.0,learnrate: 0})
        b['V'+str(j+1)] = 'Validation '+n['L'+str(j+1)]+" Accuracy: "+ str(np.mean(acc))
        acc=None

    print('training time elapsed in seconds: ',b['telapsed'])
    n['G1']='bo-'
    n['G2']='go-'
    n['G3']='ro-'
    n['G4']='ko-'
    n['G5']='mo-'
    n['G6']='bs-'
    n['G7']='gs-'
    n['G8']='rs-'
    n['G9']='ks-'
    
    n['G10']='b^-'
    n['G11']='g^-'
    n['G12']='r^-'
    n['G13']='k^-'
    n['G14']='m^-'
    
    n['G15']='bx-'
    n['G16']='gx-'
    n['G17']='rx-'
    n['G18']='kx-'
    n['G19']='mx-'
    
    n['G20']='bD-'
    n['G21']='gD-'
    n['G22']='rD-'
    n['G23']='kD-'
    n['G24']='mD-'
    
    n['G25']='b+-'
    n['G26']='g+-'
    n['G27']='r+-'
    n['G28']='k+-'
    n['G29']='m+-'
    
    n['G30']='bp-'
    n['G31']='gp-'
    n['G32']='rp-'
    n['G33']='kp-'
    n['G34']='mp-'
    n['G35']='b*-'
    n['G36']='g*-'
    n['G37']='r*-'
    n['G38']='k*-'

    f= open(filename+'.txt',"w+")
    for item in b:
        print(b[item])
        f.write("%s\n" % b[item])
    f.close()
    

    #import os
    savepath = os.path.abspath('C:/Users/Tesla/Documents/Ronni Zahra/CNNnew/'+filename)
    """
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        print("Home directory %s was created." %savepath)
    print(savepath)
    """
    #
    #graphs
    if(num_epochs>3):
        costss=np.zeros((num_epochs,1))
        for k in range(0,num_epochs):
            costss[k,0]=n['cost'+str(k)]
        fig2 = plt.figure(figsize=(18,9))
        cost = fig2.add_axes([0.1, 0.1, 0.9, 0.9])
        cost.plot(costss)
        cost.set_title(filename+': cost')
        cost.set_ylabel('Cost')
        cost.set_xlabel('Epochs')
        fig2.savefig(savepath+'cost.pdf',dpi=fig2.dpi)
    
    #
    #softmax
    MT=100
    for m in range(0,C):
        T1=n['TV'+str(m)]
        T2=n['TV'+str(m+1)]
        graph=np.zeros((T2-T1,C))
        L=math.floor((T2-T1)/MT)-1 
        for h in range(0,L):
            graph[h*MT:(1+h)*MT,:]=SOFTmax.eval(feed_dict={X: A[T1:T1+MT,:,:,:], Y: B[:,T1:T1+MT], keep_prob : 1.0,learnrate: 0})
            T1=T1+MT
        graph[L*MT:T2-n['TV'+str(m)],:]=SOFTmax.eval(feed_dict={X: A[T1:T2,:,:,:], Y: B[:,T1:T2], keep_prob : 1.0,learnrate: 0})

        fig = plt.figure(figsize=(18,9))
        ax = fig.add_axes([0.05, 0.05, 0.75, 0.75])
        for j in range(0,C):
            ax.plot(graph[:,j],n['G'+str(j+1)], label=n['L'+str(j+1)])
        ax.set_title(filename+' :'+n['L'+str(m+1)])
        ax.set_ylabel('Softmax validation probability')
        ax.set_xlabel('Minibatch Sample')
        ax.legend(bbox_to_anchor =(1, 1.25))
        fig.savefig(savepath+n['L'+str(m+1)]+'.png',dpi=fig.dpi)
