# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:34:22 2016

@author: pi
"""

import numpy as np
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)

t=np.loadtxt("/home/pi/Downloads/td.csv")
l=np.loadtxt("/home/pi/Downloads/labels.csv")
x=np.array(t)
y=np.array(l)

xr=x.reshape(-1,1)

from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(xr,y)
f=np.matrix('-0.00018683;1')
#print f
import struct 
from socket import *
#import pickle

SIZE = 1024      # packet size

hostName = gethostbyname('172.17.27.176')

mySocket  = socket( AF_INET, SOCK_DGRAM )
mySocket.bind((hostName,17725))

repeat = True
while repeat:
    (data,addr) = mySocket.recvfrom(SIZE)       
    #print(len(data))
    data = struct.unpack('<dd',data)
    #print data  
    #dat = pickle.loads(str(data))
    dats = np.array(data)
    dr = dats.reshape(1,-1)
    A=np.array([tuple(i) for i in dr])    
    #print (np.transpose(dats))  
    #print (np.transpose(m).shape)
    #print(np.transpose(dats).shape)
    #dats=np.array([[data[:,0]],[data[:,1]]])
    #print (dr.shape)
    #ft=np.transpose(f)    
    inp=np.dot(A,f)
    #print inp
    import time
    start = time.time()
    y_pred = clf.predict(inp)
    print start
    print time.time() - start
    print y_pred
    if y_pred == 1:
       GPIO.output(21, True)
    else:
       GPIO.output(21, False)
    