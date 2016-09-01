import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile

class GPUTransferCost:
    def __init__(self):
        self.dt = np.dtype([("Size",int,1),
            ("MemCreate",float,1),
            ("ToGPU",float,1),
            ("FromGPU",float,1)])
        self._ref = np.zeros((8), dtype=self.dt)
        self._ref["Size"] = [100000,1000000,5000000,10000000,20000000,40000000,80000000,100000000]
        self._ref["MemCreate"] = [0.000155,0.000126,0.000116,0.000127,0.000138,0.000185,0.000250,0.000243]
        self._ref["ToGPU"] = [0.000069,0.000411,0.001516,0.002809,0.005641,0.011146,0.022241,0.027750]
        self._ref["FromGPU"] = [0.000065,0.000375,0.001082,0.002031,0.004297,0.009709,0.020883,0.026491]

    def EstimateCost(self,data):
        out = np.zeros((data.size), dtype=self.dt)
        for x in range(0,data.size):
            found = False
            for i in range(0,len(self._ref["Size"])):
                if data[x] < self._ref["Size"][i]:
                    out["Size"][x] = data[x]
                    out["MemCreate"][x] = data[x] * (self._ref["MemCreate"][i] / self._ref["Size"][i])
                    out["ToGPU"][x] = data[x] * (self._ref["ToGPU"][i] / self._ref["Size"][i])
                    out["FromGPU"][x] = data[x] * (self._ref["FromGPU"][i] / self._ref["Size"][i])
                    found = True
                    break
            if found == False:
                out["Size"][x] = data[x]
                out["MemCreate"][x] = data[x] * (self._ref["MemCreate"][-1] / self._ref["Size"][-1])
                out["ToGPU"][x] = data[x] * (self._ref["ToGPU"][-1] / self._ref["Size"][-1])
                out["FromGPU"][x] = data[x] * (self._ref["FromGPU"][-1] / self._ref["Size"][-1])       
                #print "WE COULD NOT FIND THIS VALUE IN TABLE - Assume larger than 100MB"
        return out

