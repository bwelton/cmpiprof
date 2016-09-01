import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile

class TransmissionDelayEstimator:

    def __init__(self):
        # 10 mbit (cisco): 0.001200
        # 100 Mbit (Cisco): 120 Microseconds -> 0.000120
        # 1GBit (Cisco): 12 Microseconds -> 0.000012
        # 10gbit (cisco): 1.2 Microseconds -> 0.0000012
        # Titan Min Delay: 0.00000127 
        # Titan Max Delay: 0.00000388
        # http://www.nersc.gov/users/computational-systems/retired-systems/hopper/configuration/interconnect/
        self.dt = np.dtype([("10mbit",float,1),
             ("100mbit",float,1),
             ("1gbit",float,1),
             ("10gbit",float,1),
             ("Titan_Nearest", float, 1),
             ("Titan_Furthest",float,1)])
        self._delays = np.zeros((1), dtype=self.dt)
        self._delays["10mbit"][0] = 0.001200
        self._delays["100mbit"][0] = 0.000120
        self._delays["1gbit"][0] = 0.000012
        self._delays["10gbit"][0] = 0.0000012
        self._delays["Titan_Nearest"][0] = 0.00000127
        self._delays["Titan_Furthest"][0] = 0.00000388

        dt2 = np.dtype([("10mbit",int,1),
             ("100mbit",int,1),
             ("1gbit",int,1),
             ("10gbit",int,1),
             ("Titan_Nearest", int, 1),
             ("Titan_Furthest",int,1)])
        self._transmissionSpeed = np.zeros((1), dtype=dt2)
        self._transmissionSpeed["10mbit"][0] = 1250000
        self._transmissionSpeed["100mbit"][0] = 12500000
        self._transmissionSpeed["1gbit"][0] =  125000000
        self._transmissionSpeed["10gbit"][0] =  1250000000
        self._transmissionSpeed["Titan_Nearest"][0] =  8600000000
        self._transmissionSpeed["Titan_Furthest"][0] =  8600000000
        pass

    def EstimateDelay(self, dataSize):
        values = np.zeros((dataSize.size), dtype=self.dt)
        values["10mbit"] = dataSize * self._transmissionSpeed["10mbit"][0] + self._delays["10mbit"][0]  
        values["100mbit"] = dataSize * self._transmissionSpeed["100mbit"][0] + self._delays["100mbit"][0]
        values["1gbit"] = dataSize * self._transmissionSpeed["1gbit"][0] + self._delays["1gbit"][0]
        values["10gbit"] = dataSize * self._transmissionSpeed["10gbit"][0] + self._delays["10gbit"][0]
        values["Titan_Nearest"] = dataSize * self._transmissionSpeed["Titan_Nearest"][0] + self._delays["Titan_Nearest"][0]
        values["Titan_Furthest"] = dataSize * self._transmissionSpeed["Titan_Furthest"][0] + self._delays["Titan_Furthest"][0]
        return values




