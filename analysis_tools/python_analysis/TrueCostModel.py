import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile
from GPUTransferCost import GPUTransferCost
from TransmissionDelayEstimator import TransmissionDelayEstimator

class TrueCostModel: 
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    def CalculateCosts(self):
		pass