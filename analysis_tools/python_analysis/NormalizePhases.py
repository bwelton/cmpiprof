import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile

class NormalizeInput:
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    def NormInput(self):
        print "****** Normalizing Input Phase Counts *********"
        ## Determine Maximum phase length
        maxPhaseCount = 0
        for x in self._data:
            tmp = len(x.GetPhases())
            if tmp > maxPhaseCount:
                maxPhaseCount = tmp
        print "- Setting phase counts to " + str(maxPhaseCount)
        for x in range(0, len(self._data)):
            self._data[x].FixPhase(maxPhaseCount, ident=x)
        print "****** Completed Phase Normalization ******"