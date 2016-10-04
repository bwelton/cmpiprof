import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile

##
# Calculate the percent imbalanced between nodes
class HighLevelStatistics:
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    ## Simple averaging of all nodes, absolute best case rebalancing.
    # No transfer cost associated with this. 
    # This calculation is essentially CPUTime - GPUTime + AvgGPUTime for every
    # phase
    def HLStats(self):
        np.seterr(divide="warn")
        phases = self._data[0].GetPhases()
        expectedSize = phases.size

        percent = []

        for x in self._data:
            percent.append(np.sum(x.GetPhases()["GPUTime"]) / np.sum(x.GetPhases()["TotalTime"]))

        print "Percent of time in GPU: " + str(sum(percent) / float(len(percent)))