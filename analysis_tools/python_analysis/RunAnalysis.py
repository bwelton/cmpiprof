import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile
from PercentImbalanced import PercentImbalance
from BuildBestCase import BestCaseRebalancing
from TrueCostModel import TrueCostModel

mypath = sys.argv[1]
allFiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

## Grab only the log files
logFiles = []
for x in allFiles:
    if "bappdata" in x:
        logFiles.append(x)

## Import the data
data = []
for x in logFiles:
    print x
    tmp = ImportFile(x)
    tmp.ReadFile()
    data.append(tmp)

## Run percent imbalanced analysis
pi = PercentImbalance(data)
pi.Calculate(join(sys.argv[2], "PercentImbalanced.csv"))

bc = BestCaseRebalancing(data)
bc.SimpleAverage()

tc = TrueCostModel(data)
tc.CalculateCosts()
