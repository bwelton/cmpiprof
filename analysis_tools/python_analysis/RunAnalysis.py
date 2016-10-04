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
from NormalizePhases import NormalizeInput
from EvenDistribution import EvenDistribution
from HighLevelStatistics import HighLevelStatistics

mypath = sys.argv[1]
allFiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

## Grab only the log files
logFiles = []
for x in allFiles:
    if "bappdata" in x:
        logFiles.append(x)

IORows = []
if len(sys.argv) == 4:
	IORows =  sys.argv[3].split(",")

print "Skipping the following rows"
print IORows

## Import the data
data = []
for x in logFiles:
    print x
    tmp = ImportFile(x)
    tmp.ReadFile()
    tmp.RemoveIOPhases(IORows)
    data.append(tmp)

## Normalize the input
norm = NormalizeInput(data)
norm.NormInput()

## Run percent imbalanced analysis
pi = PercentImbalance(data)
pi.Calculate(join(sys.argv[2], "PercentImbalanced.csv"))

bc = BestCaseRebalancing(data)
bc.SimpleAverage()

tc = TrueCostModel(data)
tc.CalculateCosts(join(sys.argv[2], "TrueCostModel.csv"))

hs = HighLevelStatistics(data)
hs.HLStats()

#ed = EvenDistribution(data)
#ed.RunEvenDistribution(join(sys.argv[2], "EvenDistribution.csv"))
