import sys
import os
from os import listdir
from os.path import isfile, join

PATH = sys.argv[1]

## Calculate Avg PI:
f = open(join(PATH,"PercentImbalanced.csv"),"r")
data = f.readlines()
f.close()
data = data[1:-1]
avgPI = 0.0
for x in data:
	avgPI +=  float(x.split(",")[1])
avgPI =  avgPI / len(data)
phaseCount =  len(data)
## Calculate Avg Phase Length
f = open(join(PATH,"TrueCostModel.csv"),"r")
data = f.readlines()
f.close()
data = data[1:]
avgPhaseLength = 0.0
totalBefore = 0.0 
totalAfter = 0.0
gpuBefore = 0.0
gpuAfter = 0.0
percentgpuTimeSaved = 0.0
for x in data:
	avgPhaseLength +=  float(x.split(",")[0])
	totalBefore += float(x.split(",")[0])
	gpuBefore += float(x.split(",")[1])
	gpuAfter += float(x.split(",")[2])
	totalAfter += float(x.split(",")[3])
avgPhaseLength = avgPhaseLength / len(data)
percentSavedTime = (totalBefore - totalAfter) / totalBefore
percentgpuTimeSaved = (gpuBefore - gpuAfter) / gpuBefore

print str(phaseCount) + "," + str(avgPhaseLength) + "," + str(avgPI) + "," + str(percentSavedTime) + "," + str(percentgpuTimeSaved)