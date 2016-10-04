import struct
import sys
import os
from os import listdir
from os.path import isfile, join

indir = sys.argv[1]
allFiles = [join(indir, f) for f in listdir(indir) if isfile(join(indir, f))]

## Grab only the log files
logFiles = []
for x in allFiles:
    if ".out" in x:
        logFiles.append(x)

for x in logFiles:
	rep = x.split(".")[0] + ".bappdata"
	os.system("./convertbinary " + x + " " + rep)

if len(sys.argv) == 4:
	os.system("python RunAnalysis.py " + sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3])
else:
	os.system("python RunAnalysis.py " + sys.argv[1] + " " + sys.argv[2])