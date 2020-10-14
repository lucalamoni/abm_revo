import numpy as np
import io
import os
import math
import multiprocessing as mp

from ABM_14082019_python3_refactored import songModel

def use_songModel(filename, FGS, MGS, modelMode):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode)
	return 1

modelMods = ['revolution', 'novelty']#,'weightedEditsD','weightedEditsN']

densities = [0.001,0.01,0.1,1.0,3.0,5.0]
exp = []
for x in range(0,2):
	for modelMod in modelMods:
		for density in densities:
			expNum = 'ExperimentN' + str(x) + str(density)
			MGS = math.sqrt(33/(density*math.pi))
			FGS = MGS
			exp.append((expNum, FGS, MGS, modelMod))

if __name__ == '__main__':
	pool = mp.Pool(processes=4)
	pool.starmap(use_songModel, exp)
	pool.close()
