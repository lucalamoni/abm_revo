import numpy as np
import io
import os
import math
import multiprocessing as mp

from ABM_14082019_python3_refactored import songModel

def use_songModel(filename, FGS, MGS, modelMode):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode)
	return 1

modelMods = ['revolution'] #,'weightedEditsD','weightedEditsN']

densities = [1.0]
exp = []
for x in range(0,200):
	for modelMod in modelMods:
		for density in densities:
			expNum = str(modelMod) + '_Sec1_2IA_ExperimentN' + str(x) + '_' + str(density) + '_'
			MGS = math.sqrt(52/(density*math.pi))
			FGS = MGS
			exp.append((expNum, FGS, MGS, modelMod))

if __name__ == '__main__':
	pool = mp.Pool(processes=48)
	pool.starmap(use_songModel, exp)
	pool.close()
