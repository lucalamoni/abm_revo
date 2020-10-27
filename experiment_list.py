import numpy as np
import io
import os
import math
import multiprocessing as mp

from ABM_14082019_python3_refactored import songModel

def use_songModel(filename, FGS, MGS, modelMode, memory_conservatism):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode, memory_conservatism = memory_conservatism)
	return 1

modelMods = ['revolution'] #,'weightedEditsD','weightedEditsN']
memory_conservatisms = [0.1,0.5,0.9]
densities = [0.001,1.0]
exp = []
for x in range(0,200):
	for modelMod in modelMods:
		for memory_conservatism in memory_conservatisms:
			for density in densities:
				expNum = str(modelMod) + '_Sec2_ExperimentN' + str(x) + '_' + str(density) + '_memo_conserv_' + str(memory_conservatism) + '_'
				MGS = math.sqrt(53/(density*math.pi))
				FGS = MGS
				exp.append((expNum, FGS, MGS, modelMod, memory_conservatism))

if __name__ == '__main__':
	pool = mp.Pool(processes=48)
	pool.starmap(use_songModel, exp)
	pool.close()
