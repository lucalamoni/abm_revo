import numpy as np
import io
import os
import math
import multiprocessing as mp

from ABM_07112020_sol1_random_LR_memory_and_CM_changed import songModel

def use_songModel(filename, FGS, MGS, modelMode, memory_conservatism, n_of_immigrants):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode, memory_conservatism = memory_conservatism, n_of_immigrants = n_of_immigrants)
	return 1

modelMods = ['revolution'] #,'weightedEditsD','weightedEditsN']
memory_conservatisms = [0.1,0.5,0.9]
densities = [3]
n_of_immigrants = [1,5]
exp = []
for x in range(0,100):
	for modelMod in modelMods:
		for memory_conservatism in memory_conservatisms:
			for n_of_immigrant in n_of_immigrants:
				for density in densities:
					expNum = str(modelMod) + '_ExperimentN' + str(x) + '_' + str(n_of_immigrant) + '_memo_conserv_' + str(memory_conservatism) + '_'
					MGS = math.sqrt((30+n_of_immigrant)/(density*math.pi))
					FGS = MGS
					exp.append((expNum, FGS, MGS, modelMod, memory_conservatism, n_of_immigrant))

if __name__ == '__main__':
	pool = mp.Pool(processes=48)
	pool.starmap(use_songModel, exp)
	pool.close()
