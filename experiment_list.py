import numpy as np
import io
import os
import math
import multiprocessing as mp
import itertools

from ABM_16112020_sol1_random import songModel

def use_songModel(filename, FGS, MGS, modelMode, memory_conservatism, n_of_immigrants):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode, memory_conservatism = memory_conservatism, n_of_immigrants = n_of_immigrants)
	return 1

modelMods = ['distance','revolution'] #,'weightedEditsD','weightedEditsN']
memory_conservatisms = [0.1,0.5,0.9]
densities = [0.001,0.1,1,5]
n_of_immigrants = [1,5,10]

n_combinations = list(itertools.product(modelMods, memory_conservatisms,densities,n_of_immigrants))
n_simulations = len(n_combinations)*100

exp = []
for x in range(0,n_simulations):
	for modelMod in modelMods:
		for memory_conservatism in memory_conservatisms:
			for n_of_immigrant in n_of_immigrants:
				for density in densities:
					expNum = 'ExperimentN' + str(x) + '_'
					MGS = math.sqrt((20)/(density*math.pi))
					FGS = math.sqrt((20+n_of_immigrant)/(density*math.pi))
					exp.append((expNum, FGS, MGS, modelMod, memory_conservatism, n_of_immigrant))

if __name__ == '__main__':
	pool = mp.Pool(processes=48)
	pool.starmap(use_songModel, exp)
	pool.close()
