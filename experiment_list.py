import numpy as np
import io
import os
import math
import multiprocessing as mp


from ABM_31122020_sol1_random import songModel

def use_songModel(filename, FGS, MGS, modelMode, memory_conservatism, n_of_immigrants):
	songModel(filename=filename, FGS=FGS, MGS=MGS, modelMode = modelMode, memory_conservatism = memory_conservatism, n_of_immigrants = n_of_immigrants)
	return 1

modelMods = ['revolution'] 
memory_conservatisms = [0.1,0.5,0.9]
densities = [0.001,0.01,0.1,1,5]
n_of_immigrants = [1,5,10]



exp = []
for x in range(0,100):
	for modelMod in modelMods:
		for memory_conservatism in memory_conservatisms:
			for n_of_immigrant in n_of_immigrants:
				for density in densities:
					expNum = f'ABM1_SimN_{x}_' + f'{modelMod}_' + f'c_{memory_conservatism}_' + f'n_imm_{n_of_immigrant}_' + f'd_{density}_'
					MGS = math.sqrt((20)/(density*math.pi))
					FGS = math.sqrt((20+n_of_immigrant)/(density*math.pi))
					exp.append((expNum, FGS, MGS, modelMod, memory_conservatism, n_of_immigrant))

if __name__ == '__main__':
	pool = mp.Pool(processes=64) # here you can change the number of cores
	pool.starmap(use_songModel, exp)
	pool.close()
