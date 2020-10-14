#import panda as pd 
import numpy as np
#import collections, re
#import cProfile
#import pstats
import io
import os
import math

from use_songModel import use_songModel
#execfile('RandomExcelSheets_LL.py')
os.chdir('C:/Users/Luca/ABM_master')
#exec(open('C:/Users/Luca/REVO_ABM/ABM_14082019_python3_refactored.py').read())
modelMod = ['revolution']#,'novelty','weightedEditsD','weightedEditsN']

density = [0.001,0.01,0.1,1.0,3.0,5.0]
DF = {}


exp = []
for x in range(0,2):
	#execfile('RandomExcelSheets_LL_mod.py')
	#for a in range(0,len(modelMod)):
	for b in range(0,len(density)):
		expNum = 'ExperimentN' + str(x) + str(density[b])
		#DF[expNum] = [modelMod[a]]
		MGS = math.sqrt(33/(density[b]*math.pi))
		FGS = MGS

		exp.append((expNum, FGS, MGS))


#def use_songModel(modelMode, filename, FGS, MGS):
	#songModel(modelMode = modelMode, filename = filename, FGS = FGS, MGS = MGS, i = 2000 , iSave = 500, mRuns = 1)
	#return 1

import multiprocessing as mp

#pool = mp.Pool(processes=4)
#pool.starmap(use_songModel, exp)
#pool.close()



if __name__ == '__main__':
	pool = mp.Pool(processes=4)
	pool.starmap(use_songModel, exp)
	pool.close()
