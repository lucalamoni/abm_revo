import os
os.chdir('C:/Users/Luca/ABM_master')
exec(open('C:/Users/Luca/ABM_master/ABM_14082019_python3_refactored.py').read())

def use_songModel(filename, FGS, MGS):
	songModel(filename = filename, FGS = FGS, MGS = MGS)
	return 1