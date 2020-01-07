import numpy as np
import random
import math
import time as tt
import optimization_master as opt
import planet_sampler_pop


def make_stats(result_file, ps, IWA, OWA, min_radius, max_radius, min_flux, max_flux, wavelength = 10, exptime_default = 35000):


	exptime_default = 35000.

	lam_F560W = 5.6E-6 # meters
	lam_F1000W = 10E-6 # meters
	lam_F1500W = 15E-6 # meters
	sensF560W = 0.16 # micro-Janskys
	sensF1000W = 0.54 # micro-Janskys
	sensF1500W = 1.39 # micro-Janskys

	if(wavelength == 5.6):
		sens_limit = sensF560W
		iwa_factor = (lam_F560W/lam_F1000W)
		used_flux = 8
	elif(wavelength == 10):
		sens_limit = sensF1000W
		used_flux = 9
		iwa_factor = 1
	elif(wavelength == 15):
		sens_limit = sensF1500W
		used_flux = 10
		iwa_factor = (lam_F1500W/lam_F1000W)
	else:
		print("Wrong wavelength band.")
	#We first get the parameters
	f = open(result_file, "r")
	stars_observed = []
	obs_times = []
	lines = f.readlines()
	lines.pop(0)
	for line in lines:
	    line = line.split("\t")
	    #We are only interested in stars that were observed at least once for the statistics
	    if(int(line[1]) > 0):
	        #If the star was observed, then we keep its index and its exposure time
	        stars_observed.append(int(line[0]))
	        obs_times.append(int(line[1]))
	f.close()
	
	
	obs_time_arr = []
	for it, star in enumerate(ps.snumber):
		if star in stars_observed:
			obs_time_arr.append(obs_times[stars_observed.index(star)])
		else:
			obs_time_arr.append(0)

	obs_time_arr = np.array(obs_time_arr)

	stars_observed = np.array(stars_observed)
	obs_times = np.array(obs_times)
	#We now want an array with all the

	#We create the array containing all the statistics and info about the planets
	data = np.vstack([ps.snumber, np.zeros(len(ps.snumber)), ps.a, ps.ang_sep, ps.Rp, ps.F_inc, ps.sdist, ps.period, ps.F560W, ps.F1000W, ps.F1500W, obs_time_arr])
	stype_arr = ps.stype
	#We remove all the planets for which the host star is not observed at all
	star_mask = np.isin(ps.snumber, stars_observed)
	data = data[:,star_mask]
	stype_arr = stype_arr[star_mask]

	#We now need to keep only the detectable planets
	#To this end, we need to compute the sensitivity that we achieve for all the stars by using their obs time
	#Once we have this, we remove all the planets whose flux is smaller than this sensitivity value for the concerned star

	lim_sens = sens_limit / np.sqrt(obs_times/exptime_default)
	#lim_sens is now the array containing the max sensitivity that we reach for each of the star
	#We create a new mask for the flux
	lim_sens_planets = np.ones(len(data[0]))
	for it, star in enumerate(stars_observed):
	    #lim_sens_planets is the same size as data, and each entry correspond to the max sensitivity for the planet
	    #depending on its host star
	    lim_sens_planets[data[0] == star] = lim_sens[it]
	#We can now create a mask that removes all the planets which are below the limit sensitivity for their star
	flux_mask = data[used_flux] > lim_sens_planets
	ang_mask = data[3] >= IWA*iwa_factor 
	owa_mask = data[3] <= OWA

	detection_mask =  owa_mask #& flux_mask & ang_mask 
	#We remove all the undetected planets
	data = data[:, detection_mask]

	stype_arr = stype_arr[detection_mask]

	#At this point, we have all the planets that we achieve to detect, however we are not removing the non 
	#habitable ones

	
	min_radius_mask = data[4] > min_radius
	max_radius_mask = data[4] < max_radius
	min_flux_mask = data[5] > min_flux
	max_flux_mask = data[5] < max_flux
	
	habitable_mask = max_radius_mask & min_radius_mask & min_flux_mask & max_flux_mask 

	#We now remove all the non habitable planets
	data_hab = data[:, habitable_mask]
	stype_arr_hab = stype_arr[habitable_mask]
	#We write the results for habitable planets in a file
	logfile = open("Stats/stats_habitable_" + result_file, "w")
	logfile.write('nstar\tstype\ta\tang_sep\tRp\tFinc\tdist\tPorb\tF560W\tF1000W\tF1500W\tobs_time\n')
	for i in range(len(data_hab[0])):
		logfile.write(str(data_hab[0][i]) + "\t" + str(stype_arr_hab[i]) + "\t" + str(data_hab[2][i]) + "\t" + str(data_hab[3][i]) + "\t" +  str(data_hab[4][i]) + "\t"+ str(data_hab[5][i])+ "\t"  + str(data_hab[6][i]) + "\t"+ str(data_hab[7][i])+ "\t"+ str(data_hab[8][i])+ "\t"+ str(data_hab[9][i])+ "\t"+ str(data_hab[10][i]) + "\t" + str(data_hab[11][i]) +"\n")
	logfile.close()

	#We also write the results for total planets in another file

	logfile = open("Stats/stats_tot_" + result_file, "w")
	logfile.write('nstar\tstype\ta\tang_sep\tRp\tFinc\tdist\tPorb\tF560W\tF1000W\tF1500W\tobs_time\n')
	for i in range(len(data[0])):
		logfile.write(str(data[0][i]) + "\t" + str(stype_arr[i]) + "\t" + str(data[2][i]) + "\t" + str(data[3][i]) + "\t" +  str(data[4][i]) + "\t"+ str(data[5][i]) + "\t"+   str(data[6][i]) + "\t"+ str(data[7][i]) +"\t"+ str(data[8][i]) +"\t"+ str(data[9][i]) +"\t"+ str(data[10][i]) + "\t" + str(data[11][i]) + "\n")
	logfile.close()

	return 