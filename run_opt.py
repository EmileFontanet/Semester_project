import numpy as np
import random
import math
import time as tt
import optimization_master as opt
import planet_sampler_pop
import planet_stats as pstat

# Name of the file for the results
logfile_name = "new_baseline"

#The files containing the planets and their fluxes
planet_table = "emile_testing_SAG13_bright1000.txt"
flux_table = "emile_testing_SAG13_bright1000_LIFE.txt"

# Wavelength band that will be used for the optimization
wavelength = 10
	# 5.6
	# 10
	# 15
# Minimum/maximum planet radius that will be used for the optimization (in earth radii)
min_radius = 0.5
max_radius = 1.5

# Minimum/maximum insolation of the planet that will be used for the optimization (in Earth insolation)
min_flux = 0.37
max_flux = 1.75

# IWA of the telescope 
IWA = 5E-3

# OWA of the telescope
OWA = 1

# The total mission time that we allow for the optimization
#mission_time = 3*3.154e+7 # 3 years
mission_time = 35000 * 326 # Every star is observed for 35000 sec, which is the baseline scenario in Kam&Quanz

#The time required for the telescope to go from one star to another
time_between_stars = 0
# The delta T to use for the optimization
delta_t = 1000

# The stars on which we do the optimization
stars_sample = range(0,326,1)

# The stellar types that we want to consider in the optimization
stypes = ["A","M", "F", "G", "K"]

#We load the planet tables and fluxes into the script
print("Loading the tables..")
planet_sample = planet_sampler_pop.PlanetSample(planet_table, dataset = "SAG")
planet_sample.append_fluxes(flux_table, dataset = "SAG")

# We run the optimization
print("Computing the baseline..")
opt.dte(planet_sample, stars_sample, mission_time, delta_t, IWA, wavelength, max_radius = max_radius, min_radius = min_radius, max_flux = max_flux, min_flux = min_flux, time_between_stars = time_between_stars, logfile_name = logfile_name + "_baseline.txt")

print("Starting the optimization")
opt.dthgs_by_time(planet_sample, stars_sample, mission_time, delta_t, IWA, wavelength, max_radius = max_radius, min_radius = min_radius, max_flux = max_flux, min_flux = min_flux, time_between_stars = time_between_stars, logfile_name = logfile_name + "_OPT.txt")
print("Optimization done, saving stats..")

# We can now create the stats 
pstat.make_stats(logfile_name + "_baseline.txt", planet_sample, IWA, OWA, min_radius, max_radius, min_flux, max_flux)
pstat.make_stats(logfile_name + "_OPT.txt", planet_sample, IWA, OWA, min_radius, max_radius, min_flux, max_flux)
