import numpy as np
import planet_sampler_pop
import random
import math
import time as tt
### USEFUL FUNCTIONS FOR ALGORITHMS
def make_sens_array(total_time_allowed, delta_t, wavelength = 5.6):
    """
    This function creates the sensitivity array.
    Inputs : 
    total_time_allowed: int corresponding to the max time for wich we should compute a sensitivity value
    delta_t: the time step used for the array
    wavelength: the wavelength band we want to use (which then give different values of sens_lim according to LIFE)

    The function returns the sensitivity array

    """
    if(wavelength == 5.6):
        sens_limit = 0.16
    elif(wavelength == 10):
        sens_limit = 0.54
    elif(wavelength == 15):
        sens_limit = 1.39
    else:
        print("Error! Wrong value of wavelength, please choose between 5.6, 10 and 15.")
    exptime_default=35000.
    exptime_arr=np.arange(0.,total_time_allowed,step = delta_t)#create the array of exptimes from 0 - tot_time
    exptime_arr[0]=1
    factor_arr=np.sqrt(exptime_arr/exptime_default)#divide the array by 35'000 making it range from 1/35000 to tot_time/35000
    #and take sqrt
    sens_limit_arr=sens_limit/factor_arr
    return sens_limit_arr


def make_mask(planet_sample, IWA = 5E-3, wavelength = 5.6, max_radius = 6, min_radius = 0, max_temp = 5000, min_temp = 0, min_flux = 0, max_flux = 5, stypes = ["A","M", "F", "G", "K"]):
    """
    Create a mask in the form of a boolean numpy array
    Inputs:
    planet_sample: numpy array containing all the fluxes and information about the planets in the dataset
    The other parameters are pretty straightforward and simply give the boundaries for the different masks we want to use

    """
    OWA = 1
    lam_F560W = 5.6E-6 # meters
    lam_F1000W = 10E-6 # meters
    lam_F1500W = 15E-6 # meters
    if(wavelength == 5.6):
        owa_mask = planet_sample.ang_sep <= OWA 
        ang_mask = planet_sample.ang_sep >= IWA*(lam_F560W/lam_F1000W) # inner working angle @ 5.6 microns
    elif(wavelength == 10):
        ang_mask = planet_sample.ang_sep >= IWA 
        owa_mask = planet_sample.ang_sep <= OWA # inner working angle @ 10 microns
    elif(wavelength == 15):
        owa_mask = planet_sample.ang_sep <= OWA 
        ang_mask = planet_sample.ang_sep >= IWA*(lam_F1500W/lam_F1000W)# inner working angle @ 15 microns
    else:
        print("Error! Wrong value of wavelength, please choose between 5.6, 10 and 15.")
    max_radius_mask = planet_sample.Rp <= max_radius 
    min_radius_mask = planet_sample.Rp >= min_radius
    max_temp_mask = planet_sample.Tp <= max_temp
    min_temp_mask = planet_sample.Tp >= min_temp
    max_flux_mask = planet_sample.F_inc <= max_flux # Fixed HZ
    min_flux_mask = planet_sample.F_inc >= min_flux
    #max_flux_mask = planet_sample.F_inc <= planet_sample.HZ_in
    #min_flux_mask = planet_sample.F_inc >= planet_sample.HZ_out #Dynamic HZ

    stype_mask = np.isin(planet_sample.stype, stypes)
    mask = max_radius_mask & ang_mask & owa_mask & min_radius_mask & max_temp_mask & min_temp_mask & max_flux_mask & min_flux_mask & stype_mask
    return mask
        
def fill_detec_time_arr(planet_sample, stars_per_dt, sens_limit_arr, base_mask, stars_to_optimize,exptime_arr, wavelength = 5.6, stypes = ["A","M", "F", "G", "K"]):
    """
    This function creates the detection times array. This array simply contains all the detection times of all the planets in the sample.
    It is returning the same stars_per_dt dictionnary as we give in input but after adding an additional array to it (the detection_times array)
    inputs:
    planet_sample: the original planet sample
    stars_per_dt: the dictionnary containing all the stars indexed by their names 
    sens_limit_arr: the array obtained by using the function creating the sens limit arr above
    base_mask: the mask obtained by using the function above creating the mask
    stars_to_optimize: an array containing the indexes of all the stars on which we want to do the optimization
    exptime_arr: an array containing the different values of exposure time
    stypes: array containing all the stellar types that we want to make the optimization on
    """

    if(wavelength == 5.6):
        planet_fluxes = planet_sample.F560W
    elif(wavelength == 10):
        planet_fluxes = planet_sample.F1000W
    elif(wavelength == 15):
        planet_fluxes = planet_sample.F1500W
    else:
        print("Error! Wrong value of wavelength, please choose between 5.6, 10 and 15.")
    for star_index in stars_to_optimize:
        temp_mask = planet_sample.snumber == star_index
        total_mask = base_mask & temp_mask
        planet_indices = np.arange(len(total_mask))[total_mask]#get the indices of the planets that are conserved by the mask
        star_name = "Star " + str(star_index)
        star_type = planet_sample.stype[np.where(planet_sample.snumber == star_index)[0][0]]
        if(not(star_type in stypes)): #if the star we are checking is not of the right type
            stars_per_dt.pop(star_name, None)
        else:
            stars_per_dt[star_name]["star_type"] = star_type
            for it, flux in enumerate(planet_fluxes[total_mask]): # we look at each value of planetary flux
                test = sens_limit_arr[sens_limit_arr <= flux]
                corresp_planet_index = planet_indices[it]#retrieve the planet index
                if(len(test) > 0): # and we see if this planet could be detectable
                    thresh = test[0]
                    itemindex = int(np.where(sens_limit_arr==thresh)[0][0])
                    time = exptime_arr[itemindex] # and we compute the corresponding time of detection
                    data = time
                    stars_per_dt[star_name]["detection_times"].append(data)
            stars_per_dt[star_name]["detection_times"] = np.array(stars_per_dt[star_name]["detection_times"])
    return stars_per_dt



#### ALGORITHM PART
def dte(ps, stars_to_optimize, total_time_allowed, delta_t, IWA = 5E-3, wavelength = 5.6, max_radius = 6, min_radius = 0, max_temp = 5000, min_temp = 0, min_flux = 0, max_flux = 5, time_between_stars = 0, logfile_name = "results.txt", OWA = 1):
    '''
    Distributing Time Equally 
    Inputs:
    ps : the planet sample data obtained by the simulation
    stars_to_optimize: an array of ints containing the indexes of the stars to optimize
    total_time_allowed: the total mission time available
    delta_t: the time step
    IWA: inner working angle of the telescope
    wavelength: the wavelength band
    max/min radius/temp/flux : parameters for the mask
    time_between_stars: time needed for the telescope to go from one star to another
    logfile_name : the name of the file in which the results will be stored
    The function returns two dictionaries : stars_per_dt and stars_per_dt_count
    The first one contains all the information about the optimized planets (the one in the optimization range)
    The second one contains all the information about all the planets, also counting the ones that we would detect by accident and which are outside the opt range
    Each dictionnary has one entry per star, which has info about the number of planets detected around that star, how long it has been observed and its distance
    There are also statistics on the temperature, radii and flux for all the planets
    '''
    nMC = max(ps.nMC) + 1
    exptime_default = 35000

    n_of_stars = len(stars_to_optimize)
    total_time_allowed = total_time_allowed - (n_of_stars)*time_between_stars
    time_per_star = total_time_allowed // n_of_stars
    
    if(wavelength == 5.6):
        fluxes = ps.F560W
        sens_limit = 0.16
        IWA = IWA * (5.6/10)
    elif(wavelength == 10):
        fluxes = ps.F1000W
        sens_limit = 0.54
    elif(wavelength == 15):
        fluxes = ps.F1500W
        sens_limit = 1.39
        IWA = IWA*(15/10)
    else:
        print("Wrong value of walevength")

    reached_sensitivity = sens_limit / math.sqrt(time_per_star/exptime_default)
    #We load all the necessary information in a np array
    data = np.vstack([ps.snumber, fluxes, ps.ang_sep,  ps.sdist, ps.Rp, ps.F_inc])

    # We compute the masks for detectable planets
    flux_mask = data[1] > reached_sensitivity
    ang_mask = data[2] >= IWA 
    owa_mask = data[2] <= OWA
    detection_mask = flux_mask & ang_mask & owa_mask

    # We remove all the undetectable planets of the sample
    data = data[:, detection_mask]

    #We compute the mas for habitable planets
    min_radius_mask = data[4] > min_radius
    max_radius_mask = data[4] < max_radius
    min_flux_mask = data[5] > min_flux
    max_flux_mask = data[5] < max_flux
    
    habitable_mask = max_radius_mask & min_radius_mask & min_flux_mask & max_flux_mask 

    #We now remove all the non habitable planets
    data = data[:, habitable_mask]    
    tot_p = len(data[0])
    
    print("EQUAL -- Got a total of " + str(tot_p/nMC) + " planets when analyzing " + str(len(stars_to_optimize)) + " stars for a total of " + str(total_time_allowed) + " seconds")

    
    logfile = open(logfile_name, 'w')
    logfile.write('star_number\tobs_time\tstar_type\tplanets_detected\tsdist\n')
    for it, star in enumerate(stars_to_optimize):
        try:
            temp = data[:, data[0] == star]
            logfile.write(str(int(temp[0][0])) + "\t" + str(int(time_per_star)) + "\t" + str("A") + "\t" + str(len(temp[0])) + "\t" + str(temp[3][0]) + "\n")
        except:
            logfile.write(str(int(it)) + "\t" + str(int(time_per_star)) + "\t" + str("A") + "\t" + str(0) + "\t" + "None" + "\n")
    logfile.close()

    return 





def dthgs_by_time(planet_sample, stars_to_optimize, total_time_allowed, delta_t, IWA = 5E-3, wavelength = 5.6, max_radius = 6, min_radius = 0, max_temp = 100000, min_temp = 0, min_flux = 0, max_flux = 5, stypes = ["A","M", "F", "G", "K"], time_between_stars = 0, logfile_name = "results.txt"):
    '''
    Distributing Time to Highest Global Slope

    Inputs:
    planet_sample : the planet sample data obtained by the simulation
    stars_to_optimize: an array of ints containing the indexes of the stars to optimize
    total_time_allowed: the total mission time available, when all the time has been distributed to stars the algorithm stops
    delta_t: the time step
    IWA: inner working angle of the telescope
    wavelength: the wavelength band
    max/min radius/temp/flux : parameters for the mask
    time_between_stars: time needed for the telescope to go from one star to another

    The function returns two dictionaries : stars_per_dt and stars_per_dt_count
    The first one contains all the information about the optimized planets (the one in the optimization range)
    The second one contains all the information about all the planets, also counting the ones that we would detect by accident and which are outside the opt range
    Each dictionnary has one entry per star, which has info about the number of planets detected around that star, how long it has been observed and its distance
    There are also statistics on the temperature, radii and flux for all the planets
    
    '''
    nMC = max(planet_sample.nMC) + 1
    start = tt.time()
    stars_per_dt = {}
    print("Creating the sensitivity arrays")
    sens_limit_arr = make_sens_array(total_time_allowed, delta_t, wavelength)
    unobserved_stars = []
    for star_index in stars_to_optimize:#Might cause an error if there is no planet around a given star (for the sdist part)
        star_name = "Star " + str(star_index)
        try:
            sdist_ = planet_sample.sdist[planet_sample.snumber == star_index][0]
            stars_per_dt[star_name]  = {
            "detection_times" : [],
            "number_of_planets" : 0,
            "star_number" : star_index,
            "sdist" : sdist_

        }
        except:
            print("Removing star " + str(star_index) + " because it has no planets")

        
    #We create the mask corresponding to the type of planets we want to detect 
    mask = make_mask(planet_sample, IWA, wavelength, max_radius, min_radius, max_temp, min_temp, min_flux, max_flux)
    #We create an exp time array
    exptime_arr=np.arange(0.,total_time_allowed,step = delta_t) #create the array of exptimes from 0 - tot_time
    #For each star, we fill its detection time array with the planet table
    print("Computing detection times")
    stars_per_dt = fill_detec_time_arr(planet_sample, stars_per_dt, sens_limit_arr, mask, stars_to_optimize,exptime_arr, wavelength)
    #We create the bin edges array
    bins_edges = np.linspace(0, total_time_allowed, int(total_time_allowed/delta_t) +1)
    ### We then need to bin our data into bins of size delta T
    tot_t = 0.
    tot_p = 0.
    len_bins_edges = len(bins_edges)

    #We loop over all the stars that have at least one planet (which might not be detectable)
    for star in stars_per_dt:
        #If the star contains detectable planets
        if(len(stars_per_dt[star]['detection_times']) > 0 ):
            
            stars_per_dt[star]['planets_per_dt'], stars_per_dt[star]['bins_edges'] = np.histogram(stars_per_dt[star]['detection_times'], bins_edges)
            for i in range(len_bins_edges - 1):#after making a hist of the detect times we make the same for the stats
                temp_mask =( stars_per_dt[star]['detection_times'] >= bins_edges[i]) & (stars_per_dt[star]["detection_times"] < bins_edges[i+1])
            #We compute the slope array
            stars_per_dt[star]['slopes'] = np.cumsum(stars_per_dt[star]['planets_per_dt'], dtype = float)/(time_between_stars + delta_t*(np.arange(len(stars_per_dt[star]['planets_per_dt']))+1))
            stars_per_dt[star]['planets_per_dt'] = list(stars_per_dt[star]['planets_per_dt'])
            stars_per_dt[star]['times_observed'] = 0
        
        #If the star has no detectable planets we add it to the unobserved_stars array and then remove it from stars_per_dt
        else:
            unobserved_stars.append(star)
    for star in unobserved_stars:
        del stars_per_dt[star]
    #We define a few variables in order to keep track of the stars we are observing
    total_time = 0
    current_star = ""
    observed_stars = []
    stars_status = np.zeros(len(stars_per_dt), dtype = int)
    #current_star is the star that we just observed, temp star is the star that we are deciding to observe next
    print("Optimizing the observing strategy")
    ##This loop goes on until the time has reached the value fixed by the mission
    while(total_time < total_time_allowed): 
        #We compute how much time is left 
        time_available = total_time_allowed - total_time

        #If the time available is smaller that a delta t, then we stop
        if(time_available < delta_t):
            break
        temp = np.zeros(len(stars_per_dt))

        ## Loop over each star of the dict
        for it, star in enumerate(stars_per_dt): 
            #We append the max value of slope of each star to the temp array
            temp[it] = np.max(stars_per_dt[star]['slopes'][stars_status[it]:int(stars_status[it] + time_available/delta_t)]) ## Fill an array with the number of planets detected inthe first time bin for each star
        #We find the index for which we have the higehst slope
        temp_star_id = np.argmax(temp)
        #And we find the star it corresponds to 
        temp_star = list(stars_per_dt.keys())[temp_star_id] ## Find for which star we have the largest value
        #We find how much time the value we chose corresponds to
        corresp_t = np.argmax(stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:int(stars_status[temp_star_id] + time_available/delta_t)]) + 1    
        
        for i in range(corresp_t):# We add the planets and stats corresponding to the time we add
            stars_per_dt[temp_star]['number_of_planets'] += stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id] + i]#increase the number of planets for this star by the ammount we are popping out
        stars_status[temp_star_id] += corresp_t

        if(current_star == ""):#If it is the first star, we also hvae to add the time between stars
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:], dtype = float)/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t *delta_t + time_between_stars
        elif(current_star == temp_star):#If we stayed on the same star
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:], dtype = float)/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t#We decrease the total time by the amount of exposure time we juste removed for one star
        elif((current_star != temp_star) and (not(temp_star in observed_stars)))  :  #If we changed star
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:], dtype = float)/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t + time_between_stars#We decrease the total time by the amount of exposure time we juste removed for one star
        elif((current_star != temp_star) and (temp_star in observed_stars)):
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:], dtype = float)/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t
        #We update the parameters to keep track of the routine
        current_star = temp_star
        observed_stars.append(temp_star)
        stars_per_dt[temp_star]['times_observed'] += corresp_t
    
    for star in stars_per_dt:
        tot_p += stars_per_dt[star]['number_of_planets']
        tot_t += delta_t * stars_per_dt[star]['times_observed']
        
        stars_per_dt[star]['obs_time'] = delta_t * stars_per_dt[star]['times_observed']
        stars_per_dt[star].pop('planets_per_dt', None)
        stars_per_dt[star].pop('bins_edges', None)
        stars_per_dt[star].pop('detection_times', None)
        stars_per_dt[star].pop('slopes', None)
    #free_p, stars_per_dt_count = count_total_planets(stars_per_dt, planet_sample, stars_to_optimize, total_time_allowed, delta_t, IWA, wavelength)
    free_p = 0.
    end = tt.time()
    print("Got a total of " + str(tot_p/nMC) + " planets when analyzing " + str(len(stars_to_optimize)) + " stars for a total of " + str(total_time) + " seconds.\nRunning time : " + str(end-start))
    logfile = open(logfile_name, 'w')
    logfile.write('star_number\tobs_time\tstar_type\tplanets_detected\tsdist\n')
    for star in stars_per_dt:
        temp = stars_per_dt[star]
        logfile.write(str(temp['star_number']) + "\t" + str(temp['obs_time']) + "\t" + str(temp["star_type"]) + "\t" + str(temp['number_of_planets']) + "\t" + str(temp['sdist']) + "\n")
    logfile.close()
    return 
    

def dthgs_by_planets(planet_sample, stars_to_optimize, min_n_of_planets, delta_t, IWA = 5E-3, wavelength = 5.6, max_radius = 6, min_radius = 0, max_temp = 100000, min_temp = 0, min_flux = 0, max_flux = 5, stypes = ["A","M", "F", "G", "K"], time_between_stars = 0):
    '''
    Distributing Time to Highest Global Slope

    Inputs:
    planet_sample : the planet sample data obtained by the simulation
    stars_to_optimize: an array of ints containing the indexes of the stars to optimize
    min_n_of_planets: number of planets required before stopping the algorithm. once the algorithm reaches this value it stops, there is however also a max time value to prevent too long computations
    this value is given by the total_time_allowed variable below
    delta_t: the time step
    IWA: inner working angle of the telescope
    wavelength: the wavelength band
    max/min radius/temp/flux : parameters for the mask
    time_between_stars: time needed for the telescope to go from one star to another

    The function returns two dictionaries : stars_per_dt and stars_per_dt_count
    The first one contains all the information about the optimized planets (the one in the optimization range)
    The second one contains all the information about all the planets, also counting the ones that we would detect by accident and which are outside the opt range
    Each dictionnary has one entry per star, which has info about the number of planets detected around that star, how long it has been observed and its distance
    There are also statistics on the temperature, radii and flux for all the planets
    
    '''
    start = tt.time()
    #total_time_allowed = (35000+time_between_stars)*len(stars_to_optimize)*4
    total_time_allowed = 1.577e+8
    stars_per_dt = {}
    sens_limit_arr = make_sens_array(total_time_allowed, delta_t, wavelength)
    for star_index in stars_to_optimize:
        star_name = "Star " + str(star_index)
        stars_per_dt[star_name]  = {
        "detection_times" : [],
        "number_of_planets" : 0,
        "radius_stats" : [],
        "F_inc_stats" : [],
        "temp_stats" : [],
        "radius_stats_per_dt" : [],
        "F_inc_stats_per_dt" : [],
        "temp_stats_per_dt" : [],
        "star_number" : star_index,
        "sdist" : planet_sample.sdist[planet_sample.snumber == star_index][0]

    }
    mask = make_mask(planet_sample, IWA, wavelength, max_radius, min_radius, max_temp, min_temp, min_flux, max_flux, stypes)
    exptime_arr=np.arange(0.,total_time_allowed,step = delta_t) #create the array of exptimes from 0 - tot_time
    stars_per_dt = fill_detec_time_arr(planet_sample, stars_per_dt, sens_limit_arr, mask, stars_to_optimize,exptime_arr, wavelength, stypes)
    bins_edges = np.linspace(0, total_time_allowed, int(total_time_allowed/delta_t) +1)
    ### We then need to bin our data into bins of size delta T
    tot_t = 0.
    tot_p = 0.
    len_bins_edges = len(bins_edges)
    for star in stars_per_dt:
        if(len(stars_per_dt[star]['detection_times']) > 0 ):
            stars_per_dt[star]['planets_per_dt'], stars_per_dt[star]['bins_edges'] = np.histogram(stars_per_dt[star]['detection_times'][:,0], bins_edges)
            for i in range(len_bins_edges - 1):#after making a hist of the detect times we make the same for the stats
                temp_mask =( stars_per_dt[star]['detection_times'][:,0] >= bins_edges[i]) & (stars_per_dt[star]["detection_times"][:,0] < bins_edges[i+1])
                temp_stats = stars_per_dt[star]["detection_times"][:,1:][temp_mask]
                stars_per_dt[star]["radius_stats_per_dt"].append(temp_stats[:,0])
                stars_per_dt[star]["F_inc_stats_per_dt"].append(temp_stats[:,1])
                stars_per_dt[star]["temp_stats_per_dt"].append(temp_stats[:,2])
            stars_per_dt[star]['slopes'] = np.cumsum(stars_per_dt[star]['planets_per_dt'])/(time_between_stars + delta_t*(np.arange(len(stars_per_dt[star]['planets_per_dt']))+1))
            stars_per_dt[star]['planets_per_dt'] = list(stars_per_dt[star]['planets_per_dt'])
            stars_per_dt[star]['times_observed'] = 0
        else:
            stars_per_dt[star]['planets_per_dt'] = np.zeros(len_bins_edges)
            stars_per_dt[star]['slopes'] = np.zeros(len_bins_edges)
            stars_per_dt[star]['obs_time'] = 0
            stars_per_dt[star]['times_observed'] = 0
    total_time = 0
    current_star = ""
    observed_stars = []
    stars_status = np.zeros(len(stars_per_dt), dtype = int)
    #current_star is the star that we just observed, temp star is the star that we are deciding to observe next
    while((tot_p < min_n_of_planets) and (total_time < total_time_allowed)): ##This loop goes on until the time has reached the value fixed by the mission

        time_available = total_time_allowed - total_time
        if(time_available < delta_t):
            break
        temp = np.zeros(len(stars_per_dt))
        for it, star in enumerate(stars_per_dt): ## Loop over each star of the dict
            temp[it] = np.max(stars_per_dt[star]['slopes'][stars_status[it]:int(stars_status[it] + time_available/delta_t)]) ## Fill an array with the number of planets detected inthe first time bin for each star
        temp_star_id = np.argmax(temp)
        temp_star = list(stars_per_dt.keys())[temp_star_id] ## Find for which star we have the largest value
        corresp_t = np.argmax(stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:int(stars_status[temp_star_id] + time_available/delta_t)]) + 1    
        for i in range(corresp_t):# We add the planets and stats corresponding to the time we add
            tot_p += stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id] + i]
            stars_per_dt[temp_star]['number_of_planets'] += stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id] + i]#increase the number of planets for this star by the ammount we are popping out
            stars_per_dt[temp_star]['radius_stats'].append(stars_per_dt[temp_star]["radius_stats_per_dt"][stars_status[temp_star_id] + i])
            stars_per_dt[temp_star]['F_inc_stats'].append(stars_per_dt[temp_star]["F_inc_stats_per_dt"][stars_status[temp_star_id] + i])
            stars_per_dt[temp_star]['temp_stats'].append(stars_per_dt[temp_star]["temp_stats_per_dt"][stars_status[temp_star_id] + i])
        stars_status[temp_star_id] += corresp_t

        if(current_star == ""):#If it is the first star, we also hvae to add the time between stars
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:])/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t *delta_t + time_between_stars
        elif(current_star == temp_star):#If we stayed on the same star
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:])/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t#We decrease the total time by the amount of exposure time we juste removed for one star
        elif((current_star != temp_star) and (not(temp_star in observed_stars)))  :  #If we changed star
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:])/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t + time_between_stars#We decrease the total time by the amount of exposure time we juste removed for one star
        elif((current_star != temp_star) and (temp_star in observed_stars)):
            stars_per_dt[temp_star]['slopes'][stars_status[temp_star_id]:] = np.cumsum(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:])/(delta_t * (np.arange(len(stars_per_dt[temp_star]['planets_per_dt'][stars_status[temp_star_id]:]))+1))
            total_time += corresp_t * delta_t
        current_star = temp_star
        observed_stars.append(temp_star)
        stars_per_dt[temp_star]['times_observed'] += corresp_t
    radius_stats = []
    F_inc_stats = []
    temp_stats = []
    for star in stars_per_dt:
        tot_t += delta_t * stars_per_dt[star]['times_observed']
        for i in range(len(stars_per_dt[star]['radius_stats'])):
            radius_stats.append(stars_per_dt[star]['radius_stats'][i])
            F_inc_stats.append(stars_per_dt[star]['F_inc_stats'][i])
            temp_stats.append(stars_per_dt[star]['temp_stats'][i])
        stars_per_dt[star]['obs_time'] = delta_t * stars_per_dt[star]['times_observed']
        stars_per_dt[star].pop('planets_per_dt', None)
        stars_per_dt[star].pop('bins_edges', None)
        stars_per_dt[star].pop('radius_stats_per_dt', None)
        stars_per_dt[star].pop('F_inc_stats_per_dt', None) ## Popping everything to free up some RAM
        stars_per_dt[star].pop('temp_stats_per_dt', None)
        stars_per_dt[star].pop('detection_times', None)
        stars_per_dt[star].pop('slopes', None)
        #print("After the cleaning, planet " + star + " has a total of " + str(stars_per_dt[star]['number_of_planets']) + " planets and is observed for " + str(stars_per_dt[star]['obs_time']) + " sec" )
    free_p, stars_per_dt_count = count_total_planets(stars_per_dt, planet_sample, stars_to_optimize, total_time_allowed, delta_t, IWA, wavelength, stypes)
    end = tt.time()
    print("DTHGS -- Got a total of " + str(tot_p/5000) + " (" + str(free_p/5000) +") planets when analyzing " + str(len(stars_to_optimize)) + " stars for a total of " + str(total_time) + " seconds. Running time : " + str(end-start))
    stars_per_dt['tot_p'] = tot_p
    stars_per_dt['tot_t'] = tot_t
    stars_per_dt['radius_stats'] = radius_stats
    stars_per_dt['temp_stats'] = temp_stats
    stars_per_dt['F_inc_stats'] =F_inc_stats
      
    return stars_per_dt, stars_per_dt_count
 
def count_total_planets(stars_per_dt, planet_sample, stars_to_optimize, total_time_allowed, delta_t, IWA = 5E-3, wavelength = 5.6, stypes = ["A","M", "F", "G", "K"] ):
    """
    This function counts how many planets we would detect in total, including the ones outside the optimization range, by applying the exact same distribution of time as the one it is given
    but without applying any mask except the one from inner working angle.

    """
    stars_per_dt_count = {}
    sens_limit_arr = make_sens_array(total_time_allowed, delta_t, wavelength)
    for star_index in stars_to_optimize:
        star_name = "Star " + str(star_index)
        if star_name in stars_per_dt.keys():
            stars_per_dt_count[star_name]  = {
            "detection_times" : [],
            "number_of_planets" : 0,
            "radius_stats_per_dt" : [],
            "F_inc_stats_per_dt" : [],
            "temp_stats_per_dt" : [],
            "star_number" : star_index,
            "sdist" : planet_sample.sdist[planet_sample.snumber == star_index][0]

        }
    mask = make_mask(planet_sample, IWA, wavelength, 999999, 0, 9999, 0, 0, 99999)#We make a mask which actually does not filter any planets out
    exptime_arr=np.arange(0.,total_time_allowed,step = delta_t) #create the array of exptimes from 0 - tot_time
    stars_per_dt_count = fill_detec_time_arr(planet_sample, stars_per_dt_count, sens_limit_arr, mask, stars_to_optimize,exptime_arr, wavelength, stypes)
    bins_edges = np.linspace(0, total_time_allowed, int(total_time_allowed/delta_t) +1)
    ### We then need to bin our data into bins of size delta T
    tot_t = 0
    tot_p = 0

    for star in stars_per_dt_count:
        stars_per_dt_count[star]['planets_per_dt'], stars_per_dt_count[star]['bins_edges'] = np.histogram(stars_per_dt_count[star]['detection_times'][:,0], bins_edges)
        for i in range(len(bins_edges) - 1):#after making a hist of the detect times we make the same for the stats
            temp_mask =( stars_per_dt_count[star]['detection_times'][:,0] >= bins_edges[i]) & (stars_per_dt_count[star]["detection_times"][:,0] < bins_edges[i+1])
            temp_stats = stars_per_dt_count[star]["detection_times"][:,1:][temp_mask]
            stars_per_dt_count[star]["radius_stats_per_dt"].append(temp_stats[:,0])
            stars_per_dt_count[star]["F_inc_stats_per_dt"].append(temp_stats[:,1])
            stars_per_dt_count[star]["temp_stats_per_dt"].append(temp_stats[:,2])
        stars_per_dt_count[star]['planets_per_dt'] = list(stars_per_dt_count[star]['planets_per_dt'])
        stars_per_dt_count[star]['times_observed'] = 0
    

    stars_per_dt_count['radius_stats'] = []
    stars_per_dt_count['temp_stats'] = []
    stars_per_dt_count['F_inc_stats'] = []
    for star in stars_per_dt:
        star_exp_time = int(stars_per_dt[star]["obs_time"]/delta_t)
        if(star_exp_time  == 0):
            stars_per_dt_count[star]['obs_time'] = 0
        for i in range(star_exp_time):
            tot_p += stars_per_dt_count[star]['planets_per_dt'][i]
            stars_per_dt_count[star]["number_of_planets"] += stars_per_dt_count[star]['planets_per_dt'][i]
            stars_per_dt_count['radius_stats'].append(stars_per_dt_count[star]["radius_stats_per_dt"][i])
            stars_per_dt_count['F_inc_stats'].append(stars_per_dt_count[star]["F_inc_stats_per_dt"][i])
            stars_per_dt_count['temp_stats'].append(stars_per_dt_count[star]["temp_stats_per_dt"][i])
            stars_per_dt_count[star]['obs_time'] = delta_t * star_exp_time
        stars_per_dt_count[star].pop('planets_per_dt', None)
        stars_per_dt_count[star].pop('bins_edges', None)
        stars_per_dt_count[star].pop('radius_stats_per_dt', None)
        stars_per_dt_count[star].pop('F_inc_stats_per_dt', None) ## Popping everything to free up some RAM
        stars_per_dt_count[star].pop('temp_stats_per_dt', None)
        stars_per_dt_count[star].pop('detection_times', None)
    stars_per_dt_count['tot_p'] = tot_p

    return tot_p, stars_per_dt_count

