import numpy as np
class PlanetSample:
    def __init__(self, sample_url, dataset = "SAG"):
        planets = open(sample_url, 'r')
        planets_lines = planets.readlines()
        # Initialize lists
        Rp = [] #planet radius
        Tp = [] # planet equilibrium temperature
        F_inc = [] # incoming stellar flux
        ang_sep = [] #planet to host star angular separation
        snumber = [] # host star number
        stype = [] #host star spectral type
        sdist = []
        nMC = []
        a = []
        period = []
        ecc = []
        Ms = []
        Mp = []
        Ts = []

        if(dataset == "ragrug"):
            data_pos = {
                'Rp' : 1,
                "Tp" : 10,
                "F_inc" : 19,
                "ang_sep" : 6,
                "snumber" : 21,
                "stype" : 20,
                "sdist" : 15,
            }
        elif(dataset == "SAG"):
            data_pos = {
                'Rp' : 1,
                "Tp" : 17,
                "F_inc" : 12,
                "ang_sep" : 5,
                "snumber" : 27,
                "stype" : 23,
                "sdist" : 19,
                'nMC' : 0,
                'a' : 3,
                'period' : 2,
                'ecc' : 11,
                'Ms' : 22,
                'Mp' : 18,
                'Ts' : 21
            }
            
        else:
            print("Please choose a dataset value between 'ragrug' and 'SAG'")

        #Remove the header line
        planets_lines.pop(0)

        # Go through all lines
        for line in planets_lines:
            line_temp = line.split('\t')

            # Fill lists
            Rp += [float(line_temp[data_pos['Rp']])]
            Tp += [float(line_temp[data_pos['Tp']])]
            F_inc += [float(line_temp[data_pos['F_inc']])]
            ang_sep += [float(line_temp[data_pos['ang_sep']])]
            snumber += [float(line_temp[data_pos['snumber']])]
            stype += [str(line_temp[data_pos['stype']])]
            sdist += [float(line_temp[data_pos['sdist']])]
            nMC += [float(line_temp[data_pos['nMC']])]
            a += [float(line_temp[data_pos['a']])]
            period += [float(line_temp[data_pos['period']])]
            ecc += [float(line_temp[data_pos['ecc']])]
            Ms += [float(line_temp[data_pos["Ms"]])]
            Mp += [float(line_temp[data_pos["Mp"]])]
            Ts += [float(line_temp[data_pos["Ts"]])]


        planets.close()
        # Convert lists to arrays (more handy)
        self.Rp = np.array(Rp)
        self.Tp = np.array(Tp)
        self.F_inc = np.array(F_inc)
        self.ang_sep = np.array(ang_sep)
        self.snumber = np.array(snumber)
        self.stype = np.array(stype)
        self.sdist = np.array(sdist)
        self.nMC = np.array(nMC)
        self.a = np.array(a)
        self.period = np.array(period)
        self.ecc = np.array(ecc)
        self.Ms = np.array(Ms)
        self.Mp = np.array(Mp)
        self.Ts = np.array(Ts)
        self.compute_HZ(model = "MS")   

    def compute_HZ(self, model = "MS"):
        if(model == "MS"):
            s_zero_in, s_zero_out = 1.7665, 0.324
            A_in, A_out = 1.3351E-4, 5.3221E-5
            B_in, B_out  = 3.1515E-9, 1.4288E-9
            C_in, C_out = -3.3488E-12, -1.1049E-12
            T_star = self.Ts - 5780
            self.HZ_in = s_zero_in + A_in * T_star + B_in*T_star**2 + C_in*T_star**3
            self.HZ_out = s_zero_out + A_out * T_star + B_out*T_star**2 + C_out*T_star**3
        elif(model == "POST-MS"):
            s_zero_in, s_zero_out = 1.1066, 0.324
            A_in, A_out = 1.2181E-4, 5.3221E-5
            B_in, B_out  = 1.534E-8, 1.4288E-9
            C_in, C_out = -1.5018E-12, -1.1049E-12
            T_star = self.Ts - 5780
            self.HZ_in = s_zero_in + A_in * T_star + B_in*T_star**2 + C_in*T_star**3
            self.HZ_out = s_zero_out + A_out * T_star + B_out*T_star**2 + C_out*T_star**3
        else:
            print("Wrong model, please choose between 'MS' and 'POST-MS'")
        return
        
    def append_fluxes(self, flux_url, dataset = "SAG"):
        fluxes = open(flux_url, 'r')
        fluxes_lines = fluxes.readlines()
        fluxes_lines.pop(0)
        # Initialize lists for the flux in each band
        F560W = []
        F1000W = []
        F1500W = []
        # Go through all lines
        for line in fluxes_lines:
            line_temp = line.split('\t')
            # Fill lists with integrated planet thermal black body flux +  integrated planet reflected host star flux
            # in each band -> total flux of the planet in each band
            F560W += [float(line_temp[1])+float(line_temp[2])]
            F1000W += [float(line_temp[4])+float(line_temp[5])]
            F1500W += [float(line_temp[7])+float(line_temp[8])]
        fluxes.close()

        # Convert lists to arrays (more handy)
        self.F560W = np.array(F560W)
        self.F1000W = np.array(F1000W)
        self.F1500W = np.array(F1500W)
 