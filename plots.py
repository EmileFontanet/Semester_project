import numpy as np
from matplotlib import pyplot as plt


import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import NullFormatter

# PARAMETERS
#==============================================================================
planets_path = 'stats_tot_new_baseline_baseline.txt'
nMC = 1000
OWA = 1
IWA = 5E-3
lam_F560W = 5.6E-6 # meters
lam_F1000W = 10E-6 # meters
lam_F1500W = 15E-6 # meters
sensF560W = 0.16 # micro-Janskys
sensF1000W = 0.54 # micro-Janskys
sensF1500W = 1.39 # micro-Janskys

exptime_default = 35000

# MAIN
#==============================================================================
planets = open('Stats/' + planets_path, 'r')
planets_lines = planets.readlines()

# Initialize lists
Rp = []
Porb = []
ang_sep = []
stype = []
dist = []
Finc = []
nstar = []
obs_time = []
F560W = []
F1000W = []
F1500W = []
# Go through all lines
i = 0
for line in planets_lines:
    line_temp = line.split('\t')
    line_temp[-1] = line_temp[-1].strip()
    if (i == 0):
        col_Rp = np.where(np.array(line_temp) == 'Rp')[0][0]
        col_Porb = np.where(np.array(line_temp) == 'Porb')[0][0]
        col_ang_sep = np.where(np.array(line_temp) == 'ang_sep')[0][0]
        col_stype = np.where(np.array(line_temp) == 'stype')[0][0]
        col_dist = np.where(np.array(line_temp) == 'dist')[0][0]
        col_Finc = np.where(np.array(line_temp) == 'Finc')[0][0]
        col_nstar = np.where(np.array(line_temp) == 'nstar')[0][0]
        col_obs_time = np.where(np.array(line_temp) == 'obs_time')[0][0]
        col_F560W = np.where(np.array(line_temp) == 'F560W')[0][0]
        col_F1000W = np.where(np.array(line_temp) == 'F1000W')[0][0]
        col_F1500W = np.where(np.array(line_temp) == 'F1500W')[0][0]

    else:
        Rp += [float(line_temp[col_Rp])]
        Porb += [float(line_temp[col_Porb])]
        ang_sep += [float(line_temp[col_ang_sep])]
        stype += [str(line_temp[col_stype])]
        dist += [float(line_temp[col_dist])]
        Finc += [float(line_temp[col_Finc])]
        nstar += [int(float(line_temp[col_nstar]))]
        obs_time += [int(float(line_temp[col_obs_time]))]
        F560W += [float(line_temp[col_F560W])]
        F1000W += [float(line_temp[col_F1000W])]
        F1500W += [float(line_temp[col_F1500W])]
    i += 1
planets.close()
print(len(F1500W))
# Convert lists to arrays (more handy)
Rp = np.array(Rp)
Porb = np.array(Porb)
ang_sep = np.array(ang_sep)
stype = np.array(stype)
dist = np.array(dist)
Finc = np.array(Finc)
nstar = np.array(nstar)
obs_time = np.array(obs_time)
F560W = np.array(F560W)
F1000W = np.array(F1000W)
F1500W = np.array(F1500W)

#Get the limit sensitivity for each planet in the 10 micron filter 

lim_sens = sensF1000W/ np.sqrt(obs_time/exptime_default)


## Make masks for observable planets

mask2_F560W = ang_sep >= IWA*(lam_F560W/lam_F1000W) # inner working angle @ 5.6 microns
mask2_F1000W = ang_sep >= IWA # inner working angle @ 10 microns
mask2_F1500W = ang_sep >= IWA*(lam_F1500W/lam_F1000W) # inner working angle @ 15 microns

mask4 = F560W >= lim_sens * sensF560W / sensF1000W # sensitivity remapped to 5.6 micron 
mask5 = F1000W >= lim_sens # sensitivity
mask6 = F1500W >= lim_sens * sensF1500W / sensF1000W # sensitivity remapped to 15 micron


mask560 = (mask2_F560W & mask4) 
mask1000 = (mask2_F1000W & mask5)  
mask1500 = (mask2_F1500W & mask6) 

#HIST2D
#==============================================================================
# Bins

masks = [mask560, mask1000, mask1500]
hist2d_labels = ["56_microns", "10_microns", "15_microns"]
for it, mask in enumerate(masks):
  Rp_hist2d = Rp[mask]
  Finc_hist2d = Finc[mask]
  bins_Rp = np.array([0.50, 1.25, 2.00, 4.00, 6.00])
  bins_Rp_fine = np.hstack((np.linspace(0.50, 1.25, 4, endpoint=False), np.hstack((np.linspace(1.25, 2.00, 4, endpoint=False), np.hstack((np.linspace(2.00, 4.00, 4, endpoint=False), np.linspace(4.00, 6.00, 5)))))))
  bins_Finc = np.logspace(-2, 3, 11)
  bins_Finc_fine = np.logspace(-2, 3, 51)
  bins_Finc_labels = ['$10^{-2}$', '$10^{-1.5}$', '$10^{-1}$', '$10^{-0.5}$', '$10^{0}$', '$10^{0.5}$', '$10^{1}$', '$10^{1.5}$', '$10^{2}$', '$10^{2.5}$', '$10^{3}$']

  # Make histograms

  H, _, _ = np.histogram2d(Rp_hist2d, Finc_hist2d, bins=[bins_Rp, bins_Finc])
  H_fine, _, _ = np.histogram2d(Rp_hist2d, Finc_hist2d, bins=[bins_Rp_fine, bins_Finc_fine])

  xdata = np.zeros(len(Finc_hist2d))
  for i in range(len(Finc_hist2d)):
     if (Rp_hist2d[i] >= np.min(bins_Rp) and Rp_hist2d[i] <= np.max(bins_Rp)):
         if (Finc_hist2d[i] < np.min(bins_Finc)):
             xdata[i] = -1
         elif (Finc_hist2d[i] > np.max(bins_Finc)):
             xdata[i] = -1
         else:
             for j in range(len(bins_Finc)-1):
                 if (Finc_hist2d[i] >= bins_Finc[j] and Finc_hist2d[i] < bins_Finc[j+1]):
                     xdata[i] = 0.5+j
                     break
  ydata = np.zeros(len(Rp_hist2d))
  for i in range(len(Rp_hist2d)):
     if (Finc_hist2d[i] >= np.min(bins_Finc) and Finc_hist2d[i] <= np.max(bins_Finc)):
         if (Rp_hist2d[i] < np.min(bins_Rp)):
             ydata[i] = -1
         elif (Rp_hist2d[i] > np.max(bins_Rp)):
             ydata[i] = -1
         else:
             for j in range(len(bins_Rp)-1):
                 if (Rp_hist2d[i] >= bins_Rp[j] and Rp_hist2d[i] < bins_Rp[j+1]):
                     ydata[i] = 0.5+j
                     break

  # Set figure dimensions
  left, width, bottom, height = 0.10, 0.65, 0.10, 0.65
  left_h = bottom_h = left+width+0.025
  rect_hist2 = [left, bottom, width, height]
  rect_histx = [left, bottom_h, width, 0.175]
  rect_histy = [left_h, bottom, 0.175, height]

  # Make figure
  plt.figure()
  ax_hist2 = plt.axes(rect_hist2)
  ax_histx = plt.axes(rect_histx)
  ax_histy = plt.axes(rect_histy)

  # 
  nf = NullFormatter()
  ax_histx.xaxis.set_major_formatter(nf)
  ax_histy.yaxis.set_major_formatter(nf)

  # Plot 2d histogram
  ax_hist2.imshow(np.flipud(H_fine/float(nMC)), cmap='YlGn', aspect='auto', interpolation='none', extent=[0, H.shape[1], 0, H.shape[0]])
  x, y = np.meshgrid(np.linspace(0.5, len(bins_Finc)-1.5, len(bins_Finc)-1), np.linspace(0.5, len(bins_Rp)-1.5, len(bins_Rp)-1))
  for x_val, y_val in zip(x.flatten(), y.flatten()):
     ax_hist2.text(x_val, y_val, '%.1f' % (float(H[int(y_val), int(x_val)])/float(nMC)), va='center', ha='center', size=10)

  # Set axes properties
  ax_hist2.set_xticks(range(len(bins_Finc)))
  ax_hist2.set_xticklabels(bins_Finc_labels, fontsize=12)
  ax_hist2.set_yticks(range(len(bins_Rp)))
  ax_hist2.set_yticklabels(bins_Rp, fontsize=12)
  ax_hist2.set_xlabel('Stellar insolation [Solar constants]', fontsize=12)
  ax_hist2.set_ylabel('Radius [Earth radii]', fontsize=12)

  # Add grid
  for i in range(len(bins_Finc)-2):
     ax_hist2.axvline(x=i+1, linestyle=':', linewidth=0.5, color='k')
  for i in range(len(bins_Rp)-2):
     ax_hist2.axhline(y=i+1, linestyle=':', linewidth=0.5, color='k')

  # Plot 1d histogram
  ax_histx.hist(xdata, bins=range(len(bins_Finc)), range=(0, len(bins_Finc)), weights=np.ones_like(xdata)/float(nMC), histtype='step', color='darkgreen')

  # Set axes properties
  ax_histx.set_xlim(0, len(bins_Finc)-1)
  ax_histx.set_ylim(bottom = 0)
  ax_histx.set_xticks(range(len(bins_Finc)))
  ax_histx.set_yticks([0, 60, 120, 180])
  ax_histx.set_yticklabels([0, 60, 120, 180], fontsize=12)
  #ax_histx.set_yticks([0, 80, 160, 240])
  #ax_histx.set_yticklabels([0, 80, 160, 240], fontsize=12)
  ax_histx.yaxis.grid(b=True, which='both')

  # Plot 1d histogram
  ax_histy.hist(ydata, bins=range(len(bins_Rp)), range=(0, len(bins_Rp)), weights=np.ones_like(ydata)/float(nMC), histtype='step', color='darkgreen', orientation='horizontal')
         
  # Set axes properties
  ax_histy.set_ylim(0, len(bins_Rp)-1)
  ax_histy.set_xlim(left = 0)
  ax_histy.set_yticks(range(len(bins_Rp)))
  ax_histy.set_xticks([0, 60, 120, 180])
  ax_histy.set_xticklabels([0, 60, 120, 180], fontsize=12)
  #ax_histx.set_yticks([0, 80, 160, 240])
  #ax_histx.set_yticklabels([0, 80, 160, 240], fontsize=12)
  ax_histy.xaxis.grid(b=True, which='both')

  plt.savefig('Plots/' + planets_path[0:-4]  + ' hist2d' + hist2d_labels[it] + '.pdf', bbox_inches='tight')
  plt.close()



# WATERFALL
#==============================================================================


master_mask = (mask2_F560W & mask4) | (mask2_F1000W & mask5) | (mask2_F1500W & mask6) # consider all planets which can be observed in at least 1 filter


# Make 1d histogram
total = np.zeros(2)
total[0] = np.sum(master_mask & (stype == 'M'))
total[1] = np.sum(master_mask & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
total /= float(nMC)

band560 = np.zeros(2)
band560[0] = np.sum((mask560 & np.logical_not(mask1000) & np.logical_not(mask1500)) & (stype == 'M'))
band560[1] = np.sum((mask560 & np.logical_not(mask1000) & np.logical_not(mask1500)) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band560 /= float(nMC)
band1000 = np.zeros(2)
band1000[0] = np.sum((np.logical_not(mask560) & mask1000 & np.logical_not(mask1500)) & (stype == 'M'))
band1000[1] = np.sum((np.logical_not(mask560) & mask1000 & np.logical_not(mask1500)) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band1000 /= float(nMC)
band1500 = np.zeros(2)
band1500[0] = np.sum((np.logical_not(mask560) & np.logical_not(mask1000) & mask1500) & (stype == 'M'))
band1500[1] = np.sum((np.logical_not(mask560) & np.logical_not(mask1000) & mask1500) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band1500 /= float(nMC)

band5601000 = np.zeros(2)
band5601000[0] = np.sum((mask560 & mask1000 & np.logical_not(mask1500)) & (stype == 'M'))
band5601000[1] = np.sum((mask560 & mask1000 & np.logical_not(mask1500)) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band5601000 /= float(nMC)
band10001500 = np.zeros(2)
band10001500[0] = np.sum((np.logical_not(mask560) & mask1000 & mask1500) & (stype == 'M'))
band10001500[1] = np.sum((np.logical_not(mask560) & mask1000 & mask1500) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band10001500 /= float(nMC)
band5601500 = np.zeros(2)
band5601500[0] = np.sum((mask560 & np.logical_not(mask1000) & mask1500) & (stype == 'M'))
band5601500[1] = np.sum((mask560 & np.logical_not(mask1000) & mask1500) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band5601500 /= float(nMC)

band56010001500 = np.zeros(2)
band56010001500[0] = np.sum((mask560 & mask1000 & mask1500) & (stype == 'M'))
band56010001500[1] = np.sum((mask560 & mask1000 & mask1500) & ((stype == 'F') | (stype == 'G') | (stype == 'K')))
band56010001500 /= float(nMC)

# Make figure
plt.figure()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# Plot 1d histogram
plt.bar(0, height=np.sum(total), width=0.8, color='orange', edgecolor='black', label='FGK')
plt.bar(0, height=total[0], width=0.8, color='red', edgecolor='black', label='M')

plt.bar(1, height=np.sum(total), width=0.8, color='orange', edgecolor='black')
plt.bar(1, height=np.sum(total)-band560[1], width=0.8, color='red', edgecolor='black')
plt.bar(1, height=np.sum(total)-np.sum(band560), width=0.9, color='white', edgecolor='none')
plt.bar(2, height=np.sum(total)-np.sum(band560), width=0.8, color='orange', edgecolor='black')
plt.bar(2, height=np.sum(total)-np.sum(band560)-band1000[1], width=0.8, color='red', edgecolor='black')
plt.bar(2, height=np.sum(total)-np.sum(band560)-np.sum(band1000), width=0.9, color='white', edgecolor='none')
plt.bar(3, height=np.sum(total)-np.sum(band560)-np.sum(band1000), width=0.8, color='orange', edgecolor='black')
plt.bar(3, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-band1500[1], width=0.8, color='red', edgecolor='black')
plt.bar(3, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500), width=0.9, color='white', edgecolor='none')

plt.bar(4, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500), width=0.8, color='orange', edgecolor='black')
plt.bar(4, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-band5601000[1], width=0.8, color='red', edgecolor='black')
plt.bar(4, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000), width=0.9, color='white', edgecolor='none')
plt.bar(5, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000), width=0.8, color='orange', edgecolor='black')
plt.bar(5, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-band10001500[1], width=0.8, color='red', edgecolor='black')
plt.bar(5, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500), width=0.9, color='white', edgecolor='none')
plt.bar(6, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500), width=0.8, color='orange', edgecolor='black')
plt.bar(6, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-band5601500[1], width=0.8, color='red', edgecolor='black')
plt.bar(6, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-np.sum(band5601500), width=0.9, color='white', edgecolor='none')

plt.bar(7, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-np.sum(band5601500), width=0.8, color='orange', edgecolor='black')
plt.bar(7, height=np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-np.sum(band5601500)-band56010001500[1], width=0.8, color='red', edgecolor='black')

plt.axhline(np.sum(total), 0.11/8., 1.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560), 1.11/8., 2.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560)-np.sum(band1000), 2.11/8., 3.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500), 3.11/8., 4.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000), 4.11/8., 5.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500), 5.11/8., 6.89/8., color='black')
plt.axhline(np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-np.sum(band5601500), 6.11/8., 7.89/8., color='black')

plt.axvline(0.5, linestyle='--', color='black')
plt.axvline(3.5, linestyle='--', color='black')
plt.axvline(6.5, linestyle='--', color='black')

# Annotate 1d histogram
plt.text(0, np.sum(total)+20, '%.0f' % (np.sum(total)), va='center', ha='center', size=12)
plt.text(1, np.sum(total)+20, '%.0f' % (np.sum(band560)), va='center', ha='center', size=12)
plt.text(2, np.sum(total)-np.sum(band560)+20, '%.0f' % (np.sum(band1000)), va='center', ha='center', size=12)
plt.text(3, np.sum(total)-np.sum(band560)-np.sum(band1000)+20, '%.0f' % (np.sum(band1500)), va='center', ha='center', size=12)
plt.text(4, np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)+20, '%.0f' % (np.sum(band5601000)), va='center', ha='center', size=12)
plt.text(5, np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)+20, '%.0f' % (np.sum(band10001500)), va='center', ha='center', size=12)
plt.text(6, np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)+20, '%.0f' % (np.sum(band5601500)), va='center', ha='center', size=12)
plt.text(7, np.sum(total)-np.sum(band560)-np.sum(band1000)-np.sum(band1500)-np.sum(band5601000)-np.sum(band10001500)-np.sum(band5601500)+20, '%.0f' % (np.sum(band56010001500)), va='center', ha='center', size=12)

# Set axes properties
plt.xlim([-0.5, 7.5])
plt.ylim([0, np.sum(total)])
#plt.ylim([0, 600])
plt.gca().tick_params(axis='x', direction='in', pad=-10)
plt.gca().set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
plt.gca().set_xticklabels(['Total', '5.6', '10', '15', '5.6 & 10', '10 & 15', '5.6 & 15', 'All'], fontsize=12, rotation=90, va='bottom')
plt.xlabel('Bands in which planets are detectable [microns]', fontsize=12)
plt.ylabel('Expected number of detectable planets', fontsize=12)
plt.gca().yaxis.grid(True)
plt.legend(loc='upper right', fontsize=12)

plt.savefig('Plots/' + planets_path[0:-4]  + 'waterfall.pdf', bbox_inches='tight')
plt.close()


# HABITABLE
#==============================================================================
# Make masks for habitable planets
mask_hab = (0.5 < Rp) & (Rp < 1.5) & (0.37 < Finc) & (Finc < 1.75)

base_F = np.sum(master_mask & mask_hab & (stype == 'F'))/float(nMC)
base_G = np.sum(master_mask & mask_hab & (stype == 'G'))/float(nMC)
base_K = np.sum(master_mask & mask_hab & (stype == 'K'))/float(nMC)
base_M = np.sum(master_mask & mask_hab & (stype == 'M'))/float(nMC)

mask4 = F560W >= lim_sens * sensF560W / sensF1000W/10 # sensitivity remapped to 5.6 micron 
mask5 = F1000W >= lim_sens/10 # sensitivity
mask6 = F1500W >= lim_sens * sensF1500W / sensF1000W/10 # sensitivity remapped to 15 micron

# mask4 = F560W >= sensF560W/10. # sensitivity
# mask5 = F1000W >= sensF1000W/10. # sensitivity
# mask6 = F1500W >= sensF1500W/10. # sensitivity
master_mask = (mask2_F560W & mask4) | (mask2_F1000W & mask5) | (mask2_F1500W & mask6) # consider all planets which can be observed in at least 1 filter
master_mask = master_mask  # consider all planets up to 6 Earth radii and within outer working angle

sens10_F = np.sum(master_mask & mask_hab & (stype == 'F'))/float(nMC)
sens10_G = np.sum(master_mask & mask_hab & (stype == 'G'))/float(nMC)
sens10_K = np.sum(master_mask & mask_hab & (stype == 'K'))/float(nMC)
sens10_M = np.sum(master_mask & mask_hab & (stype == 'M'))/float(nMC)

mask4 = F560W >= lim_sens * sensF560W / sensF1000W/100 # sensitivity remapped to 5.6 micron 
mask5 = F1000W >= lim_sens/100 # sensitivity
mask6 = F1500W >= lim_sens * sensF1500W / sensF1000W/100 # sensitivity remapped to 15 micron

# mask4 = F560W >= sensF560W/100. # sensitivity
# mask5 = F1000W >= sensF1000W/100. # sensitivity
# mask6 = F1500W >= sensF1500W/100. # sensitivity

master_mask = (mask2_F560W & mask4) | (mask2_F1000W & mask5) | (mask2_F1500W & mask6) # consider all planets which can be observed in at least 1 filter
master_mask = master_mask  # consider all planets up to 6 Earth radii and within outer working angle

sens100_F = np.sum(master_mask & mask_hab & (stype == 'F'))/float(nMC)
sens100_G = np.sum(master_mask & mask_hab & (stype == 'G'))/float(nMC)
sens100_K = np.sum(master_mask & mask_hab & (stype == 'K'))/float(nMC)
sens100_M = np.sum(master_mask & mask_hab & (stype == 'M'))/float(nMC)

# Make figure
plt.figure()
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', handleheight=2.5)

# Plot 1d histogram
plt.bar(0, height=sens100_F, width=0.8, color='lightyellow', edgecolor='grey', hatch='O', zorder=0)
plt.bar(0, height=sens100_F, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(0, height=sens10_F, width=0.8, color='lightyellow', edgecolor='grey', hatch='o', zorder=0)
plt.bar(0, height=sens10_F, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(0, height=base_F, width=0.8, color='lightyellow', edgecolor='grey', hatch='.', zorder=0)
plt.bar(0, height=base_F, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(1, height=sens100_G, width=0.8, color='yellow', edgecolor='grey', hatch='O', zorder=0)
plt.bar(1, height=sens100_G, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(1, height=sens10_G, width=0.8, color='yellow', edgecolor='grey', hatch='o', zorder=0)
plt.bar(1, height=sens10_G, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(1, height=base_G, width=0.8, color='yellow', edgecolor='grey', hatch='.', zorder=0)
plt.bar(1, height=base_G, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(2, height=sens100_K, width=0.8, color='orange', edgecolor='grey', hatch='O', zorder=0)
plt.bar(2, height=sens100_K, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(2, height=sens10_K, width=0.8, color='orange', edgecolor='grey', hatch='o', zorder=0)
plt.bar(2, height=sens10_K, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(2, height=base_K, width=0.8, color='orange', edgecolor='grey', hatch='.', zorder=0)
plt.bar(2, height=base_K, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(3, height=sens100_M, width=0.8, color='red', edgecolor='grey', hatch='O', zorder=0)
plt.bar(3, height=sens100_M, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(3, height=sens10_M, width=0.8, color='red', edgecolor='grey', hatch='o', zorder=0)
plt.bar(3, height=sens10_M, width=0.8, color='none', edgecolor='black', zorder=1)
plt.bar(3, height=base_M, width=0.8, color='red', edgecolor='grey', hatch='.', zorder=0)
plt.bar(3, height=base_M, width=0.8, color='none', edgecolor='black', zorder=1)

# Annotate 1d histogram
text = plt.text(0, sens100_F+2, '%.0f' % sens100_F, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(0, sens10_F+2, '%.0f' % sens10_F, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(0, base_F+2, '%.0f' % base_F, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(1, sens100_G+2, '%.0f' % sens100_G, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(1, sens10_G+2, '%.0f' % sens10_G, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(1, base_G+2, '%.0f' % base_G, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(2, sens100_K+2, '%.0f' % sens100_K, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(2, sens10_K+2, '%.0f' % sens10_K, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(2, base_K+2, '%.0f' % base_K, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(3, sens100_M+2, '%.0f' % sens100_M, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(3, sens10_M+2, '%.0f' % sens10_M, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
text = plt.text(3, base_M+2, '%.0f' % base_M, va='center', ha='center', size=12)
text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# Set axes properties
plt.xlim([-0.5, 3.5])
plt.ylim([0, 60])
#plt.ylim([0, 100])
plt.gca().set_xticks([0, 1, 2, 3])
plt.gca().set_xticklabels(['F', 'G', 'K', 'M'], fontsize=12)
plt.ylabel('Expected number of det. hab. planets', fontsize=12)
plt.gca().yaxis.grid(True)

# Set legend properties
base_patch = patches.Patch(facecolor='white', edgecolor='black', hatch='.', label='Baseline')
sens10_patch = patches.Patch(facecolor='white', edgecolor='black', hatch='o', label='Sens$\cdot$10')
sens100_patch = patches.Patch(facecolor='white', edgecolor='black', hatch='O', label='Sens$\cdot$100')
plt.legend(handles=[base_patch, sens10_patch, sens100_patch], loc='upper left')

plt.savefig('Plots/' + 'habitable.pdf', bbox_inches='tight')
plt.close()