import pickle
import math
import time
import glob
import os
import pandas as pd
import numpy as np
from cryodrgn import mrc
from cryodrgn import analysis
from cryodrgn import utils

### Define relevant variables
results_dir = '../' #change as necessary
output_dir = './' #change as necessary
subunits = ['subunit1', 'subunit2'] #change as necessary
mask_names = {'subunit1': '/path/to/subunit1_mask.mrc', 'subunit2': '/path/to/subunit2_mask.mrc'}
Apix = 1 #pixel size in the final downsampled volumes
flip = False #whether or not to flip the handedness of the generated volumes
df_path = 'kmeans500_df.csv' #dataframe written out by cryodrgn_viz.ipynb, where each particle is a row
cuda = 0 #change cuda device for volume generation if desired
bin_thr = None #set to None if you don't want to binarize, otherwise provide a threshold for binarization
downsample = 64 #boxsize at which to generate volumes; boxsize 64 is recommended
epoch = 29
tmp_vol_dir = './tmp' #temporary folder where volumes will be generated
###

def generate_volumes(zvalues, outdir, **kwargs):
    '''Helper function to call cryodrgn eval_vol and generate new volumes'''
    np.savetxt(f'{outdir}/zfile.txt', zvalues)
    analysis.gen_volumes(f'{results_dir}weights.29.pkl',
                         f'{results_dir}config.pkl',
                         f'{outdir}/zfile.txt',
                         f'{outdir}', **kwargs)
    return

t0 = time.time()

#read in indices and masks
with open(f'{results_dir}z.{epoch}.pkl','rb') as f:
    z = pickle.load(f)
df = pd.read_csv(df_path, index_col = 0)
all_inds = df.index
batch_size = 1000
mask_dict = {i: mrc.parse_mrc(mask_names[i])[0].flatten() for i in mask_names}

#define volume generation choices
if not os.path.exists(tmp_vol_dir):
    os.mkdir(tmp_vol_dir)


#initialize occupancy arrays
subunit_occupancies = {i: np.array([]) for i in mask_names}

#iterate through all particles in batches
for i in range(math.ceil(len(all_inds)/batch_size)):
    ind_selected = all_inds[i*batch_size:(i+1)*batch_size]
    
    #generate volumes within batch
    generate_volumes(z[ind_selected], tmp_vol_dir, Apix=Apix, flip=flip, downsample=downsample, cuda=cuda)
    
    #read in and measure volumes
    vol_list = np.sort(glob.glob(tmp_vol_dir + '/*.mrc'))
    vol_arrays = {j: np.zeros((batch_size, len(mask_dict[j]))) for  j in mask_dict}
    for k, vol in enumerate(vol_list):
        data = mrc.parse_mrc(vol)[0].flatten()
        if bin_thr is not None:
            data = np.where(data > bin_thr, 1, 0)
        for j in mask_dict:
            vol_arrays[j][k] = data*mask_dict[j]
    
    for j in subunit_occupancies:
        subunit_occupancies[j] = np.concatenate((subunit_occupancies[j], vol_arrays[j].sum(axis = 1)))
    
    #delete volumes
    for file in vol_list:
        os.remove(file)
    os.remove(vol_list[0].split('vol')[0] + 'zfile.txt')
    
    #estimate time remaining
    dt = (time.time() - t0)
    est_time = dt/((i+1)*batch_size)*(len(all_inds)-(i+1)*batch_size)
    print(f'{dt/60} min elapsed; estimated time remaining = {est_time/60} min')
    
for j in subunit_occupancies:
    utils.save_pkl(subunit_occupancies[j], f'{output_dir}otf_{j}_occupancies.pkl')
