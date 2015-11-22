from __future__ import print_function
import h5py
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Wirtes out text file having song properties from hdf5 file')
parser.add_argument('path', type=str, nargs=1,
                        help='writee_properties <absolute path to hdf5 file>')

args    = parser.parse_args()
h5_path = args.path[0]
h5file  = h5py.File(h5_path)
songAllProps = h5file['metadata']['songs']
# Write a funtion that replaces NaNs -----
intProp     = ['song_id','artist_familiarity','artist_hotttnesss']
songProps   = np.array([songAllProps[intProp[0]],songAllProps[intProp[1]],songAllProps[intProp[2]]]).transpose()
np.savetxt('Song_Properties.txt',songProps,delimiter='\t',fmt='%s')
