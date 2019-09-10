import matplotlib.pyplot as plt
import pickle as pl
import numpy as np

def dump_pickle(filename, fig):
    full_filename = '{}{}{}'.format('../Data/', filename, '.pickle')
    pl.dump(fig, open(full_filename, 'wb'))
    
def load_pickle(filename):
    full_filename = '{}{}{}'.format('../Data/', filename, '.pickle')
    fig_handle = pl.load(open(full_filename,'rb'))
    fig_handle.show()
    

load_pickle('09-09-2019-TestDataSet-MADGE')