'''
module for matching s6 calls to probes
'''
import sys, os, logging
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from numba import jit
import dask.dataframe as dd
import multiprocessing as mp

pd.options.mode.chained_assignment = None

# import logger
log = logging.getLogger(__name__)

class Encode_files():
    
    # instance parameters
    def __init__(self, s6_df, encoding_df, output_dir, cpus, client):
        self.s6_df = s6_df
        self.col_names =  [x for x in self.s6_df.columns]
        self.encoding_df = encoding_df
        self.out_dir = output_dir
        self.invalids_file = os.path.join(self.out_dir, "invalids")
        self.cpus = cpus
        self.client = client
    
    @jit        
    def map_basecalls(self, s6_df, encoding_df):
        '''
        Purpose: Replace color codes from s6 file with targets from encoding.txt
        Input: 
            s6_df = s6 dataframe
            encoding_df = encoding df containing target and color_code columns
        Output: s6_df with target seqs instead of color codes
        '''
        
        # dask decided not to join well on index, which is stupid
        s6_df["idx"] = s6_df.BC.astype(str) + ":" + s6_df.pool.astype(str)
        encoding_df["idx"] = encoding_df.BC.astype(str) + ":" + encoding_df.PoolID.astype(str)
        
        s6_df["idx"] = s6_df.idx.astype(str)
        encoding_df["idx"] = encoding_df.idx.astype(str)
        
        # match targets to base calls by merging s6_df and encoding_df
        encoded_df = dd.merge(s6_df, encoding_df, 
                              how='left',
                              on='idx')
        
        # drop duplicated columns
        encoded_df = encoded_df.drop(["PoolID", "BC_y", "idx"], axis=1)
        encoded_df = encoded_df.rename(columns={"BC_x":"BC"})
        
        # check for and store info on base calls not valid in encoding file
        parity_df = encoded_df[encoded_df['Target'].isnull()]
        
        if len(parity_df > 0):
            parity_df.to_parquet(self.invalids_file, engine='fastparquet')
         
        return encoded_df, parity_df
    
    @jit
    def remove_invalidBC(self, encoded_df):
        '''
        purpose: remove barcodes that did not match a target and return dataframe
        input:
        output:
        '''
        
        # remove basecalls not valid in encoding file and reset index
        encoded_df = encoded_df[~encoded_df['Target'].isnull()]
        encoded_df["bc_length"]= encoded_df.bc_length.astype(int)
        
        assert len(encoded_df) != 0, "No valid barcodes found in encoding file."

        return encoded_df
    @jit
    def main(self, s6_df, encoded_df):
        mapped_df, parity_df = self.map_basecalls(self.s6_df, self.encoding_df)
        enocoded_df = self.remove_invalidBC(mapped_df)
        return enocoded_df, parity_df