'''
module for matching s6 calls to probes
'''

import pandas as pd
import numpy as np
from numba import jit

pd.options.mode.chained_assignment = None

class Encode_files():
    
    # instance parameters
    def __init__(self, s6_df, encoding_df, output_dir):
        self.s6_df = s6_df
        self.col_names =  [x for x in self.s6_df.columns]
        self.encoding_df = encoding_df
        self.out_dir = output_dir
    
    @jit        
    def map_basecalls(self, s6_df, encoding_df):
        '''
        Purpose: Replace color codes from s6 file with targets from encoding.txt
        Input: 
            s6_df = s6 dataframe
            encoding_df = encoding df containing target and color_code columns
        Output: s6_df with target seqs instead of color codes
        '''

        print("Matching basecalls with color encoding")
        off_file = "{}/invalids.tsv".format(self.out_dir)

        # match targets to base calls by merging s6_df and encoding_df
        encoded_df = self.s6_df.merge(encoding_df, how='left', on=["PoolID", "BC"])
        encoded_df.sort_values(["FeatureID", "Target"], inplace=True)
        encoded_df.reset_index(inplace=True, drop=True)
        
        # check for and store info on base calls not valid in encoding file
        parity_df = encoded_df[encoded_df['Target'].isnull()]
        parity_df["filter"] = "invalid"
        parity_df.drop("Target", axis=1, inplace=True)
        
        if not parity_df.empty:
            parity_df.to_csv(off_file, sep="\t", index=False)
        return encoded_df, parity_df
    
    def remove_invalidBC(self, encoded_df):
        
        try:
            # remove basecalls not valid in encoding file and reset index
            encoded_df = encoded_df[~encoded_df['Target'].isnull()].reset_index(drop=True)
            assert encoded_df.shape[0] > 1
        except Exception as e:
            error_msg = "No valid basecalls. Check that encoding file is accurate. \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)

        return encoded_df
    
    def main(self, s6_df, encoded_df):
        mapped_df, parity_df = self.map_basecalls(self.s6_df, self.encoding_df)
        enocoded_df = self.remove_invalidBC(mapped_df).reset_index(drop=True)
        return enocoded_df, parity_df