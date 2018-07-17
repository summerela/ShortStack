'''
module for matching s6 calls to probes
'''

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

class Encode_files():
    
    # instance parameters
    def __init__(self, s6_df, encoding_df):
        self.s6_df = s6_df
        self.col_names =  [x for x in self.s6_df.columns]
        self.encoding_df = encoding_df
            
    def map_basecalls(self, s6_df, encoding_df, dropna=False):
        '''
        Purpose: Replace color codes from s6 file with targets from encoding.txt
        Input: 
            s6_df = s6 dataframe
            encoding_df = encoding df containing target and color_code columns
        Output: s6_df with target seqs instead of color codes
        '''
        
        print("Matching basecalls with color encoding")

        # match targets to base calls by merging s6_df and encoding_df
        encoded_df = self.s6_df.merge(encoding_df, how='inner', on=["PoolID", "BC"])
        encoded_df = encoded_df[["FeatureID", "Target"]]
        encoded_df.sort_values(["FeatureID", "Target"], inplace=True)
        # check for and store info on base calls not valid in encoding file
        parity_df = encoded_df[encoded_df['Target'].isnull()]
        parity_df["filter"] = "bc_parity"
        parity_df.drop("Target", axis=1, inplace=True)
        
        try:
            # remove basecalls not valid in encoding file and reset index
            encoded_df = encoded_df[~encoded_df['Target'].isnull()].reset_index(drop=True)
            assert encoded_df.shape[0] > 1
        except Exception as e:
            error_msg = "No valid basecalls. Check that encoding file is accurate. \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)

        return encoded_df, parity_df
