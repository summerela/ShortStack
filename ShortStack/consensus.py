'''
consensus.py

'''
import warnings
from dask.dataframe.methods import sample
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re
import swifter
from collections import defaultdict, Counter
from pathlib import Path
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
import psutil

# import logger
log = logging.getLogger(__name__)

class Consensus():
    
    def __init__(self, 
                 molecule_seqs,
                 ref_df,
                 out_dir,
                 cpus,
                 client):
    
        self.molecule_seqs = molecule_seqs
        self.ref_df = ref_df
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
    
    def count_molecules(self, x):
        
        sample_size = len(x)
        
        nucs = defaultdict(Counter)
        
        for edge in x.swifter.apply(
            lambda row: 
            [(i,
             c,
             1) for i,c in enumerate(row.seq)],
            axis=1):
            for tuple in edge:
                key = tuple[0]
                nuc = tuple[1]
                weight = tuple[2]
                
                nucs[key][nuc] +=  weight
        
        nucs = pd.DataFrame.from_dict(nucs,
                                      orient='index')
        nucs = nucs.reset_index(drop=False)
        nucs.columns = ["pos", "N", "G", "C", "A", "T"]

        nucs["samp_size"] = sample_size
        nucs = nucs.fillna(0)

        return nucs
    
    def get_MAF(self, group):
        
        # get frequency per position
        group[["A", "C", "G", "T", "N"]] = group[["A", "C", "G", "T", "N"]]\
            .apply(lambda x: round((x/group.samp_size * 100),3), axis=0)

        # reorder columns
        group = group[["region", "pos", "samp_size", "A", "T", "G", "C", "N"]]
                               
        return group
    
    @jit
    def pos_match(self, group, ref_df):
        
        # pull out region
        group_region = group.region.unique()[0]
         
        # pull out ref seqs that match group region
        refs = ref_df[ref_df.region == group_region]
         
        # correct starting position to reference
        start_pos = int(min(refs.pos))
        group["pos"] = group.pos.astype(int) + start_pos
         
        return group

    @jit  
    def main(self):
        
        # count nucleotides by base for each region      
        freqs = self.molecule_seqs.groupby("region").apply(self.count_molecules)
        freqs = freqs.reset_index()
        freqs = freqs.drop("level_1", axis=1)

        # get maf for each region
        maf = freqs.groupby("region").apply(self.get_MAF)
        
        # add reference position
        maf_df = maf.groupby("region").apply(self.pos_match, self.ref_df)
        maf_df = maf_df.reset_index(drop=True)
        
#         # prep dataframes for merge
#         maf_df["pos"] = maf_df.pos.astype('int')
#         maf_df = pd.DataFrame(maf_df)
#     
#         self.ref_df["pos"] = self.ref_df.pos.astype('int')
#         self.ref_df.reset_index(inplace=True, drop=True)
#         
#         # merge dataframes
#         seq_df = pd.merge(maf_df, self.ref_df,
#                           how='left',
#                         on=["region", "pos"])

        # save molecule sequences to file
        consensus_out = Path("{}/consensus_maf.tsv".format(self.output_dir))
        maf_df.to_csv(consensus_out, sep="\t", index=False) 
