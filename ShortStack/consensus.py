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
                
        nucs = pd.DataFrame(nucs).T
        nucs["samp_size"] = sample_size
        nucs = nucs.fillna(0)

        return nucs
    
    def get_MAF(self, group):
        
        # get frequency per position
        group[["A", "C", "G", "T", "N"]] = group[["A", "C", "G", "T", "N"]]\
            .apply(lambda x: round((x/group.samp_size * 100),3), axis=0)

        # reorder columns
        group = group[["region", "samp_size", "A", "T", "G", "C", "N"]]
        
        return group
    
    @jit
    def pos_match(self, group):
        
        # pull out region
        group_region = group.region.unique()[0]
        group = group.reset_index()
        group.columns = ["pos", "region", "samp_size",
                         "A", "T", "G", "C", "N"]
        
        # pull out ref seqs that match group region
        refs = self.ref_df[self.ref_df.region == group_region]
        
        # correct starting position to reference
        start_pos = int(min(refs.pos))
        group["pos"] = group.pos.astype(int) + start_pos
        
        # get starting position for each group id
        group["pos"] = group.pos.astype(int)
        refs["pos"] = refs.pos.astype(int)
        freqs = pd.merge(group, refs,
                         on=["region", "pos"])
       
        return freqs

    @jit  
    def main(self):
        
        maf_list = []
           
        # count nucleotides by base for each region      
        freqs = self.molecule_seqs.groupby("region").apply(self.count_molecules)
        freqs = freqs.reset_index()

        # get maf for each region
        maf = freqs.groupby("region").apply(self.get_MAF)
        
        maf_df = maf.groupby("region").apply(self.pos_match)

        # save molecule sequences to file
        consensus_out = Path("{}/consensus_maf.tsv".format(self.output_dir))
        maf_df.to_csv(consensus_out, sep="\t", index=False) 
