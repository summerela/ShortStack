'''
consensus.py

'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re
import swifter
from collections import defaultdict, Counter
from pathlib import Path
import dask
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
import psutil

# import logger
log = logging.getLogger(__name__)

class Consensus():
    
    def __init__(self, 
                 molecule_seqs,
                 fasta_df,
                 out_dir,
                 cpus):
    
        self.molecule_seqs = molecule_seqs
        self.fasta_df = fasta_df
        self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
        self.output_dir = out_dir
        self.cpus = cpus
    
    def count_molecules(self, x):
        
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
                
                if nuc.upper() == "N":
                    nuc_weight = 0
                else:
                    nuc_weight = 1
                
                nucs[key][nuc] +=  nuc_weight
                
        nucs = pd.DataFrame(nucs).T
       
        return nucs
    

    def get_MAF(self, group):
        
        sample_size = group.shape[0]
        
        # replace NaN with 0
        freqs = group.fillna(0)
        freqs = freqs.reset_index()
        freqs.columns = ["id", "pos", "A", "C", "G", "T", "N"]
        freqs["samp_size"]= sample_size
        
        # convert to dask for speed
        freq_dd = dd.from_pandas(freqs, npartitions=self.cpus)
         
        # create list of col names
        freq_cols = freq_dd.columns
         
        # if N column not found for chunk, add column for downstream concat
        if "N" not in freq_cols:
            freq_dd["N"] = 0.0
        else:
            freq_dd["N"] = freq_dd["N"].astype(float)
          
        # get frequency per position
        freq_counts = freq_dd[["A", "C", "G", "T", "N"]].apply(lambda x: round((x/sample_size * 100),3),
                                                            meta={
                                                                  'A': 'float',
                                                                  'C':'float',
                                                                  'G':'float',
                                                                  'T':'float',
                                                                  'N':'float'}, 
                                                                 axis=1)
        
        # drop raw counts
        freq_ids = freqs.drop(["A", "C", "G", "T", "N"], axis=1)
        
        # merge keys and counts back together
        freq_df =dd.concat([freq_ids, freq_counts], axis=1).compute()
        
        # reorder columns
        freq_df = freq_df[["id", "pos", "samp_size", "A", "T", "G", "C", "N"]]
        
        return freq_df
    
    @jit
    def pos_match(self, group):
        
        # get starting position for each group id
        freqs = dd.merge(group, self.fasta_df[["id", "start", "chrom"]],
                         on="id")
        
        # label with correct genomic location
        freqs["pos"] = freqs.pos.astype(int) + freqs.start.astype(int)
        freqs = freqs.drop("start", axis=1)
        
        # parse output
        freqs = freqs.reset_index(drop=True)
        freqs = freqs[["id", "chrom", "pos", "samp_size", "A", "T", "G", "C", "N"]]
       
        return freqs

    @jit  
    def main(self):
        
        maf_list = []
                
        freqs = self.molecule_seqs.groupby("region").apply(self.count_molecules)
        
        maf = freqs.groupby("region").apply(self.get_MAF)
        
        maf_df = maf.groupby("region").apply(self.pos_match)
        
        # save molecule sequences to file
        consensus_out = Path("{}/consensus_maf.tsv".format(self.output_dir))
        maf_df.to_csv(consensus_out, sep="\t", index=False) 
