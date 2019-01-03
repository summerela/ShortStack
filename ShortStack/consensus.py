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
from dask.distributed import Client
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
                 client,
                 cpus):
    
        self.molecule_seqs = molecule_seqs
        self.fasta_df = fasta_df
        self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
        self.output_dir = out_dir
        self.client = client
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
       
        return nucs
    
    def get_MAF(self, groupID, nuc_counts, sample_size):
        
        # get allele count per position
        freqs = pd.DataFrame(nuc_counts).T
        freqs = dd.from_pandas(freqs, npartitions=self.cpus)
        
        # replace NaN with 0
        freqs = freqs.fillna(0)

        # get frequency per position
        freqs = freqs.loc[:, freqs.columns != 'pos'].apply(lambda x: round((x/sample_size * 100),3),
                                                            meta={
                                                                  'A': 'float',
                                                                  'C':'float',
                                                                  'G':'float',
                                                                  'N':'float',
                                                                  'T':'float'}, 
                                                                 axis=1)
        freqs = freqs.reset_index()
        freqs.columns = ["pos", "A", "C", "G", "N", "T"]
        freqs["id"] = str(groupID)
        freqs = freqs[["id", "pos", "A", "C", "G", "T", "N"]]
        
        # keep in memory for downstream parsing
        freqs = self.client.compute(freqs)
        freqs = freqs.result()
        
        return freqs
    
    @jit
    def pos_match(self, freqs, fasta_df):
        
        # get starting position for each group id
        freqs = dd.merge(freqs, fasta_df[["id", "start", "chrom"]],
                         on="id")
        
        # label with correct genomic location
        freqs["pos"] = freqs.pos.astype(int) + freqs.start.astype(int)
        freqs = freqs.drop("start", axis=1)
        
        return freqs

        
    def main(self):
        
        maf_list = []
        
        # get allele frequencies for each feature
        for feature, data in self.molecule_seqs.groupby("region"):
            
            group_size = data.shape[0]

            groupID = ''.join(data.region.unique())
            
            # get base counts
            freqs = self.count_molecules(data)
            
            maf = self.get_MAF(feature, freqs, group_size)
            
            # format base counts to allele frequencies
            maf_df = self.pos_match(maf, self.fasta_df)
            maf_df = maf_df[["id", "chrom", "pos", "A", "T", "G", "C", "N"]]
            
            # add regional df to maf_list
            maf_list.append(maf_df)
        
        # concatenate maf dataframes for each region
        maf_dfs = dd.multi.concat(maf_list,  interleave_partitions=True)
        maf_out = maf_dfs.compute()

        # save molecule sequences to file
        consensus_out = Path("{}/consensus_maf.tsv".format(self.output_dir))
        maf_out.to_csv(consensus_out, sep="\t", index=False) 
