'''
consensus.py

'''
import sys, re, swifter, psutil, os
import warnings
from dask.dataframe.methods import sample
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions

# import logger
log = logging.getLogger(__name__)

class Consensus():
    
    def __init__(self, 
                 molecule_df,
                 ref_df,
                 out_dir,
                 cpus,
                 client):
    
        self.molecule_df = molecule_df
        self.ref_df = ref_df
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
    
    @jit    
    def weigh_molecules(self, molecule_df):
        '''
        purpose: sum weights for each base/position
        input: molecule_df created as output of sequenceer.py
        output: molecule_df grouped by region with a sum of weights at eaach pos
        '''
        
        # calculate sample size for each region
        size_df = dd.from_pandas(molecule_df[["FeatureID", "region"]].drop_duplicates(),
                                 npartitions=self.cpus)
        sample_sizes = size_df.groupby('region')['FeatureID'].count().reset_index()
        sample_sizes.columns = ["region", "sample_size"]

        # set per molecule weights to 1, unless < 1 then set base to N
        molecule_df["base"][molecule_df.weight < 1] = "N"
        molecule_df["weight"][molecule_df.weight > 1] = 1
        
        # group by region and sum weights
        molecule_df["base_weight"] = molecule_df.groupby(["region", "pos", "base"])["weight"].transform('sum')
        molecule_df = molecule_df.drop(["weight", "FeatureID"],  axis=1)
        
        # divide count by sample size to get frequency
        molecule_df = dd.merge(molecule_df, sample_sizes, 
                                        on="region",
                                        how="left")

        return molecule_df
    
    @jit
    def get_AF(self, molecule_df):
        '''
        purpose: divide weights by sample size to get allele frequency
        input: molecule_df of summed weights from weigh_molecules
        output: molecule_df with allele frequencies at each position
        '''
        
        # get frequency per position
        molecule_df["base_freq"] = (molecule_df["base_weight"]/molecule_df["sample_size"]) * 100
        molecule_df = molecule_df.drop("base_weight", axis=1)
        
        return molecule_df
    
    @jit
    def parse_consensus(self, molecule_df):
        '''
        purpose: convert format to one row per position for each molecule
        input: molecule_df 
        output: final consensus output to output_dir/consensus_counts.tsv
        '''
        
        molecule_df = molecule_df.compute()
        
        consensus_df = pd.pivot_table(molecule_df, 
                                      values = ['base_freq'],
                                      index=['region','chrom', 
                                             'pos', 'ref_base', 
                                             'sample_size'],
                                      columns='base',
                                      fill_value=0).reset_index()
        
        # sort and parse columns                             
        consensus_df.columns = consensus_df.columns.droplevel(1)
        consensus_df.columns = ["region", "chrom", "pos", "ref_base", "sample_size", "A", "C", "G", "N", "T"]
        consensus_df = consensus_df[["region", "chrom", "pos", "ref_base", "A", "T", "G", "C", "N", "sample_size"]]
        consensus_df = consensus_df.sort_values(by=["region", "pos"])
          
        # save to a file
        out_file = os.path.join(self.output_dir, "consensus_counts.tsv")   
        consensus_df.to_csv(out_file, sep="\t", index=False)                         
        
        
        
    def main(self):
        
        # set all molecule weights to 1 and sum
        consensus_weights = self.weigh_molecules(self.molecule_df)
        
        # calculate allele frequencies
        maf_df = self.get_AF(consensus_weights)
        
        # parse results and save to file
        self.parse_consensus(maf_df)
        
        
    

