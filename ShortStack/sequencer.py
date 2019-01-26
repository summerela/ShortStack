'''
sequencer.py

'''
import sys, warnings, logging, re, os, swifter, dask, psutil
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import multiprocessing as mp
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
from dask.bag.core import split
dask.config.set(scheduler='tasks')


# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 counts,
                 fasta_df,
                 out_dir,
                 cpus,
                 client):
    
        self.counts = counts[['FeatureID', 'region', 'pos', 'Target','bc_count']] # sequencing to region
        self.fasta_df = fasta_df
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
        
    def munge_counts(self, count_df):
        '''
        purpose: split bases into (pos, nuc) tuples
        input: all_counts df from ftm.py
        output: df of tuples for each feature
        '''

        # convert counts to dask dataframe
        counts = dd.from_pandas(self.counts, npartitions=self.cpus)
        counts["pos"] = counts.pos.astype(int)

        # split apart targets with position
        counts["nuc"] = counts.apply(lambda row: [(x + row.pos,y) for x,y in enumerate(row.Target)],
                                     axis=1)

        return counts
    
    @jit
    def split_bases(self, count_df):
        '''
        purpose: split feature tuple rows into one per column with 
            index retaining feature/region/count info
        input: munged counts df from munge_counts()
        output: df of one (pos, nuc) tuple per column per feature
        '''
        
        # set index to retain information
        count_df["idx"] = count_df.FeatureID + ":" + count_df.region + ":"  + count_df.bc_count.astype(str)
        count_df = count_df.set_index("idx")
        count_df = count_df.compute()

        # split tuples into columns and return back to dask
        base_df = pd.DataFrame(count_df["nuc"].values.tolist(), index=count_df.index).reset_index()
        base_df = dd.from_pandas(base_df, npartitions=self.cpus)
        
        return base_df
      
    def melt(self, df, id_vars=None, value_vars=None, var_name=None,
         value_name='value', col_level=None):
        '''
        purpose: dask doesn't have custom melt feature, wrapping pandas melt
        '''

        from dask.dataframe.core import no_default
        
        # map pandas melt to each chunk of df
        return df.map_partitions(pd.melt, meta=no_default, id_vars=id_vars,
                                value_vars=value_vars,
                                var_name=var_name, value_name=value_name,
                                col_level=col_level, token='melt')
    
    @jit    
    def melt_edges(self, df):
        '''
        purpose: unravel rows with multipel columns of tuples
            into df of one row per position per feature
        input: base_df created in split_bases
        output: df with one position per row, per feature
        '''
        
        df = self.melt(df, 
                       id_vars='idx',
                       value_name="nuc").drop("variable", axis=1)
        df = df.set_index("idx")
              
        return df
    
    @jit
    def split_tuples(self, df):
        '''
        purpose: split apart tuples and index to create final counts df for 
            base calling
        input: melted df created in melt_edges
        output: df with feature | region | chrom |  pos | nuc  | count 
        '''
        
        # break apart tuples
        df[["pos", "base"]] = pd.DataFrame(df['nuc'].values.tolist(), index=df.index)
        df = df.drop("nuc", axis=1).reset_index(drop=False)

        df = dd.from_pandas(df, npartitions=self.cpus)
        
        # do a lot of extra munging in dask to split idx to cols
        df['FeatureID'] = df["idx"].str.partition(":")[0]
        df['region'] = df['idx'].str.partition(":")[2]
        df['region'] = df['region'].str.partition(":")[0]
        df['counts'] = df['idx'].str.partition(":")[2]
        df['counts'] = df['counts'].str.partition(":")[2]
    
        df = df.drop("idx", axis=1)
        df["counts"] = df.counts.astype(float)

        return df
    
    @jit
    def calc_weights(self, df):
        '''
        purpose:sum the egde weights for each position
        input: result dataframe created in split_deges
        output: df with summed weights for each nuc per position
        '''
        df = df.compute()
        df["weight"] = df.groupby(["FeatureID", "region", "pos", "base"])["counts"].transform('sum')
        df = df.drop("counts", axis=1)
        df = df.drop_duplicates()
        
        return df
    
    def split_refs(self, fasta_df):
        '''
        purpose: split apart wt regions for bases with no coverage and vcf output
        input: fasta_df created in parse_input.py
        output: rows of (pos, nuc) tuples for each wt reference
        '''
        
        # subset fasta to wt windows
        region_fasta = self.fasta_df[self.fasta_df.id == self.fasta_df.region]
        
        # break apart sequences into list of ngrams
        fasta_dd = dd.from_pandas(region_fasta[["region", "chrom",
                                                 "start", "seq"]], 
                                  npartitions=self.cpus)
        
        # coerce start to int
        fasta_dd["start"] = fasta_dd.start.astype(int)
        
        # remove end line character
        fasta_dd["seq"] = fasta_dd.seq.str.strip()
        
        # break apart sequences intob bases
        fasta_dd["nucs"] = fasta_dd.seq.apply(lambda x: cpy.ngrams(x, 1))
        
        # split targets with position
        fasta_dd["bases"] = fasta_dd.apply(lambda row: [(x + row.start,y) for x,y in enumerate(row.nucs)],
                                     axis=1)
        
        return fasta_dd
    
    @jit
    def parse_refs(self, fasta_dd):
        '''
        purpose: pull apart the fasta_dd rows of tuples and parse into df
        input: fasta_dd created in split_refs
        output: fasta_df with one row per position for each input wt ref seq 
        '''
         
        # set index to retain information
        fasta_dd["idx"] = fasta_dd.region + ":" + fasta_dd.chrom
        fasta_dd = fasta_dd.set_index("idx")
        fasta_dd = fasta_dd.compute()
 
        # split tuples into columns 
        fasta_df = pd.DataFrame(fasta_dd["bases"].values.tolist(), index=fasta_dd.index).reset_index()
        fasta_dd = dd.from_pandas(fasta_df, npartitions=self.cpus)
         
        # split tuples per row into columns
        fasta_df = self.melt_edges(fasta_dd).compute()

        # get rid of NaN where some ref seqs were longer than others
        fasta_df = fasta_df.dropna(axis=0, how='any')
        
        fasta_df = pd.DataFrame(fasta_df['nuc'].values.tolist(), index=fasta_df.index).reset_index()
        fasta_df.columns = ["idx", "pos", "ref_base"]
        
        # split out the idx column
        fasta_df[["region", "chrom"]] = fasta_df['idx'].str.split(':',expand=True)
        fasta_df = fasta_df.drop("idx", axis=1)
        
        return fasta_df
    
    @jit
    def add_refs(self, group, ref_df):
        '''
        purpose: add ref base info and keep position with base of N for areas
            with no coverage
        input: fasta_df and count df
        output: count df with matching ref allele and N bases for positions
            with no coverage
        '''
        # pull out identifiers
        feature_id = group.FeatureID.unique()[0]
        group_region = group.region.unique()[0]
        
        # subset ref_df to seq window
        region_df = ref_df[ref_df.region == group_region]
        
        # pull out chrom to fill in blanks
        region_chrom = region_df.chrom.unique()[0]

        # outer join tables to keep missing bases
        group_df = group.merge(region_df, on=["region", "pos"],
                                   how='outer')
        
        # fill out missing columns for missing bases
        group_df["FeatureID"] = feature_id
        
        # change base to N for missing bases
        group_df["base"][pd.isnull(group_df['base'])] = "N"
        group_df["weight"][pd.isnull(group_df['weight'])] = 0
        group_df["chrom"][pd.isnull(group_df['chrom'])] = region_chrom
        
        # sort by position
        group_df = group_df.sort_values(by="pos").reset_index(drop=True)

        return group_df
    
    @jit
    def get_max(self, group):
        
        # filter for nuc with max weight at each position
        group["nuc_max"] = group.groupby(['pos'])['weight'].transform(max)
        group = group[group.weight == group.nuc_max]
        group = group.drop("nuc_max", axis=1)

        return group





    
    @jit     
    def main(self):
         
        # create list of position and base pair for each target
        counts = self.munge_counts(self.counts)
              
        # break bases up into separate columns
        munged_df = self.split_bases(counts)
              
        # melt columns of tuples into dataframes
        nuc_df = self.melt_edges(munged_df)
        nuc_df = nuc_df.compute()
              
        # split tuples of pos, nuc into columns
        base_df = self.split_tuples(nuc_df)
             
        # calculate edge weights
        weighted_df = self.calc_weights(base_df)
             
        # create ref seqs for bases with no coverage
        ref_dd = self.split_refs(self.fasta_df)
        ref_df = self.parse_refs(ref_dd)

        # add ref position of N for bases with no coverage
        full_covg_df = weighted_df.groupby("FeatureID").apply(self.add_refs, ref_df)
        # parse output
        full_covg_df.set_index("FeatureID", inplace=True)
        full_covg_df.reset_index(drop=False, inplace=True) 
        
        # get max weighted base and create final sequence
        max_df = full_covg_df.groupby("FeatureID").apply(self.get_max)
        # get featureID back into dataset
        max_df.set_index("FeatureID", inplace=True)
        max_df = max_df.reset_index(drop=False)
        max_df = max_df [["FeatureID", "region", "chrom", "pos", "base", "ref_base", "weight"]]
        max_df = max_df.sort_values(by=["FeatureID", "pos"])
        
        # save molecule counts to file
        mol_counts = os.path.join(self.output_dir, "molecule_counts.tsv")
        max_df.to_csv(mol_counts, sep="\t", index=False)

        
        raise SystemExit("Consensus sequence script still under construction. Per molecule sequencing successfully completed.")
        
        