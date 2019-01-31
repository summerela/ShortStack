'''
consensus.py

'''
import sys, re, swifter, psutil, os
import warnings
from dask.dataframe.methods import sample
from Bio.Nexus.Trees import consensus
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
                 client,
                 today):
    
        self.molecule_df = molecule_df
        self.ref_df = ref_df
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
        self.today = today
    
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

        # if molecule weight < 1 then set base to N
        molecule_df["base"][molecule_df.weight < 1] = "N"
        
        # set all molecule weights to N
        molecule_df["weight"] = 1
        
        # group by region and sum weights
        molecule_df["base_weight"] = molecule_df.groupby(["region", "pos", "base"])["weight"].transform('sum')
        molecule_df = molecule_df.drop(["weight", "FeatureID"],  axis=1)
        
        # divide count by sample size to get frequency
        molecule_df = dd.merge(molecule_df, sample_sizes, 
                                        on="region",
                                        how="left")

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
                                      values = ['base_weight'],
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
        
        return consensus_df
 
    @jit   
    def find_MAF(self, consensus_df): 
         
        # find most frequent allele for each position
        consensus_df["max_nuc"] = consensus_df[["A", "T", "G", "C", "N"]].idxmax(axis=1)
         
        # find rows where the max_nuc does not equal ref_base
        mafs = consensus_df[consensus_df.ref_base != consensus_df.max_nuc]
        
        # calc row total calls
        mafs["DP"] = mafs[["A", "T", "G", "C", "N"]].sum(axis=1)
        
        # add placeholder for QV and NL values
        mafs["QV"] = 30
        mafs["NL"] = 5
        
        # add (num_ref calls, num_alt calls) 
        mafs["AD"] =  tuple(zip(mafs.lookup(mafs.index,mafs.ref_base),
                                mafs.lookup(mafs.index,mafs.max_nuc)))
        
        # calculate variant allele freq
        mafs["VF"] = round(((mafs.lookup(mafs.index,mafs.max_nuc)/mafs["sample_size"]) * 100),2)
        
        # add alt nuc
        mafs["ALT"] = mafs["max_nuc"]
        mafs["REF"] = mafs["ref_base"]
        mafs["QUAL"]= "."
        mafs["FILTER"] = "."
        
        # parse into vcf format
        mafs["INFO"] = "AD=" + mafs.AD.astype(str) + ";" + \
                       "DP=" + mafs.DP.astype(str) + ";" + \
                       "QV=" + mafs.QV.astype(str) + ";" + \
                       "NL=" + mafs.NL.astype(str) + ";" + \
                       "VF=" + mafs.VF.astype(str)
        
        # remove unnecessary columns              
        mafs = mafs.drop(["DP", "NL", "AD", "VF", "QV", 
                          "A", "T", "G", "C", "N",
                          "max_nuc", "ref_base"], axis=1)
        
        # reorder columns
        mafs  = mafs[["chrom", "pos", "REF", "ALT", "QUAL", "FILTER", "INFO"]]
        
        return mafs
         
    def make_vcf(self, maf_df):  
        
        # write VCF header to file
        today_file = "{}_ShortStack.vcf".format(self.today)
        output_VCF = os.path.join(self.output_dir, today_file)
        with open(output_VCF, 'w') as vcf:
            vcf.write("##fileformat=VCFv4.2\n")
            vcf.write("##source=ShortStackv0.1.1\n")
            vcf.write("##reference=GRCh38\n")
            vcf.write("##referenceMod=file://data/scratch/reference/reference.bin\n")
            vcf.write("##fileDate:{}\n".format(self.today))
            vcf.write("##comment='Unreleased dev version. See Summer or Nicole with questions.'\n")
            vcf.write("##FILTER=<ID=Pass,Description='All filters passed'>\n")
            vcf.write("##INFO=<ID=GENE, Number=1, Type=String, Description='Gene name'>\n")
            vcf.write("#CHROM    POS    ID    REF    ALT    QUAL    FILTER    INFO\n")
             
        # parse dataframe for vcf output
        maf_df.to_csv(output_VCF, index=False, sep="\t", header=False, mode='a')                
        
        
        
    def main(self):
        
        # set all molecule weights to 1 and sum
        consensus_weights = self.weigh_molecules(self.molecule_df)
        
        # parse results and save to file
        consensus_df = self.parse_consensus(consensus_weights)
        
        # find rows where the major allele varies from the ref allele
        maf_df = self.find_MAF(consensus_df)
        raise SystemExit(maf_df.head(100))
        
        self.make_vcf(maf_df)

        
    

