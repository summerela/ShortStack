'''
sequencer.py

'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re
import swifter

# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 ftm_counts,
                 no_calls,
                 hamming_df,
                 fasta_df,
                 out_dir):
    
        self.ftm_counts = ftm_counts
        self.no_calls = no_calls
        self.hamming_df = hamming_df
        self.fasta_df = fasta_df
        self.output_dir = out_dir
        
    def split_targets(self, ftm_counts):        
        
        nuc_df = pd.DataFrame(
            ftm_counts.apply(
                lambda row: pd.Series(
                    [(row.FeatureID, row.pos + i, c, row.bc_count) for i, c in enumerate(row.Target)]
                ),
                axis=1)
                .stack().tolist(),
            columns=["FeatureID", "pos", "nuc", "count"])
        
        return nuc_df
    
    @jit
    def base_counts(self, nuc_df):
        
        # sum counts per nuc for each position
        nuc_df["nuc_count"] = nuc_df.groupby(["FeatureID", "pos", "nuc"])["count"].transform("sum")
        nuc_df = nuc_df.sort_values(["FeatureID", "pos", "nuc"])

        # remove duplicate rows from transform function
        nuc_df = nuc_df.drop("count", axis=1)
        nuc_df = nuc_df.drop_duplicates()
        
        # save consensus df for testing
        consensus_out = "{}/consensus_counts.tsv".format(self.output_dir)
        nuc_df.to_csv(consensus_out, sep="\t")
        
        return nuc_df
    
    @jit
    def get_consensus(self, row): 

        # find max value for each position
        max_value = row.max()
        nuc = row.idxmax()
    
        # if there is a tie, return N for nocall
        if (row == max_value).sum() == 1:
            return nuc
        else:
            return "N"
    
    @jit         
    def pivot_counts(self, nuc_df):
        
       count_df = nuc_df.pivot_table(
           index=["FeatureID", "pos"],
           columns = "nuc",
           values= "nuc_count",
           fill_value=0
           )
       
       # get nucleotide with max count for each position
       count_df = pd.DataFrame(count_df.apply(self.get_consensus, axis=1)).reset_index()
       count_df.columns = ["FeatureID", "pos", "nuc"]
       
       return count_df 
        
    def return_seq(self, count_df):
        
        # convert nuc max calls to a consensus sequence and add to dataframe
        count_df['feature_seq'] = count_df.groupby(['FeatureID'])['nuc'].transform(lambda x: ''.join(x))
        
        return count_df
    
    @jit
    def get_ftmID(self, ftm_counts, fasta_df, count_df):
        count_df = count_df[['FeatureID','feature_seq']].drop_duplicates()
        
        # subset ftm calls for just gene names 
        ftm = ftm_counts[["FeatureID", "gene"]]
        ftm = ftm.drop_duplicates()
        
        # merge ftm gene names with consensus seqs
        seq_df = count_df.merge(ftm, on="FeatureID")
        
        # join sequence df with ref info for hd and vcf
        final_df = seq_df.merge(fasta_df, left_on="gene", right_on="id")
        
        return final_df
    
    def validate_seq(self, final_df):

        # calculate HD between ref and observed seq
        final_df["seq_qual"] = final_df.apply(lambda x: cpy.calc_seq_hamming(x.feature_seq,
                                                                       x.seq),
                                                                       axis=1)
        return final_df  
    
    @jit
    def format_vcf(self, final_df):

        #tired just saving to a file until tomorrow
        vcf_out = "{}/consensus_seq.tsv".format(self.output_dir)
        final_df.to_csv(vcf_out, sep="\t")

    
    def main(self):

        # convert target sequence to base and position with count
        print("Matching targets with position..\n")
        split_targets = self.split_targets(self.ftm_counts)
        
        # sum counts of each nucleotide by base
        print("Summing counts for each base...\n")
        base_counts = self.base_counts(split_targets)
        
        # pivot count table for selecting max counts
        print("Formatting results...\n")
        pivot_counts = self.pivot_counts(base_counts)

        # return consensus sequence
        print("Determining consensus sequence...\n")
        seq_df = self.return_seq(pivot_counts)

        # get FTM ID data for pulling out reference seq
        print("Comparing results to reference sequence...\n")
        final_df = self.get_ftmID(self.ftm_counts, self.fasta_df, seq_df)
        
        # validate consensus against ref seq
        print("Validating consensus sequences...\n")
        consensus_df = self.validate_seq(final_df)
        
        # create vcf output
        print("Creating VCF file...\n")
        vcf_df = self.format_vcf(consensus_df)


        
        
        


        
        
        
