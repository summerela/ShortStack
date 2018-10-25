'''
sequencer.py

'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re

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
        
        return nuc_df
    
    @jit         
    def pivot_counts(self, nuc_df):
        
       count_df = nuc_df.pivot_table(
           index=["FeatureID", "pos"],
           columns = "nuc",
           values= "nuc_count",
           fill_value=0
           )
       return count_df 
   
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
        
    def return_seq(self, count_df, ftm_df):
        
        # get nucleotide with max count for each position
        df = pd.DataFrame(count_df.apply(self.get_consensus, axis=1)).reset_index()
        df.columns = ["FeatureID", "pos", "nuc"]
        
        # convert nuc max calls to a consensus sequence and add to dataframe
        df['feature_seq'] = df.groupby(['FeatureID'])['nuc'].transform(lambda x: ''.join(x))
        df = df[['FeatureID','feature_seq']].drop_duplicates()
        
        # subset ftm calls for just gene names 
        ftm = ftm_df[["FeatureID", "gene"]]
        ftm = ftm.drop_duplicates()
        
        # merge ftm gene names with consensus seqs
        seq_df = df.merge(ftm, on="FeatureID")
        
        return seq_df
    
    def validate_seq(self, fasta_df, seq_df):
        
        # join sequence df with ref info for hd and vcf
        seq_out = "{}/seqs.tsv".format(self.output_dir)
        final_df = seq_df.merge(fasta_df, left_on="gene", right_on="id")
        final_df.to_csv(seq_out, sep="\t")

#         final_df["seq_hd"] = final_df.apply(lambda x: cpy.calc_hamming(x.feature_seq,
#                                                                        x.seq,
#                                                                        len(x.feature_seq)))
         
        
        return final_df        
    
    
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
        seq_df = self.return_seq(pivot_counts, self.ftm_counts)
        
        # validate consensus against ref seq
        print("Validating consensus sequences...\n")
        consensus_df = self.validate_seq(self.fasta_df, seq_df)
        
        
        


        
        
        
