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
                 calls,
                 fasta_df,
                 out_dir):
    
        self.calls = calls
        self.fasta_df = fasta_df
        self.output_dir = out_dir
        
    def split_targets(self, calls, fasta_df):
        '''
        purpose: split target reads into individual bases
        input: ftm_counts and hamming_df
        output: individual bases, positions and counts per base
        '''   
        
        tiny_fasta = fasta_df[["id", "start"]]
        
        # pull start position from fasta_df
        nuc_df = calls.merge(tiny_fasta, left_on="gene", right_on="id") 
        nuc_df["pos"] = nuc_df.pos.astype(int) + nuc_df.start.astype(int) 
        nuc_df.drop(["start", "id"], axis=1, inplace=True)

        nuc_df = pd.DataFrame(
            nuc_df.apply(
                lambda row: pd.Series(
                    [(row.FeatureID, row.gene, row.pos + i, c, row.bc_count) for i, c in enumerate(row.Target)]
                ),
                axis=1)
                .stack().tolist(),
            columns=["FeatureID", "gene", "pos", "nuc", "count"])
        
        return nuc_df
    
    @jit
    def base_counts(self, nuc_df):
        
        # sum counts per nuc for each position
        nuc_df["nuc_count"] = nuc_df.groupby(["FeatureID", "pos", "nuc"])["count"].transform("sum")
        nuc_df = nuc_df.sort_values(["FeatureID", "pos", "nuc"])
        
        # remove duplicate rows from transform function
        nuc_df = nuc_df.drop("count", axis=1)
        nuc_df = nuc_df.drop_duplicates().reset_index(drop=True)
        
        # save nucleotide counts for testing
        nuc_out = "{}/nuc_counts.tsv".format(self.output_dir)
        nuc_df.to_csv(nuc_out, sep="\t", index=False)
        
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
           index=["FeatureID", "gene", "pos"],
           columns = "nuc",
           values= "nuc_count",
           fill_value=0
           )
       
       # get nucleotide with max count for each position
       count_df = pd.DataFrame(count_df.apply(self.get_consensus, axis=1)).reset_index()
       count_df.columns = ["FeatureID", "gene", "pos", "nuc"]

       return count_df 
   
    def format_fasta(self, fasta_df):
        fasta_seqs = pd.DataFrame(
            fasta_df.apply(
                lambda row: pd.Series(
                    [(row.id, (int(row.start) + i), c) for i, c in enumerate(row.seq)]
                ),
                axis=1)
                .stack().tolist(),
            columns=["gene", "pos", "nuc"])
        
        return fasta_seqs
     
    def check_ref(self, fasta_seqs, count_df):
        annotated = pd.merge(count_df, fasta_seqs, 
                        on=["gene", "pos"], 
                        how="outer",
                        suffixes=["feature", "ref"])
        annotated.fillna('N', inplace=True)
        
    def return_seq(self, count_df):
        
        # convert nuc max calls to a consensus sequence and add to dataframe
        count_df['feature_seq'] = count_df.groupby(['FeatureID'])['nuc'].transform(lambda x: ''.join(x))

        count_df = count_df[['FeatureID','gene', 'feature_seq']].drop_duplicates()
        
        return count_df
    
    @jit
    def get_ftmID(self, fasta_df, count_df):
        
        # join sequence df with ref info for hd and vcf
        final_df = count_df.merge(fasta_df, left_on="gene", right_on="id")
        
        return final_df
    
    def validate_seq(self, final_df):

        # calculate HD between ref and observed seq
        final_df["seq_qual"] = final_df.apply(lambda x: cpy.calc_seq_hamming(x.feature_seq,
                                                                       x.seq), axis=1)      
        
        return final_df
    
    @jit
    def get_muts(self, final_df):

        # break apart seq validation tuple 
        final_df[['seq_hd', 'seq_alts']] = final_df.seq_qual.apply(pd.Series)
        
        # pull out seqs with hd>0 for vcf
        muts = final_df[final_df.seq_hd > 0]
        
        return final_df, muts

    
    def main(self):
 
        # convert target sequence to base and position with count
        print("Matching targets with position..\n")
        split_targets = self.split_targets(self.calls, self.fasta_df)
           
        # sum counts of each nucleotide by base
        print("Summing counts for each base...\n")
        base_counts = self.base_counts(split_targets)
          
        # pivot count table for selecting max counts
        print("Formatting results...\n")
        pivot_counts = self.pivot_counts(base_counts)
        
        # testing        
#         fasta_seqs = self.format_fasta(self.fasta_df)
#         verified = self.check_ref(fasta_seqs, pivot_counts)
        
        # return consensus sequence
        print("Determining consensus sequence...\n")
        seq_df = self.return_seq(pivot_counts)
         
        # get FTM ID data for pulling out reference seq
        print("Comparing results to reference sequence...\n")
        final_df = self.get_ftmID(self.fasta_df, seq_df)
          
        # validate consensus against ref seq
        print("Validating consensus sequences...\n")
        consensus_df = self.validate_seq(final_df)

        # create vcf output
        print("Locating mutated sequences...\n")
        final_df, muts = self.get_muts(consensus_df)


        


        
        
        


        
        
        
