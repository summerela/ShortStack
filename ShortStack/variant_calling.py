'''
find_snv.py
locate potential vars and add information to variant graph
'''

import logging
import cython_funcs as cpy
from numba import jit
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

class FindVars():
    
    def __init__(self, ngrams, non_matches, fasta_df, kmer_length,
                 mutation_vcf, output_dir, deltaZ, max_hamming_dist):
        self.ngrams = ngrams.reset_index()
        self.non_matches = non_matches
        self.fasta_df = fasta_df
        self.kmer_length = kmer_length
        self.mutation_vcf = mutation_vcf
        self.output_dir = output_dir
        self.count_file = "{}/snv_normalized_counts.tsv".format(self.output_dir)
        self.deltaz_threshold = deltaZ
        self.max_hamming_dist = max_hamming_dist

    def calc_hamming(self):
        '''
        purpose: calculate hamming distance between basecalls and targets
        input: ngrams created in ftm.create_ngrams(), non_
        output: dataframe of potential vars's with hamming <= max_hamming_dist
        '''

        # calc hamming distance for non-perfect matches
        matches = self.non_matches.Target.apply(lambda bc: self.ngrams.ngram.\
                                apply(lambda x: cpy.calc_hamming(bc, x)))
    
        # add labels for columns and index so we retain information
        matches.columns = self.ngrams.ngram.astype(str) + \
                         ":" + self.ngrams.gene.astype(str) + \
                         ":" + self.ngrams.pos.astype(str)       
        matches.index = self.non_matches.FeatureID.astype(str) + ":" +  self.non_matches.Target.astype(str)
        
        # filter dataframes to get hamming dist <= self.max_hamming_dist
        df2 = matches.stack()
        
        # find all variants that overlap with max_hamming_dist
        var_df = df2[(df2 <= self.max_hamming_dist)]
       
        # return ham>1 <= kmer <= (self.kmer_length - 1) = less strict
#         var_df = df2[df2.between(1, (self.kmer_length - 1))]

        return var_df
    
    @jit
    def parse_vars(self, var_df):
        '''
        purpose: reformat the hamming distance df 
        input: dataframe of reads with hamming distance <= self.max_hamming_dist
        output: reshaped hamming_df 
        '''
        
        var_df = var_df.reset_index()
        var_df.columns = ["Target", "Gene", "Hamming"]
        
        # split var_df index into columns to add back into dataframe
        targets = pd.DataFrame(var_df.Target.str.split(':',1).tolist())
        targets.columns = ["FeatureID", "BC"]
        matches =  pd.DataFrame(var_df.Gene.str.split(':',1).tolist())
        matches.columns = ["Target_Match", "Gene"]
        genes = pd.DataFrame(matches.Gene.str.split(':',1).tolist())
        genes.columns = ["Gene", "Pos"]
        matches.drop("Gene", axis=1, inplace=True)
        
        # concatenate dataframe back together
        vars = pd.concat([targets, matches, genes], axis=1)
    
        # join vars with fasta_df to get ref position info
        hamming_df = vars.merge(self.fasta_df[["chrom", "start", "id"]], \
                                 left_on="Gene", right_on="id")     
        
        # calculate starting position of match on reference, subtract 1 for 0 based indexing
        hamming_df["var_pos"] = (hamming_df["start"].astype(int) \
                             - hamming_df["Pos"].astype(int)) - 1
        
        # drop unnecessary columns
        hamming_df.drop(["id", "Pos", "start"], inplace=True, axis=1) 

        return hamming_df
    
    def count_vars(self, var_df):
        '''
        purpose: count and normalize multi-mapped reads
        input: reshaped hamming_df for vars's
        output: normalized counts for vars
        '''
        
        # flag targets that are multi-mapped
        var_df['multi_mapped'] = np.where(var_df.groupby(["FeatureID", "Gene", "BC"]).var_pos\
                                    .transform('nunique') > 1, "T", '')
        
        # separate multi and non multi mapped reads
        non = var_df[var_df["multi_mapped"] != "T"]\
            .drop("multi_mapped", axis=1)   
        multi = var_df[var_df["multi_mapped"] == "T"]\
            .drop("multi_mapped", axis=1)

         # add counts to non multi mapped reads
        non["count"] = non.groupby(["FeatureID", "Gene", "BC"])["var_pos"]\
            .transform("count")
        
        # add counts to multi-mapped reads with normaliztion 
        multi["count"] = multi.groupby(["FeatureID", "Gene", "BC"])["var_pos"]\
              .transform(lambda x: 1/x.nunique())

        # join the multi and non back together
        counts = pd.concat([multi, non], axis=0).sort_values("FeatureID")
        
        # save raw counts to file
        counts.to_csv(self.count_file, sep="\t", index=False, header=True)
        
        # drop target column to pass to ftm
        counts = counts.reset_index(drop=True)
        
        return counts
    
        
    def main(self):
        
        print("Calculating hamming distance on unmatched reads...\n")
        # calc hamming distance
        var_df = self.calc_hamming()
        
        print("Locating potential variants...\n")
        # parse var_df dataframe
        hamming_df = self.parse_vars(var_df)
        
        print("Normalizing variant counts...\n")
        # normalize vars counts
        var_counts = self.count_vars(hamming_df)
        
        return var_counts
        
        
        
        
        