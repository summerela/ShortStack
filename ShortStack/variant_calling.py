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
    
    def __init__(self, ngrams, non_matches, fasta_df, output_dir, deltaZ):
        self.ngrams = ngrams.reset_index()
        self.non_matches = non_matches
        self.fasta_df = fasta_df
        self.output_dir = output_dir
        self.count_file = "{}/snv_normalized_counts.tsv".format(self.output_dir)
        self.deltaz_threshold = deltaZ

    def calc_hamming(self):
        '''
        purpose: calculate hamming distance between basecalls and targets
        input: ngrams created in align.create_ngrams(), encoded_df
        output: dataframe of all potential vars's with hamming dist of 1
        '''
        
        # pull featureID out of the index from the encoding df
        encoding = self.non_matches.reset_index(drop=True)
     
        # calc hamming distance for non-perfect matches
        matches = encoding.Target.apply(lambda bc: self.ngrams.ngram.\
                                apply(lambda x: cpy.calc_hamming(bc, x)))
        
        # add labels for columns and index so we retain information
        matches.columns = self.ngrams.ngram.astype(str) + \
                         ":" + self.ngrams.gene.astype(str) + \
                         ":" + self.ngrams.pos.astype(str)       
        matches.index = encoding.FeatureID.astype(str) + ":" +  encoding.Target.astype(str)

        # merge dataframes to get hamming dist of 1
        df2 = matches.stack()
        var_df = df2[(df2 == 1)]
        return var_df
    
    @jit
    def parse_snvs(self, var_df):
        '''
        purpose: reformat the hamming distance df 
        input: dataframe of reads with hamming distance of 1
        output: reshaped hamming_df 
        '''
        
        var_df = var_df.reset_index()
        var_df.columns = ["Target", "Gene", "Hamming"]
        
        # split var_df into separate columns
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
        hamming_df = vars.merge(self.fasta_df[["chrom", "start", "id"]], left_on="Gene", right_on="id")     
        
        # calculate variant starting position on reference, subtract 1 for 0 based indexing
        hamming_df["var_pos"] = (hamming_df["start"].astype(int) - hamming_df["Pos"].astype(int)) - 1
        
        # drop unnecessary columns
        hamming_df.drop(["id", "Pos", "start"], inplace=True, axis=1) 
        
        return hamming_df
    
    def diversity_filter(self, input_df):
        '''
        purpose: filters out targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only targets per feature that meet threshold
        '''

        # group by feature ID and gene
        input_df["diverse"] = np.where(input_df.groupby(\
            ["FeatureID", "gene"])["pos"].transform('nunique') > self.diversity_threshold, "T", "")
        
        diversified = input_df[input_df.diverse == "T"]
        diversity_filtered = input_df[input_df.diverse != "T"]
        diversified.drop("diverse", inplace= True, axis=1)
        diversified.reset_index(inplace=True, drop=True)
        
        # notify user that certain features were filtered out for lack of diversity
        if diversity_filtered.shape[0] > 1:
            diversity_filtered.drop("diverse", inplace=True, axis=1)
            filter_message = "The following calls were filtered out for being below the feature diversity threshold of {}:\n {}\n".\
                format(self.diversity_threshold, diversity_filtered)
            print(filter_message)
                
        return diversified
    
    def count_snvs(self, var_df):
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
    
#     def return_indels(self, counts_df, non_matches):
#         counts_df["id"] = counts_df.FeatureID.astype(str) + "_" + counts_df.BC.astype(str)
#         indels = non_matches[~non_matches.id.isin(counts_df.id)]
#         print(indels.head())
#         print(non_matches.head())
#         raise SystemExit(counts_df.head())
        
        
    def main(self):
        
        print("Calculating hamming distance on unmatched reads...\n")
        # calc hamming distance
        var_df = self.calc_hamming()
        
        print("Locating potential variants...\n")
        # parse var_df dataframe
        hamming_df = self.parse_snvs(var_df)
        
        print("Normalizing variant counts...\n")
        # normalize vars counts
        snv_counts = self.count_snvs(hamming_df)
        
#         print("Saving unmatched features...\n")
#         indels = self.return_indels(snv_counts, self.non_matches)
        
        return snv_counts
        
        
        
        
        