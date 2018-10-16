import pandas as pd
import numpy as np
import ftm 
from numba import jit
import cython_funcs as cpy
import swifter
import distance

class CalcHamming():
    
    def __init__(self,
                 ngrams,
                 non_matches,
                 no_calls,
                 encoding,
                 max_ham_dist,
                 num_cores
                 ): 
        self.ngrams = ngrams
        self.non_matches = non_matches
        self.no_calls = no_calls
        self.encoded_df = encoding  # do we need this here? 
        self.max_ham_dist = max_ham_dist
        self.num_cores = num_cores
    
    def calc_hamming(self, ngram_df):
        '''
        purpose: calculate hamming distance between basecalls and targets
        input: ngrams created in ftm.create_ngrams(), 
        output: dataframe of potential vars's with hamming <= max_hamming_dist
        '''
        
        # reset index to get ngrams column back in df
        ngram_df.reset_index(drop=False, inplace=True)
        
        # get unique set from each list
        ngram_list = list(set(ngram_df.ngram))
        target_list = list(set(self.no_calls.Target))
        
        hd = lambda x, y: cpy.calc_hamming(x,y, self.max_ham_dist)
        hamming_list = [hd(x, y) for y in target_list for x in ngram_list]
        
        
        return hamming_list
     
    @jit
    def parse_hamming(self, hamming_list):
        '''
        purpose: reformat the hamming distance df 
        input: dataframe of reads with hamming distance <= self.max_hamming_dist
        output: reshaped hamming_df 
        '''
        
        hamming_df = pd.DataFrame(hamming_list, columns=["GeneKmer", "BaseCall", "hamming"])
        hamming_df = hamming_df[hamming_df.hamming != "X"]
        raise SystemExit(hamming_df)
        
        # filter hamming df by self.max_hamming_dist and remove self-matches covered in perfect_matches
        hamming_df = hamming_df[(hamming_df.hamming > 0) & (hamming_df.hamming <= self.max_ham_dist)]
        
        # left join non_matches to hamming_df to get FeatureID info for each target/from column
        hd2 = self.no_calls.merge(hamming_df, how="inner", left_on="Target", right_on="from")
        
        # left join hd2 with ngrams to get target information for the "from" column
        hd3 = hd2.merge(self.ngrams, how="inner", left_on="to", right_on="ngram")

        
        raise SystemExit(hd3.head(20))

#         # add labels for columns and index so we retain information
#         hamming_df.columns = ngrams.ngram.astype(str) + \
#                          ":" + ngrams.gene.astype(str) + \
#                          ":" + ngrams.pos.astype(str)       
#         hamming_df.index = self.encoded_df.FeatureID.astype(str) + ":" +  \
#             self.encoded_df.Target.astype(str)  
#          
#         # reshape dataframe for parsing    
#         hamming2 = hamming_df.stack()  
#         hamming2 = hamming2.reset_index()
#         hamming2.columns = ["Target", "Ref", "Hamming"] 
         
        
        
#           
#         # concatenate dataframe back together
#         kmers = pd.concat([targets, matches, genes, hamming2.Hamming], axis=1)
#         print(kmers.head())
#         return kmers
     
#     @jit
#     def add_ref_pos(self, kmers):
# 
#         # join kmers with fasta_df to get ref position info
#         hamming_df = kmers.merge(self.fasta_df[["chrom", "start", "id"]], \
#                                  left_on="Ref", right_on="id")     
#         
#         # calculate starting position of match on reference, subtract 1 for 0 based indexing
#         hamming_df["pos"] = (hamming_df["start"].astype(int) \
#                              - hamming_df["Pos"].astype(int)) - 1
#         
#         # drop unnecessary columns
#         hamming_df.drop(["id", "Pos", "start"], inplace=True, axis=1) 
#         hamming_df.reset_index(inplace=True, drop=True)
#         
#         return hamming_df
#     
#     @jit
#     def locate_multiMapped(self, covered_df):
#         '''
#         purpose: normalize multi-mapped reads as total count in gene/total count all genes
#         input: diversity filtered match dataframe build in align.diversity_filter()
#         output: counts normalized counts dataframe to pass to algin.score_matches()
#             raw counts saved to tsv file
#         '''
#         
#         # flag targets that are multi-mapped
#         covered_df['multi_mapped'] = np.where(covered_df.groupby(["FeatureID", "gene", "BC"]).pos\
#                                     .transform('nunique') > 1, "T", '')
# 
#         # separate multi and non multi mapped reads
#         non = covered_df[covered_df["multi_mapped"] != "T"]\
#             .drop("multi_mapped", axis=1)   
#         multi = covered_df[covered_df["multi_mapped"] == "T"]\
#             .drop("multi_mapped", axis=1)
#         
#         # non multi mapped reads get a count of 1 each
#         non["count"] = 1
#         
#         return multi, non
    
    def main(self):
        
        # calculate hamming distance on non-perfect matches
        print("Calculating hamming distance for non-perfect matches...\n")
        hamming_df = self.calc_hamming(self.ngrams)
        
        # parse the hamming dataframe
        print("Parsing the hamming dataframe...\n")
        hamming_df = self.parse_hamming(hamming_df)
        
    