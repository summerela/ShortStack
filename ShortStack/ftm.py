'''
align.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches
- creates voting table and counts perfect matches for FTM
- returns table of counting stats that is saved to output folder
- returns voting table from first round of FTM
'''

# from http.cookiejar import unmatched
import logging

from scipy.spatial import distance
from scipy import stats
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import swifter
from numpy import broadcast
from itertools import count

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, coverage_threshold, 
                 kmer_size, max_hamming_dist,
                 output_dir, diversity_threshold,
                 qc_outfile, run_info_file, num_cores):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.coverage_threshold = coverage_threshold
        self.max_hamming_dist = max_hamming_dist
        self.kmer_size = kmer_size
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.qc_outfile = qc_outfile
        self.gene_count_file = "{}/perfect_normCounts.tsv".format(self.output_dir)
        self.ftmQC_filter_file = "{}/ftm_qc_filter.tsv".format(self.output_dir)
        self.run_info_file = run_info_file
        self.num_cores = num_cores
    
    @jit    
    def target_count(self):
        '''
        purpose: count number of targets found by position within each feature
        input: encoded_df of feature ID and encoded color call
        output: raw_counts.tsv file of raw counts
        '''
        # create output file path for raw counts
        counts_outfile = "{}/all_target_counts.tsv".format(self.output_dir)
        
        # sort encoded dataframe by featureID
        self.encoded_df.sort_values("FeatureID", inplace=True)
        # count number of targets found in each barcode
        feature_counts = self.encoded_df.groupby(["FeatureID", "Target"]).size().reset_index()
        # update columnnames        
        feature_counts.columns = ["FeatureID", "Target", "count"]
        
        # save raw count to file
        feature_counts.to_csv(counts_outfile, sep="\t", index=None)
    
    def create_ngrams(self, input_fasta, input_vcf):
        '''
        purpose: breaks apart reference sequence into self.kmer_length kmers
        input: fasta_df, mutation_df if provided
        output: ngrams dataframe with kmers and associated gene and position on ref
        '''
        
        # if supervised, add mutants to fasta_df
        if input_vcf != "":
            fasta_df = pd.concat([input_fasta, input_vcf], axis=0)
        # otherwise run FTM without mutants
        else:
            fasta_df = input_fasta

        # create series of targets for faster comparison
        seq_series = pd.Series(fasta_df.seq.values, 
                               index=fasta_df.id.values,
                               dtype="object")
        
        # break apart each sequence into ngrams
        ngram_dict = {}
        for gene, seq in seq_series.iteritems():
            ngram_dict[gene] = cpy.ngrams(seq.strip(), n=self.kmer_size)
        
        # merge ngrams with gene id and starting pos of kmer    
        ngrams = pd.DataFrame(pd.concat({k: pd.Series(v) for k, v in ngram_dict.items()})).reset_index()
        ngrams.columns = ["gene", "pos", "ngram"]
        ngrams.reset_index()

        return ngrams
   
    @jit             
    def match_targets(self, ngrams):
        '''
        purpose: compares each input ref sequence with each basecall
        using ref_ngrams() from cython_funcs
        input: encoded targets, fasta_df from run_shortstack.py
        output: dataframe of matches to encoded_df
        '''
        # set indices on dataframes for faster merging
        ngrams.set_index("ngram", inplace=True)
        self.encoded_df.set_index("Target", inplace=True)
         
        # inner join between kmer seqs and targets to get matches
        matches = self.encoded_df.join(ngrams, how="inner")\
            .reset_index().rename(columns={'index' : 'target'})
        
        # rearrange dataframe for saving to file
        matches = pd.DataFrame(matches[['FeatureID', 'gene',
                                        'target', 'Category', 
                                        'PoolID', 'Qual', 
                                        'pos']])
        matches["hamming"] = 0
        
        # return featureID/target that did not perfectly match
        self.encoded_df.reset_index(inplace=True)
        non_matches = self.encoded_df[['FeatureID', 'Target']]\
            [~self.encoded_df[['FeatureID', 'Target']]\
             .isin(matches[['FeatureID', 'target']])]
        
        return matches, non_matches
     
    def diversity_filter(self, input_df):
        '''
        purpose: filters out targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only targets per feature that meet threshold
        '''
        
        # add list of unique targets for each gene
        input_df["target_list"] = input_df.groupby(["FeatureID", "gene"])["target"]\
             .transform("unique")
        
        # count number of unique basecalls matched for each FeatureID/gene
        input_df["feature_div"] = input_df.target_list.swifter.apply(lambda x: len(x))

        # filter out FeatureID/gene combos that are below feature_div threshold
        diversified = input_df[input_df.feature_div > self.diversity_threshold]
        diversified.reset_index(inplace=True, drop=True)
        
        # output diversity filtered targets for qc df
        not_diverse = input_df[input_df.feature_div <= self.diversity_threshold]

        return diversified, not_diverse

    @jit
    def locate_multiMapped(self, diversified):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: feature_div filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # flag targets that are multi-mapped
        diversified['multi_mapped'] = np.where(diversified.groupby(["FeatureID", "gene", "target"]).pos\
                                    .transform('nunique') > 1, "T", '')

        # separate multi and non multi mapped reads
        non = diversified[diversified["multi_mapped"] != "T"]\
            .drop("multi_mapped", axis=1)   
        multi = diversified[diversified["multi_mapped"] == "T"]\
            .drop("multi_mapped", axis=1)
        
        # non multi mapped reads get a count of 1 each
        non["count"] = 1
        
        return multi, non
    
    def normalize_multiMapped(self, multi):
        '''
        purpose: normalize counts for multi mapped reads
        input: multi dataframe built by ftm.locate_multiMapped() 
        output: multi dataframe with normalized counts 
        '''
        
        # add counts to multi-mapped reads with normalization 
        multi["count"] = multi.groupby(["FeatureID", "gene", "target"])["pos"]\
              .transform(lambda x: 1/x.nunique())
              
        return multi
    
    @jit
    def finalize_counts(self, multi, non):
        '''
        purpose: finalize counts for both non and multi-mapped reads
        input: non and multi dataframes built in ftm
        output: dataframe of counts per featureId/gene filtered and
            normalized for perfect matches
        '''
        
        # join the multi and non back together
        counts = pd.concat([multi, non], axis=0) 
        # round counts to two decimal places      
        counts["count"] = counts["count"].round(2)

        # groupby featureID and barcode 
        bc_counts = counts.copy(deep=True)
        bc_counts["target_count"] = counts.groupby(['FeatureID', 'target'])['count'].transform("sum")

        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "gene", "target"],
                                                   keep="first")
        

        bc_counts.drop(["count", "pos", "Category", "Qual", "PoolID", "target_list"], 
                       axis=1, inplace=True)
        
        # sort counts for nicer output
        bc_counts = bc_counts.sort_values(by=["FeatureID", 
                                        "gene", 
                                        "target"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        bc_counts.reset_index(inplace=True, drop=True)
        counts_outfile = "{}/perfectMatch_bc_counts.tsv".format(self.output_dir)
        # save basecall normalized counts to file
        bc_counts.to_csv(counts_outfile, sep="\t", 
                      index=False, 
                      header=True)
        
        # sum counts per gene
        counts = counts[["FeatureID", "gene", "hamming", "feature_div", "count", "target_list"]]
        counts["gene_count"] = counts.groupby(['FeatureID', 'gene'])['count'].transform("sum")
        
        # keep only relevant columns and drop duplicated featureID/gene combos
        count_df = counts.drop_duplicates(subset=["FeatureID", "gene"],
                                                   keep="first")
        
        count_df.drop("count", axis=1, inplace=True)
        
        # filter out featureID/gene combos below covg threshold
        count_df = count_df[count_df.gene_count.astype(int) >= self.coverage_threshold]
        
        return count_df
    
    def filter_counts(self, count_df):

        # create columns identifying top 2 gene_count scores
        count_df["max_count"] = count_df.groupby("FeatureID")["gene_count"].transform("max")
        count_df["second_max"] = count_df.groupby("FeatureID")["gene_count"].transform(lambda x: x.nlargest(2).min())
        
        # return all rows that match the max,2nd gene_count scores
        count_df = count_df[(count_df.gene_count == count_df.max_count) | (count_df.gene_count == count_df.second_max)]
        count_df.reset_index(drop=True, inplace=True)
        
        # return counts filtered out for coverage to add to qc df
        below_counts = count_df[count_df.gene_count.astype(int) < self.coverage_threshold]
        
        # filter counts table by coverage threshold
        counts = count_df[count_df.gene_count.astype(int) >= self.coverage_threshold]

        return counts, below_counts
    
    def find_multis(self, count_df):
        '''
        purpose: located features that have tied counts for best FTM call
        input: count_df created in ftm.finalize_counts.py
        output: dataframe of ties and non-tied counts_df for passing to return_ftm
        '''
        
        # separate ties to process
        counts = count_df.groupby('FeatureID').filter(lambda x: len(x)==1)
        multis = count_df.groupby('FeatureID').filter(lambda x: len(x)> 1)

        return counts, multis
    
    @jit
    def multi_breaker(self, x):
        '''
        purpose: uses cython_funcs.calc_symmetricDiff to find unique bc for ties
            with top 2 feature diversity scores, if one score is 2x or greater, keep
            else, calculate symmetric difference and keep score with most unique bc's
        input: multi_df
        output: used in return_ftm to return ties that pass this filtering logic
        '''
        
        # take top 2 highest sym_diff scores
        max_symDiff = x.sym_diff.max()
        
        # get top two count scores
        max_count = int(x.max_count.unique())
        second_max = int(x.second_max.unique())
        
        # get top two unique diversity scores
        max_div =  x.feature_div.nlargest(2).max()
        second_div = x.feature_div.nlargest(2).min()

        # if max > 3x coverage, return max
        if max_count >= (3* second_max):
            
            result = x[x.gene_count == max_count]
           
            # if more than one result meets criteria, look for diversity 2x >
            if len(result) > 1:
                
                # if one result has 2x > feature diversity, return result
                if max_div >= (2 * second_div):
                    result = x[x.feature_div == max_div]
                    return result
                
                # return result with max unique diversity
                else:
                    result = x[x.sym_diff == max_symDiff]

                    # if there is still a tie, no call can be made
                    if len(result) > 1:
                        pass
                    else:
                        return result
            # if no tie, return result
            else:
                return result
        
        # if no max > 3x covg, check for results with 2x unique feature diversity
        elif max_div >= (2 * second_div):
            
            result = x[x.feature_div == max_div]
            
            # if ties, take result with greatest unique feature diversity
            if len(result) > 1:
                result = x[x.sym_diff == max_symDiff]
                
                # if there is a still a tie, no call can be made
                if len(result) == 1:
                    return result
                else:
                    pass
            
            # return result if no tie here   
            else:
                return result       
        
        # if no 3x coverage, no 2x unique diversity, return result with highest unique feature div      
        else:
            result = x[x.sym_diff == max_symDiff]
            
            # if there is a tie here, no call can be made
            if len(result) == 1:
                return result
            else:
                pass 
        

    @jit
    def return_ftm(self, count_df, multi_df):
        '''
        purpose: process multis and return final ftm calls
        input: dataframe of top_two coverage and multis
        output: ftm calls
        '''
                
        # calculate targets unique to each region of interest
        multi_df["sym_diff"] = multi_df.groupby("FeatureID", group_keys=False).apply(cpy.calc_symmetricDiff)
        
        ### process ties ###
        broken_ties = multi_df.groupby('FeatureID').apply(self.multi_breaker)
        broken_ties = broken_ties.drop(["max_count", "second_max", "sym_diff"], axis=1)
        broken_ties.reset_index(drop=True, inplace=True)
        

        # format count_df for merging with broken_ties
        count_df = count_df.drop(['max_count', 'second_max'], axis=1)
        
        # if ties were broken add results to count_df
        if broken_ties.shape[0] > 0:
            
            broken_ties.reset_index(drop=True, inplace=True)
            ftm = pd.concat([count_df, broken_ties], ignore_index=True)
        
        # if no ties broken, just return count_df    
        else:
            ftm = count_df
            
        ftm.reset_index(drop=True, inplace=True)
        
        # format ftm df for output to file
        ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
        ftm.to_csv(ftm_calls, sep="\t", index=False) 
        return ftm  
    
    def ftm_qc(self, div_filtered, count_filtered):
        '''
        purpose: output information on which bc and targets were filtered
            during FTM
        input: diversity and count filtered dataframes built during FTM
        output: ftm_qc_filter.tsv in output directory
        '''
        
        # parse formatting
        div_filtered["filter"] = "diversity"
        count_filtered["filter"] = "counts"
        count_filtered["target"] = "any"
        
        div_filtered.drop([ 'Category', 'PoolID', 'Qual', 'pos', 'target_list'], 
                          axis=1,
                          inplace=True)
        count_filtered.drop(["gene_count", "target_list"], 
                            axis=1,
                            inplace=True)
        ftm_qc = pd.concat([div_filtered, count_filtered], 
                           ignore_index=True,
                           sort=False)
        
        
        ftm_qc.to_csv(self.ftmQC_filter_file, sep="\t", index=False)
    
    def main(self):
        
        # create raw counts file
        print("Counting reads...\n")
        self.target_count()
        
        # break reference seq into kmers
        print("Breaking up reference sequence into kmers...\n")
        ngrams = self.create_ngrams(self.fasta_df, self.mutant_fasta)        
        
        # match targets with basecalls
        print("Finding perfectly matched kmers...\n")
        perfects, non_matches = self.match_targets(ngrams)

        # filter for feature feature_div
        print("Filtering results for feature diversity...\n")
        diversified_hamming, diversity_filtered = self.diversity_filter(perfects)
        
        # locate multi-mappped reads
        print("Normalizing counts...\n")
        multi, non = self.locate_multiMapped(diversified_hamming)
        
        # normalize multi-mapped reads
        multi = self.normalize_multiMapped(multi)

        # finalize normalized counts
        print("Finalizing counts....\n")
        norm_counts = self.finalize_counts(multi, non)
        
        # filter counts for 3x coverage
        norm_counts, below_cnts = self.filter_counts(norm_counts)

        # find ties to process
        print("Processing target coverage...\n")
        count_df, multi_df = self.find_multis(norm_counts)
        
        # return ftm calls
        print("Generating FTM calls...\n")
        ftm = self.return_ftm(count_df, multi_df)
        
        # save info on reads filtered out for diversity or count threshold
        self.ftm_qc(diversity_filtered, below_cnts)

        # return ftm matches and feature_div filtered non-perfects
        return ngrams, non_matches, norm_counts, ftm
        

        