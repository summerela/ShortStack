'''
ftm.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches and hamming distance = self.max_ham_dist matches
- creates voting table and normalized counts 
- returns FTM calls
'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, coverage_threshold, 
                 kmer_size, max_hamming_dist,
                 output_dir, diversity_threshold,
                 qc_outfile, run_info_file, 
                 num_cores, hamming_weight):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.coverage_threshold = coverage_threshold
        self.max_hamming_dist = max_hamming_dist
        self.kmer_size = kmer_size
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.qc_outfile = qc_outfile
        self.raw_count_file = "{}/rawCounts.tsv".format(self.output_dir)
        self.run_info_file = run_info_file
        self.num_cores = num_cores
        self.hamming_weight = hamming_weight
    
    @jit    
    def Target_count(self):
        '''
        purpose: count number of Targets found by position within each feature
        input: encoded_df of feature ID and encoded color call
        output: raw_counts.tsv file of raw counts
        '''
        
        # sort encoded dataframe by featureID
        self.encoded_df.sort_values("FeatureID", inplace=True)
        # count number of Targets found in each barcode
        feature_counts = self.encoded_df.groupby(["FeatureID", "Target"]).size().reset_index()
        # update columnnames        
        feature_counts.columns = ["FeatureID", "Target", "counts"]
    
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

        # create series of Targets for faster comparison
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

        return ngrams
       
    def match_Targets(self, ngram_df):
        
        # get unique set from each list
        ngram_list = list(set(ngram_df.ngram))
        Target_list = list(set(self.encoded_df.Target))
         
        hd = lambda x, y: cpy.calc_hamming(x,y, self.max_hamming_dist)
        hamming_list = [hd(x, y) for y in Target_list for x in ngram_list]
        
        # check that calls were found
        if len(hamming_list) > 0:
            return hamming_list
        else:
            raise SystemExit("No calls below hamming distance threshold.")
    
    @jit
    def parse_hamming(self, hamming_list,ngram_df):
        
        # create dataframe from list of matches and subset to hamming dist threshold
        hamming_df = pd.DataFrame(hamming_list, columns=["ref_match", "bc_feature", "hamming"])
        hamming_df = hamming_df[hamming_df.hamming != "X"]
        
        # match ngrams to their matching Target regions
        hamming_df = hamming_df.merge(ngram_df, left_on="ref_match", right_on="ngram").reset_index(drop=True)
        
        # match basecalls to their matching features
        hamming_df = hamming_df.merge(self.encoded_df, left_on="bc_feature", right_on="Target").reset_index(drop=True)
        
        # sort values for nicer output
        hamming_df = hamming_df.sort_values(["FeatureID", "Target", "pos"])

        # save raw hamming counts to file
        hamming_df.to_csv(self.raw_count_file , sep="\t", index=None)

        # reorder columns and subset
        hamming_df = hamming_df[["FeatureID", "Target", "gene", "pos", "ref_match", "hamming"]]

        return hamming_df 

    @jit
    def diversity_filter(self, input_df):
        '''
        purpose: filters out Targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only Targets per feature that meet threshold
        '''
        
        # filter out FeatureID/gene combos that are below feature_div threshold
        input_df["feature_div"] = input_df.groupby(["FeatureID", "gene"])["Target"].transform("nunique")    
        diversified = input_df[input_df.feature_div > self.diversity_threshold]
        diversified.reset_index(inplace=True, drop=True)
        
        # check that calls were found
        if not diversified.empty:
            return diversified
        else:
            raise SystemExit("No calls above diversity threshold.")
    
    @jit
    def locate_multiMapped(self, diversified):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: feature_div filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # flag Targets that are multi-mapped
        diversified['multi_mapped'] = diversified.groupby(["FeatureID", "gene", "Target"]).pos\
                                    .transform('nunique') 

        # separate multi and non multi mapped reads
        non = diversified[diversified["multi_mapped"] == 1]
        multi = diversified[diversified["multi_mapped"] > 1]\
        
        # non multi mapped reads get a count of 1 each
        if not non.empty:
            non["counts"] = 1
            counts = non
        
        # noramlize multi-mapped reads
        if not multi.empty:
            multi["counts"] = 1/multi.multi_mapped
            counts = pd.concat([counts, multi], axis=0).drop("multi_mapped", axis=1).reset_index(drop=True)
        
        return counts

    @jit    
    def weight_hams(self, counts):
        '''
        purpose: add weight penalty to basecalls with hamDist > 0
            weighted_count = (ham_weight * (kmer_len - ham_dist)/kmer_len) * unweighted_count
        input: counts table with hamming distance column
        output: counts table with hd1+ weighted by HD and length
            note: this table will eventually be passed to graph functions
        '''
   
        # 
        counts["counts"][counts.hamming > 0] = (self.hamming_weight * \
            ((self.kmer_size - counts.hamming)/self.kmer_size))\
                * counts.counts

        # round counts to two decimal places      
        counts["counts"] = counts["counts"].astype(float).round(2)
        return counts
    
    @jit
    def sum_counts(self, counts):
        '''
        purpose: sum counts for both non and multi-mapped reads
        input: non and multi dataframes built in ftm
        output: dataframe of counts per featureId/gene filtered and
            normalized for perfect matches
        '''

        # count barcodes for each target per feature 
        bc_counts = counts.copy(deep=True)
        
        # sum counts per feature/gene/pos
        bc_counts["bc_count"] = counts.groupby(['FeatureID', 'Target', 'gene', 'pos'])\
            ["counts"].transform("sum")
        
        # information will be duplicated for each row, drop dups   
        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "gene", "Target", "pos"],
                                                    keep="first")
        
        # drop old counts per basecall to keep count per position
        bc_counts.drop(["counts"],axis=1, inplace=True)
        
        # sort counts for nicer output
        bc_counts = bc_counts.sort_values(by=["FeatureID", 
                                        "gene", 
                                        "Target",
                                        "pos"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        bc_counts.reset_index(inplace=True, drop=True)
        counts_outfile = "{}/bc_counts.tsv".format(self.output_dir)
        
        
        # save basecall normalized counts to file
        bc_counts.to_csv(counts_outfile, sep="\t", 
                      index=False, 
                      header=True)
        
        # sum counts per gene
        count_df = bc_counts.copy(deep=True)
        count_df["counts"] = count_df.groupby(['FeatureID', 'gene'])['bc_count'].transform("sum")
        count_df = count_df[["FeatureID", "gene", "feature_div", "counts", "hamming"]]
        
        # keep only relevant columns and drop duplicated featureID/gene combos
        count_df = count_df.drop_duplicates(subset=["FeatureID", "gene"],
                                                   keep="first").reset_index(drop=True)

        # filter out featureID/gene combos below covg threshold
        count_df = count_df[count_df["counts"].astype(int) >= self.coverage_threshold]
        count_df.reset_index(drop=True, inplace=True) 
        
        return bc_counts, count_df

    def get_top2(self, count_df):
        '''
        purpose: locate top 2 target counts for perfect matches
        input: count_df built in ftm.sum_counts()
        output: hamming_df= calls within hamming threshold
            perfect matches with columns denoting max and second max counts
        '''
        # separate perfect matches and hd1+
        count_df = count_df[count_df.hamming == 0]

        # create columns identifying top 2 gene_count scores
        count_df["max_count"] = count_df.groupby("FeatureID")["counts"].transform("max")
        count_df["second_max"] = count_df.groupby("FeatureID")["counts"].transform(lambda x: x.nlargest(2).min())
        return count_df
    
    @jit
    def find_tops(self, count_df):
        '''
        purpose: retain only top two targets with highest counts
        input: count_df of perfect matches
        output: count_df subset to top two matches 
        '''
        
        # return all rows that match the max,2nd gene_count scores
        count_df = count_df[(count_df.counts == count_df.max_count) | (count_df.counts == count_df.second_max)]
        count_df.reset_index(drop=True, inplace=True)
        
        return count_df
    
    def find_multis(self, count_df):
        '''
        purpose: located features that have tied counts for best FTM call
        input: count_df created in ftm.sum_counts.py
        output: dataframe of ties and non-tied counts for passing to return_ftm
        '''
        
        # separate ties to process
        counts = count_df.groupby('FeatureID').filter(lambda x: len(x)==1)
        multis = count_df.groupby('FeatureID').filter(lambda x: len(x)> 1)

        # filter multis for (max_count - second_max = 0)
        if not multis.empty:
            multis = multis.loc[multis.max_count - multis.second_max > 0]

        return counts, multis
    
    @jit
    def multi_breaker(self, x):
        '''
        purpose: uses cython_funcs.calc_symmetricDiff to find unique bc for ties
            with top 2 feature diversity scores, if one score is 2x or greater, keep
            else, calculate symmetric difference and keep score with most unique bc's
        input: multi_df
            x = feature
        output: used in return_ftm to return ties that pass this filtering logic
        '''
        
        # take top sym_diff score for group
        max_symDiff = x.sym_diff.max()
        
        # in case of multiple rows, get single value for max and second max
        max_count = int(x.max_count.unique())
        second_max = int(x.second_max.unique())
        
        # get top two feature diversity scores
        max_div =  x.feature_div.nlargest(2).max()
        second_div = x.feature_div.nlargest(2).min()

        # if max > 3x coverage, return max
        if max_count >= (3 * second_max):
            
            result = x[x.counts == max_count]

            # if more than one row meets criteria, look for diversity 2x >
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
    def return_ftm(self, bc_counts, count_df, multi_df):
        '''
        purpose: process multis, break ties and return final ftm calls
        input: dataframe of basecall counts, counts and multis
        output: ftm calls
        '''
        
        # if there are ties, process them
        if not multi_df.empty:
            
            # add target list to multi_df 
            target_df = bc_counts[["FeatureID", "gene", "Target"]]        
            
            # add list of unique targets for each gene
            target_df["target_list"] = target_df.groupby(["FeatureID", "gene"])["Target"]\
                 .transform("unique")
            target_df= target_df.drop_duplicates(subset=["FeatureID", "gene"],
                                                        keep="first")
            
            
            # add target lists to multi_df
            multi_df = multi_df.merge(target_df, on=["FeatureID", "gene"], how="left")
            
            # calculate Targets unique to each region of interest
            multi_df["sym_diff"] = multi_df.groupby("FeatureID", group_keys=False).apply(cpy.calc_symmetricDiff)
            
            ### process ties ###
            broken_ties = multi_df.groupby('FeatureID').apply(self.multi_breaker)        
            broken_ties = broken_ties.drop(["max_count", "second_max", "sym_diff"], axis=1)
            broken_ties.reset_index(drop=True, inplace=True)
            
            # if ties were broken add results to count_df
            if broken_ties.shape[0] > 0:
                broken_ties.drop(["Target", "target_list"], axis=1, inplace=True)
                broken_ties.reset_index(drop=True, inplace=True)
                ftm = pd.concat([count_df, broken_ties], ignore_index=True, sort=False)
            
            # if no ties broken, just return count_df    
            else:
                ftm = count_df
        
        # if not ties, return count_df table
        else:
            ftm = count_df
            
        # format df
        ftm.drop(["feature_div", "max_count", "second_max", "hamming"], axis=1, inplace=True) 
        ftm["counts"] = ftm.counts.astype(float).round(2)
        
        # output ftm to file
        ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
        ftm.to_csv(ftm_calls, sep="\t", index=False) 
        
        return ftm
    
    @jit
    def return_calls(self, ftm_df, bc_counts):
        '''
        purpose: return features and barcodes where no ftm call can be made
        input: ftm_df and encoded_df
        output: no_calls = featureID's and basecalls for features that were not able to 
            be assigned an FTM call to be run through hamming.py
        '''
        
        # get counts for features with no FTM call to pass to de bruijn graph
        no_calls = bc_counts[~bc_counts.FeatureID.isin(ftm_df.FeatureID)]\
            .dropna().reset_index(drop=True)
                           
        if not no_calls.empty:
           
            # save no_call info to file
            no_ftm = no_calls[["FeatureID"]].drop_duplicates()
            no_ftm_file = "{}/no_ftm_calls.tsv".format(self.output_dir)
            no_ftm.to_csv(no_ftm_file, index=False, sep="\t")
            
        # pull out only calls related to FTM called region for HD under threshold   
        calls = ftm_df.merge(bc_counts, on=["FeatureID", "gene"])
        calls = calls.drop(["counts", "ref_match", "hamming", "feature_div"], axis=1)
        
        return no_calls, calls

    
    def main(self):
        
        # create raw counts file
        print("Counting reads...\n")
        self.Target_count()
        
        # break reference seq into kmers
        print("Breaking up reference sequence into kmers...\n")
        ngrams = self.create_ngrams(self.fasta_df, self.mutant_fasta)        
        
        # match Targets with basecalls
        print("Locating basecalls within hamming distance threshold...\n")
        hamming_list = self.match_Targets(ngrams)
        
        # parse hamming df and return calls beyond hamming threshold for qc
        print("Parsing the hamming results...\n")
        hamming_df = self.parse_hamming(hamming_list, ngrams)
      
        # filter for feature feature_div
        print("Filtering results for feature diversity...\n")
        diversified = self.diversity_filter(hamming_df)

        # locate multi-mappped reads
        print("Normalizing counts...\n")
        counts = self.locate_multiMapped(diversified)
        
        # weight counts with hamming dist > 0
        counts = self.weight_hams(counts)

        # sum normalized counts
        print("Summing counts....\n")
        bc_counts, norm_counts = self.sum_counts(counts)

        # pull out top 2 counts for perfects, separating hd1+
        print("Finding max counts...\n")
        top2 = self.get_top2(norm_counts)  
      
        # filter top2 genes by coverage threshold
        top_df = self.find_tops(top2)

        # find ties to process
        print("Processing Target coverage...\n")
        count_df, multi_df = self.find_multis(top_df)
        
        # return ftm calls
        print("Generating FTM calls...\n")
        ftm = self.return_ftm(bc_counts, count_df, multi_df)
        
        # return no_calls and df of read for just the ftm called region
        no_calls, calls = self.return_calls(ftm, bc_counts)

        # return ftm matches and feature_div filtered non-perfects
        return calls, no_calls
        

        