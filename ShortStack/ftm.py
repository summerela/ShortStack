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
from collections import Counter
from pathlib import Path

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, coverage_threshold, 
                 kmer_size, max_hamming_dist,
                 output_dir, diversity_threshold,
                 num_cores, hamming_weight):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.coverage_threshold = coverage_threshold
        self.max_hamming_dist = max_hamming_dist
        self.kmer_size = kmer_size
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.raw_count_file = Path("{}/rawCounts.tsv".format(self.output_dir))
        self.num_cores = num_cores
        self.hamming_weight = hamming_weight
    
    def create_ngrams(self, input_fasta, input_vcf):
        '''
        purpose: breaks apart reference sequence into self.kmer_length kmers
        input: fasta_df, mutation_df if provided
        output: ngrams dataframe with kmers and associated gene and position on ref
            both to region and to each variant/wt
        '''
        
        # if supervised, add mutants to fasta_df
        if not input_vcf.empty:
            fasta_df = pd.concat([input_fasta, input_vcf], axis=0, sort=True)
        # otherwise run FTM without mutants
        else:
            fasta_df = input_fasta

        # create series of Targets for faster comparison
        seq_series = pd.Series( fasta_df.seq.values, 
                               index=fasta_df.id.values,
                               dtype="object")

        # break apart each sequence into ngrams
        ngram_dict = {}
        for gene, seq in seq_series.iteritems():
            ngram_dict[gene] = cpy.ngrams(seq.strip(), n=self.kmer_size)
        
        # merge ngrams with gene id and starting pos of kmer    
        ngrams = pd.DataFrame(pd.concat({k: pd.Series(v) for k, v in ngram_dict.items()})).reset_index()
        ngrams.columns = ["gene", "pos", "ngram"]
        
        # save all ngrams to file for easy mapping later in case we need it
        ngrams_out = Path("{}/all_ngrams.tsv".format(self.output_dir))
        ngrams.to_csv(ngrams_out, sep="\t", index=False)

        # get group information back from fasta_df
        ngrams = ngrams.merge(fasta_df[["id", "groupID"]], 
                             left_on="gene", 
                             right_on="id")
        
        ngrams_unique = ngrams.drop("id", axis=1)

        # combine unique kmers for region into group_ngrams
        group_ngrams = ngrams.groupby(['groupID', 'pos'])["ngram"].unique().reset_index()

        return ngrams_unique, group_ngrams
    
    def reshape_ngrams(self, df):
        '''
        purpose: flatten list of ngrams for each reference into dataframe
        input: group_ngrams list from ftm.create_ngrams()
        output: ngrams dataframe to region
        '''
        
        # specify column containing list
        list_col = 'ngram'

        # flatten listed column and repeate related info from other columns
        ngrams = pd.DataFrame({
              col:np.repeat(df[col].values, df[list_col].str.len())
              for col in df.columns.drop(list_col)}
            ).assign(**{list_col:np.concatenate(df[list_col].values)})[df.columns]
  
        return ngrams
       
    def match_Targets(self, ngram_df):
        '''
        purpose: match ngrams to targets 
        input: ngram_df craeted in ftm.parse_ngrams()
        output: hamming_list of tuples (ref, target match, hamming distance)
        '''

        # get unique set from each list
        ngram_list = list(set(ngram_df.ngram.values))
        Target_list = list(set(self.encoded_df.Target))
         
        hd = lambda x, y: cpy.calc_hamming(x,y, self.max_hamming_dist)
        hamming_list = [hd(x, y) for y in Target_list for x in ngram_list]
   
        # check that calls were found
        if len(hamming_list) < 1:
            raise SystemExit("No calls below hamming distance threshold.")
        else:
            return hamming_list
            
    
    @jit
    def parse_hamming(self, hamming_list,group_ngram, variant_ngrams):
        '''
        purpose: combine hamming distance information with feature level data
        input: hamming_list: created in match_targets(), list of all ngrams, basecalls and hd
            ngram_df: specifies which gene and position a target came from
            encoded_df: provided feature level information
        output: hamming_df with featureID, gene, group, target, pos
        '''

        
        # create dataframe from list of matches and subset to hamming dist threshold
        hamming_df = pd.DataFrame(hamming_list, columns=["ref_match", "bc_feature", "hamming"])
        hamming_df = hamming_df[hamming_df.hamming != "X"]
        assert not hamming_df.empty, "No matches were found below hamming distance threshold."

        # match ngrams to their matching Target regions
        hamming_df = hamming_df.merge(group_ngram, left_on="ref_match", 
                                      right_on="ngram").reset_index(drop=True)
        
        # match basecalls to their matching features
        hamming_df = hamming_df.merge(self.encoded_df, left_on="bc_feature", 
                                      right_on="Target").reset_index(drop=True)
        
        # sort values for nicer output and drop duplicates
        hamming_df = hamming_df.sort_values(["FeatureID", "groupID", "Target", "pos"])
        hamming_df.drop_duplicates(inplace=True)

        # save raw hamming counts to file
        hamming_df.to_csv(self.raw_count_file , sep="\t", index=None)

        # reorder columns and subset
        hamming_df = hamming_df[["FeatureID", "Target", "groupID", "pos", "ref_match", "hamming"]]
        
        # separate perfect matches and hd1+
        hd_plus = hamming_df[hamming_df.hamming > 0].reset_index(drop=True)
        perfects = hamming_df[hamming_df.hamming == 0].reset_index(drop=True)
        
        return hd_plus, perfects

    @jit
    def diversity_filter(self, input_df):
        '''
        purpose: filters out Targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only Targets per feature that meet threshold
        '''
        
        # filter out FeatureID/gene combos that are below feature_div threshold
        input_df["feature_div"] = input_df.groupby(["FeatureID", "groupID"])["Target"].transform("nunique") 
           
        diversified = input_df[input_df.feature_div > self.diversity_threshold]
        diversified.reset_index(inplace=True, drop=True)
        
        # check that calls were found
        if not diversified.empty:
            return diversified
        else:
            raise SystemExit("No calls pass diversity threshold.")
    
    @jit
    def locate_multiMapped(self, diversified):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: feature_div filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # flag Targets that are multi-mapped
        diversified['multi_mapped'] = diversified.groupby(["FeatureID", "groupID", "Target"]).pos\
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
    def sum_counts(self, counts):
        '''
        purpose: sum counts for both non and multi-mapped reads
        input: non and multi dataframes built in ftm
        output: dataframe of counts per featureId/gene filtered and
            normalized for perfect matches
        '''
        
         # round counts to two decimal places      
        counts["counts"] = counts["counts"].astype(float).round(2)

        # count barcodes for each target per feature 
        bc_counts = counts.copy(deep=True)
        
        # sum counts per feature/region/pos
        bc_counts["bc_count"] = counts.groupby(['FeatureID', 'Target', 'groupID', 'pos'])\
            ["counts"].transform("sum")
        
        # information will be duplicated for each row, drop dups   
        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "groupID", "Target", "pos"],
                                                    keep="first")
        
        # drop old counts per basecall to keep count per position
        bc_counts.drop(["counts"],axis=1, inplace=True)
        
        # sort counts for nicer output
        bc_counts = bc_counts.sort_values(by=["FeatureID", 
                                        "groupID", 
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
        perfects = bc_counts.copy(deep=True)
        perfects = perfects.drop(["ref_match"], axis=1)
        perfects["counts"] = perfects.groupby(['FeatureID', 'groupID'])['bc_count'].transform("sum")
        
        # save new subset copy of perfects for downstream analysis
        bc_counts2 = perfects.copy(deep=True)
        bc_counts2.drop(["counts"], axis=1, inplace=True)
        
        perfects = perfects[["FeatureID", "groupID", "feature_div", "counts", "hamming"]]
        
        # keep only relevant columns and drop duplicated featureID/gene combos
        perfects = perfects.drop_duplicates(subset=["FeatureID", "groupID"],
                                                   keep="first").reset_index(drop=True)

        # filter out featureID/gene combos below covg threshold
        perfects = perfects[perfects["counts"].astype(int) >= self.coverage_threshold]
        perfects.reset_index(drop=True, inplace=True) 
        assert len(perfects) != 0, "No matches found above coverage threshold." 
        
        return bc_counts2, perfects

    def get_top2(self, perfects):
        '''
        purpose: locate top 2 target counts for perfect matches
        input: perfects built in ftm.sum_counts()
        output: hamming_df= calls within hamming threshold
            perfect matches with columns denoting max and second max counts
        '''

        # create columns identifying top 2 gene_count scores for perfects
        perfects["max_count"] = perfects.groupby("FeatureID")["counts"].transform("max")
        perfects["second_max"] = perfects.groupby("FeatureID")["counts"].transform(lambda x: x.nlargest(2).min())
        
        return perfects
    
    @jit
    def find_tops(self, perfects):
        '''
        purpose: retain only top two targets with highest counts
        input: perfects of perfect matches
        output: perfects subset to top two matches 
        '''
        
        # return all rows that match the max,2nd gene_count scores
        perfects = perfects[(perfects.counts == perfects.max_count) | (perfects.counts == perfects.second_max)]
        perfects.reset_index(drop=True, inplace=True)

        return perfects
    
    def find_ties(self, perfects):
        '''
        purpose: located features that have tied counts for best FTM call
        input: perfects created in ftm.sum_counts.py
        output: dataframe of ties and non-tied counts for passing to return_ftm
        '''
        
        # pull out features that have ties for first or second place
        no_ties = perfects.groupby('FeatureID').filter(lambda x: len(x)==1)
        multis = perfects.groupby('FeatureID').filter(lambda x: len(x)> 1)

        # filter multis for (max_count - second_max = 0)
        multis = perfects.loc[perfects.max_count - multis.second_max > 0]

        return no_ties, multis
    
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

    def sym_diff(self, sets):
        '''
        purpose: find reads that are unique to a target/feature
        input: list of kmers from targets and input references
        output:  tuples of symmetrical differences for each target/feature pairing
        
        '''
        
        # initialize counter object
        count = Counter()
        
        # compare sets of kmers for each potential target per feature
        for s in sets:
            # update counter
            count.update(s)
        # return number of unique reads found in each comparison
        return [sum(count[x] == 1 for x in s) for s in sets]
        
#     @jit
    def return_ftm(self, bc_counts, no_ties, multi_df):
        '''
        purpose: process multis, break ties and return final ftm calls
        input: dataframe of basecall counts, counts and multis
        output: ftm calls
        '''

        # if there are multis, process them
        if not multi_df.empty:
            
            # subset to relevant rows and features for speed 
            target_df = bc_counts[["FeatureID", "groupID", "Target"]][bc_counts.FeatureID.isin(multi_df.FeatureID)]        

            # add list of unique targets for each gene
            target_df["target_list"] = target_df.groupby(["FeatureID", "groupID"])["Target"]\
                 .transform("unique")
            target_df= target_df.drop_duplicates(subset=["FeatureID", "groupID"],
                                                        keep="first")
            
            # convert target list to sets for quick comparison
            target_df["target_list"] = target_df["target_list"].apply(set)
            
            # add target lists to multi_df
            multi_df = multi_df.merge(target_df, on=["FeatureID", "groupID"], how="left")

            # calculate Targets unique to each region of interest
            s = multi_df.groupby("FeatureID").apply(self.sym_diff).reset_index()
            s.columns = ["FeatureID", "sym_diff"]
            
            # add symmetrical difference to multi_df
            sym_df = pd.concat([pd.DataFrame(v, columns=["sym_diff"])
                 for k,v in s.sym_diff.to_dict().items()])
            sym_df.reset_index(drop=True, inplace=True)
            
            multi_df["sym_diff"] = sym_df.sym_diff
            
        
            ### process ties ###
            broken_ties = multi_df.groupby('FeatureID').apply(self.multi_breaker) 
            
            # drop no_ties extra columns before merging
            no_ties = no_ties.drop(["max_count", "second_max"], axis=1)
            
            # if ties were broken add results to no_ties
            if not broken_ties.empty:
                
                broken_ties = broken_ties.drop(["max_count", "second_max", "Target",
                                            "sym_diff", "target_list"], axis=1)
                broken_ties.reset_index(drop=True, inplace=True)
                
                # combine with any no-ties
                if not no_ties.empty:
                    ftm = pd.concat([no_ties, broken_ties], ignore_index=True, sort=False)
                else:
                    ftm = broken_ties
            
            # if no ties broken, just return no_ties    
            else:
                ftm = no_ties
        
        # if no multis, return no_ties table
        else:
            ftm = no_ties
            if not ftm.empty:
                ftm = ftm.drop(["max_count", "second_max"], axis=1)

        if not ftm.empty:

            # format df
            ftm.drop(["feature_div", "hamming"], axis=1, inplace=True) 
            ftm["counts"] = ftm.counts.astype(float).round(2)
            
            # output ftm to file
            ftm_calls = Path("{}/ftm_calls.tsv".format(self.output_dir))
            ftm.to_csv(ftm_calls, sep="\t", index=False) 
    
            return ftm
        else: 
            raise SystemExit("No FTM calls can be made on this dataset.")
    
    @jit
    def return_calls(self, ftm_df, bc_counts):
        '''
        purpose: save features where no ftm call can be made to a file
            subset bc_counts to contain only features where an FTM call can be made
        input: ftm_df and encoded_df
        output: no_calls = featureID's and basecalls for features that were not able to 
            be assigned an FTM call to be run through hamming.py
            perfect_calls = bc_counts of per feature/target counts for ftm called features
        '''
        
        # get counts for features with no FTM call 
        no_calls = bc_counts[~bc_counts.FeatureID.isin(ftm_df.FeatureID)]\
            .dropna().reset_index(drop=True)
                           
        if not no_calls.empty:
           
            # save no_call info to file
            no_ftm = no_calls[["FeatureID"]].drop_duplicates()
            no_ftm_file = Path("{}/no_ftm_calls.tsv".format(self.output_dir))
            no_ftm.to_csv(no_ftm_file, index=False, sep="\t")
            
        # pull out only calls related to FTM called region for HD under threshold   
        calls = ftm_df.merge(bc_counts, on=["FeatureID", "groupID"])
        calls = calls.drop(["counts", "feature_div"], axis=1)

        return calls
    
    def return_all_calls(self, hamming_df, ftm_df, perfect_calls):
        '''
        purpose: locate hd1+ barcodes that match ftm calls and combine with 
            filtered perfects
        input: hamming_df containing all hd1+ matches
            ftm_df: ftm calls 
        output: all_counts: dataframe containing all HD0 and HD1+ barcodes 
            for each ftm call
        '''
        # get counts for HD1+ FTM call 
        ftm_df = ftm_df.drop(["counts"], axis=1)
        hd1_plus = pd.merge(hamming_df, ftm_df, on=['FeatureID', 'groupID'], how='inner')
        
        # locate multi-mapped hd1+ barcodes and add counts
        normalized = self.locate_multiMapped(hd1_plus)
        
         # round counts to two decimal places      
        normalized["counts"] = normalized["counts"].astype(float).round(2)
        
        # sum counts per feature/region/pos
        normalized["bc_count"] = normalized.groupby(['FeatureID', 'Target', 'groupID', 'pos'])\
            ["counts"].transform("sum")
        
        # information will be duplicated for each row, drop dups   
        hd_counts = normalized.drop_duplicates(subset=["FeatureID", "groupID", "Target", "pos"],
                                                    keep="first")
        
        # drop old counts per basecall to keep count per position
        hd_counts.drop(["counts", "ref_match"],axis=1, inplace=True)
        
        # order dataframes
        perfect_calls = perfect_calls[["FeatureID", "groupID", "Target", "pos", "hamming", "bc_count"]]
        hd_counts = hd_counts[["FeatureID", "groupID", "Target", "pos", "hamming", "bc_count"]]

        # concatenate dataframe
        all_counts = pd.concat([perfect_calls, hd_counts])
        
        all_counts = all_counts.sort_values(by=["FeatureID", 
                                        "pos",
                                        "groupID", 
                                        "Target"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        all_counts.reset_index(inplace=True, drop=True)
        
        counts_file = Path("{}/all_ftm_counts.tsv".format(self.output_dir))
        all_counts.to_csv(counts_file, sep="\t", index=False)
        
        return all_counts
        

    
    def main(self):
        
        # break reference seq into kmers
        print("Breaking up reference sequence into kmers...\n")
        ngrams_unique, ngrams_group = self.create_ngrams(self.fasta_df, self.mutant_fasta)   
        
        # reshape ngrams into dataframe for matching
        ngrams = self.reshape_ngrams(ngrams_group)     
        
        # match Targets with basecalls
        print("Locating basecalls within hamming distance threshold...\n")
        hamming_list = self.match_Targets(ngrams)
        
        # parse hamming df and return calls beyond hamming threshold for qc
        print("Parsing the hamming results...\n")
        hd_plus, perfects = self.parse_hamming(hamming_list, ngrams, ngrams_unique)
      
        # filter for feature feature_div
        print("Filtering results for feature diversity...\n")
        diversified = self.diversity_filter(perfects)

        # locate multi-mappped reads
        print("Normalizing counts...\n")
        counts = self.locate_multiMapped(diversified)

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
        perfects, multi_df = self.find_ties(top_df)
        
        # return ftm calls
        print("Generating FTM calls...\n")
        ftm = self.return_ftm(bc_counts, perfects, multi_df)
        
        # return no_calls and df of read for just the ftm called region
        print("Recording features with no FTM call...\n")
        perfect_calls = self.return_calls(ftm, bc_counts)
        
        # filter hd1+ to only ftm called features and combine with perfects
        print("Parsing output for sequencing...\n")
        all_counts = self.return_all_calls(hd_plus, ftm, perfect_calls)

        # return ftm matches and feature_div filtered non-perfects
        return all_counts
        

        