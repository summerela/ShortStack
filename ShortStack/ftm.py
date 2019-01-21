'''
ftm.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches and hamming distance = self.max_ham_dist matches
- creates voting table and normalized counts 
- returns FTM calls
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import dask
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
from collections import defaultdict
from itertools import chain     
import psutil 
from dask.array.random import normal

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, coverage_threshold, 
                 max_hamming_dist,
                 output_dir, diversity_threshold,
                 hamming_weight, ftm_HD0_only, cpus, client):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.coverage_threshold = coverage_threshold
        self.max_hamming_dist = max_hamming_dist
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.raw_count_file = Path("{}/rawCounts.tsv".format(self.output_dir))
        self.hamming_weight = hamming_weight
        self.kmer_lengths = [int(x) for x in list(set(self.encoded_df.bc_length.values))]
        self.cpus = cpus
        self.ftm_HD0_only = ftm_HD0_only
        self.client = client
        
    @jit
    def create_fastaDF(self, input_fasta, input_vcf):

        # if supervised, add mutants to fasta_df
        if not input_vcf.empty:
            fasta_df = pd.concat([input_fasta, input_vcf], axis=0, sort=True)
        # otherwise run FTM without mutants
        else:
            fasta_df = input_fasta
            
        fasta_df = fasta_df.reset_index(drop=True)
        
        return fasta_df
    
    @jit
    def create_ngrams(self, row):
        '''
        purpose: breaks apart reference sequence into self.kmer_length kmers
        input: fasta_df, mutation_df if provided
        output: ngrams list to pass to parse_ngrams
        '''
        
        ngram_list =[]
        for size in self.kmer_lengths:
            # break apart reads into edges
            ngram = cpy.ngrams(row.seq.strip(), size)
            ngram_list.append(ngram)
            
        return ngram_list
    
    @jit
    def parse_ngrams(self, fasta_df):
        
        # unravel list to dataframe
        ngram_df = fasta_df.set_index(["id", "region", "chrom", "start"])['ngrams'].apply(pd.Series).stack()
        ngram_df = ngram_df.reset_index()
        ngram_df = ngram_df.drop("level_4", axis=1)
        ngram_df.columns = ['id', 'region', 'chrom', 'start', 'ngrams']
        
        return ngram_df

    def split_ngrams(self, row):  
        
         # match edge grams with position
        ngram_list = [(row.id, 
                row.region,
                row.chrom,
                str(int(row.start) + i), 
                c) for i, c in enumerate(row.ngrams)]

        return ngram_list
    
    def final_ngrams(self, ngram_df):
        
        # break apart dataframe into one row per tuple
        final_df = pd.DataFrame(dd.from_pandas(ngram_df.apply(lambda x: \
                pd.Series(x[0]),axis=1).stack().reset_index(level=1, drop=True),
                npartitions=self.cpus).compute())
        final_df.columns = ["tups"]
                             
        # break apart rows of tuples to final df
        final_df[['id', 'region', 'chrom', 'pos', 'ngram']] = final_df['tups'].apply(pd.Series)
        final_df = final_df.drop("tups", axis=1).reset_index(drop=True)
        
        # group together by region and drop duplicates to avoid double counting
        final_df = final_df.drop_duplicates(subset=["chrom", "pos", "ngram"]).reset_index(drop=True)

        return final_df
    
    def calc_hamDist(self, ngram_df, encoded_df):
        '''
        purpose: match ngrams to targets 
        input: ngram_df craeted in ftm.parse_ngrams()
        output: hamming_list of tuples (ref, target match, hamming distance)
        '''

        # get unique set from each list
        ngram_list = list(set(ngram_df.ngram.values))
        target_list = list(set(encoded_df.Target.values))
        
        hd = lambda x, y: cpy.calc_hamming(x,y, self.max_hamming_dist)
        hamming_list = [hd(x, y) for y in target_list for x in ngram_list]

        # check that calls were found
        if len(hamming_list) < 1:
            raise SystemExit("No calls below hamming distance threshold.")
        else:
            return hamming_list
        

    def parse_hamming(self, hamming_list, ngram_df):
        '''
        purpose: combine hamming distance information with feature level data
        input: hamming_list: created in match_targets(), list of all ngrams, basecalls and hd
            ngram_df: specifies which gene and position a target came from
            encoded_df: provided feature level information
        output: hamming_df with featureID, gene, group, target, pos
        '''
        
        # create dataframe from list of matches and subset to hamming dist threshold
        hamming_df = pd.DataFrame(hamming_list, columns=["ref_match", "bc_feature", "hamming"])
        hamming_df = dd.from_pandas(hamming_df, npartitions=self.cpus)
        hamming_df = hamming_df[hamming_df.hamming != "X"].reset_index(drop=True)
        hamming_df = hamming_df.drop_duplicates()
     
        assert len(hamming_df) != 0, "No matches were found below hamming distance threshold."

        # match ngrams to their matching Target regions
        hamming_df = dd.merge(hamming_df, ngram_df, 
                            left_on="ref_match", 
                            right_on="ngram",
                            how="left")
        
        # match basecalls to their matching features
        hamming_df = dd.merge(hamming_df, self.encoded_df, 
                              left_on="bc_feature", 
                              right_on="Target",
                              how='left')
        
        # reorder columns  and subset
        hamming_df = hamming_df[["FeatureID", "id", "region", "pos", 
                                 "Target", "ref_match", 
                                 "BC", "cycle", "pool", "hamming"]]
        
        hamming_df["hamming"] = hamming_df.hamming.astype(int)
        hamming_df = hamming_df.compute()
        hamming_df = hamming_df.sort_values(by=["FeatureID", "pos"])
        
        # save raw hamming counts to file
        hamming_df.to_csv(self.raw_count_file , sep="\t", index=None)

        return hamming_df
    
    @jit
    def parse_hd1(self, hamming_df):
        '''
        purpose: separate kmers with HD1+ from perfect matches
            if only using HD0 for FTM, otherwise keep together
        input: hamming_df from parse_hamming()
        output: separate dataframes for perfect matches and hd1+
        '''
        
        # if running ftm with only HD0, separate HD1+
        if self.ftm_HD0_only:
            # separate perfect matches and hd1+
            hd_plus = hamming_df[hamming_df.hamming > 0]
            perfects = hamming_df[hamming_df.hamming == 0]
        
        # else keep HD1+ in perfects table
        else: 
            hd_plus = pd.DataFrame()
            perfects = hamming_df
            
        return hd_plus, perfects

    @jit
    def diversity_filter(self, input_df, diversity_threshold):
        '''
        purpose: filters out Targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only Targets per feature that meet threshold
        '''
        
        # filter out FeatureID/gene combos that are below feature_div threshold
        input_df["feature_div"] = input_df.groupby(["FeatureID", "region"])["Target"].transform("nunique") 
        
        diversified = input_df[input_df.feature_div >= diversity_threshold]
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
        
        # count multi-mapped reads
        diversified["multi"] = diversified.groupby(["FeatureID",
                                           "region",
                                           "pos"]).Target.transform('count')
                                           
        diversified = dd.from_pandas(diversified, npartitions=self.cpus)

        # separate multi and non multi mapped reads
        non = diversified[diversified["multi"] == 1]
        multi = diversified[diversified["multi"] > 1]
        
        # non multi mapped reads get a count of 1 each
        if len(non) > 1:
            non["counts"] = 1
        
        # noramlize multi-mapped reads
        if len(multi) > 1:
            multi["counts"] = 1/multi.multi
 
        # combine results
        counts = dd.concat([non, multi], 
                            interleave_partitions=True,
                            axis=0).drop("multi", axis=1).reset_index(drop=True)
        
        # round counts to two decimal places      
        counts["counts"] = counts["counts"].astype(float).round(2)
        counts = counts.compute()
        
        return counts
    
    @jit
    def barcode_counts(self, counts):
        '''
        purpose: sum counts for each barcode per position
        input: counts table from locate_MultiMapped
        output: bc_counts of counts aggregated by barcode to each region
        '''
        
        # sum counts per feature/region/pos
        counts["bc_count"] = counts.groupby(['FeatureID', 'Target', 'region', 'pos'])\
            ["counts"].transform("sum")
        
        ## calculate counts per barcode
        bc_counts = counts.drop(["counts"],axis=1)
        
        # information will be duplicated for each row, drop dups   
        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "region", "Target", "pos"],
                                                    keep="first")
        
        # sort counts for nicer output
        bc_counts = bc_counts.sort_values(by=["FeatureID", 
                                        "region", 
                                        "id",
                                        "Target",
                                        "pos"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        
        # save basecall normalized counts to file
        bc_counts.reset_index(inplace=True, drop=True)
        counts_outfile = "{}/bc_counts.tsv".format(self.output_dir)
        
        bc_counts.to_csv(counts_outfile, sep="\t", 
                      index=False, 
                      header=True)
        
        return bc_counts
    
    @jit   
    def regional_counts(self, bc_counts):
        '''
        purpose: sum together molecule counts for each region
        input: bc_counts table created in barcode_counts()
        output: regional counts for making ftm calls by region
        '''
        
        # drop columns 
        bad_cols = ['pool' 'cycle' 'ref_match' 'BC'] 
        
        for x in bc_counts.columns:
            if x in bad_cols:
                bc_counts = bc_counts.drop(x, axis=1)
               
        # subset and sum counts per region
        regional_counts = bc_counts
        
        regional_counts["counts"] = regional_counts.groupby(['FeatureID', 'region'])['bc_count'].transform("sum")
        
        # keep only relevant columns and drop duplicated featureID/gene combos
        regional_counts = regional_counts.drop_duplicates(subset=["FeatureID", "region"],
                                                   keep="first").reset_index(drop=True)

        # filter out featureID/gene combos below covg threshold
        regional_counts = regional_counts[regional_counts["counts"].astype(int) >= self.coverage_threshold]
        regional_counts = regional_counts.drop(["bc_count"], axis=1).reset_index(drop=True)
 
        assert len(regional_counts) != 0, "No matches found above coverage threshold." 

        return regional_counts
    
    def get_top2(self, regional_counts):
        '''
        purpose: locate top 2 target counts for perfect matches
        input: perfects built in ftm.sum_counts()
        output: hamming_df= calls within hamming threshold
            perfect matches with columns denoting max and second max counts
        '''

        # create columns identifying top 2 gene_count scores for perfects
        regional_counts["max_count"] = regional_counts.groupby("FeatureID")["counts"].transform("max")
        regional_counts["second_max"] = regional_counts.groupby("FeatureID")["counts"].transform(lambda x: x.nlargest(2).min())

        return regional_counts

    
    @jit
    def find_tops(self, regional_counts):
        '''
        purpose: retain only top two targets with highest counts
        input: perfects of perfect matches
        output: perfects subset to top two matches 
        '''
        
        # return all rows that match the max,2nd gene_count scores
        tops = regional_counts[(regional_counts.counts == regional_counts.max_count) | (regional_counts.counts == regional_counts.second_max)]
        tops = tops.reset_index(drop=True)
        
        # assign data types as floats for matching
        tops["counts"] = tops.counts.astype(float)
        tops["max_count"] = tops.max_count.astype(float)
        tops["second_max"] = tops.second_max.astype(float)
        
        return tops
    
    @jit
    def calc_symDiff(self, group, bc_counts):
        
        # subset bc_counts to only features with FTM calls
        # subset to relevant rows and features for speed 
        target_df = bc_counts[["FeatureID", "region", "Target"]].merge(group[["FeatureID",
                                                                             "region"]])        

        # create list of unique targets for each gene
        target_df["target_list"] = target_df.groupby(["FeatureID", "region"])["Target"].transform('unique')
        target_df = dd.from_pandas(target_df, npartitions=self.cpus)
        target_df= target_df.drop_duplicates(subset=["FeatureID", "region"],
                                                        keep="first")
        
        # convert target list to sets for quick comparison
        target_df["target_list"] = target_df["target_list"].apply(set)
        
        # calculate Targets unique to each region of interest
        symDiff = target_df.groupby("FeatureID").apply(cpy.calc_symmetricDiff).compute()
        symDiff = symDiff.reset_index(drop=False)
        symDiff.columns = ["FeatureID", "sym_diff"]
        
        return symDiff
    
    def parse_symDiff(self, group, bc_counts):
        
        df = self.calc_symDiff(group, bc_counts)
               
        # add symmetrical difference to top2
        symDiff = pd.concat([pd.DataFrame(v, columns=["sym_diff"])
                     for k,v in df.sym_diff.to_dict().items()]).reset_index()
        
        # add sym_diff column to top two df 
        group["sym_diff"] = symDiff.sym_diff
        
        return group
    
    @jit   
    def decision_tree(self, x, bc_counts):
        
        '''
        purpose: check coverage. If one has more than 3x coverage, keep. 
            Else check feature diversity scores, if one score is 2x or greater, keep
            else, keep score with highest sym_diff
        input: multi_df
            x = feature's top two rows
        output: used in return_ftm to return ties that pass this filtering logic
        '''
        
        # check if result with max count >= 3x second count
        result = x[x.counts > (3 * x.second_max)]
         
        # if there is only one result, return it
        if len(result) == 1:
            return result
        # otherwise check feature div
        else:
            result = x[x.feature_div >= (2 * x.feature_div.nlargest(2).min())]
             
        # if there is only one result now, return it
        if len(result) == 1:
            return result
         
        # otherwise check symmetrical difference
        else:
            sym_df = self.parse_symDiff(x, bc_counts)
            
            # take top sym_diff score for group
            max_symDiff = sym_df.sym_diff.max()
            
            result = sym_df[sym_df.sym_diff == max_symDiff]
        
        # if no decision at this point, no FTM call    
        if len(result) != 1:
            pass
            
    @jit
    def return_calls(self, ftm_df, hamming_df):
        '''
        purpose: save features where no ftm call can be made to a file
            subset bc_counts to contain only features where an FTM call can be made
        input: ftm_df and encoded_df
        output: no_calls = featureID's and basecalls for features that were not able to 
            be assigned an FTM call to be run through hamming.py
            perfect_calls = bc_counts of per feature/target counts for ftm called features
        '''
        
        # get counts for features with no FTM call 
        no_calls = hamming_df[~hamming_df.FeatureID.isin(ftm_df.FeatureID)]\
            .dropna().reset_index(drop=True)
                           
        if not no_calls.empty:
           
            # save no_call info to file
            no_ftm = no_calls[["FeatureID"]].drop_duplicates()
            no_ftm_file = Path("{}/no_ftm_calls.tsv".format(self.output_dir))
            no_ftm.to_csv(no_ftm_file, index=False, sep="\t")
            
        # pull out only calls related to FTM called region for HD under threshold   
        all_calls = ftm_df.merge(hamming_df, 
                             on=["FeatureID", "region"],
                             how="left")
        
        # format columns
        all_calls = all_calls.drop(["counts", "ref_match_x"
                           ], axis=1)
        
        drop_y = []
        for x in all_calls.columns:
            if x.endswith("_y"):
                drop_y.append(x)
                
        all_calls = all_calls.drop(drop_y, axis=1)
        
        # calculate mean diversity for filtering all calls
        div_filter = round(all_calls.feature_div.mean())
        all_calls = all_calls.drop("feature_div", axis=1)
        
        return all_calls, div_filter
    
    @jit
    def filter_allCalls(self, all_calls, div_thresh):

        # filter for feature diversity
        all_diversified = self.diversity_filter(all_calls, div_thresh)
           
        # normalize multi-mapped reads and count
        all_norm = self.locate_multiMapped(all_diversified)
           
        # group counts together for each bar code
        all_BCcounts = self.barcode_counts(all_norm)
        
        new_cols = []
        for x in all_BCcounts.columns:
            x = x.strip("_x")
            new_cols.append(x)
        all_BCcounts.columns = new_cols   
        
        all_BCcounts = all_BCcounts.sort_values(by=["FeatureID", 
                                        "region",
                                        "pos", 
                                        "Target"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        all_BCcounts.reset_index(inplace=True, drop=True)
        
        counts_file = Path("{}/ftm_counts.tsv".format(self.output_dir)) 
        all_BCcounts.to_csv(counts_file, sep="\t", index=False)

        return all_BCcounts
    
    def main(self):
        
        # create fasta dataframe and combine with mutations if provided
        fasta_df = self.create_fastaDF(self.fasta_df, self.mutant_fasta)
            
        # break reference seq into kmers
        fasta_df['ngrams'] = fasta_df.apply(self.create_ngrams, 
                              axis=1)
            
        # unravel dataframe with lists of ngrams
        ngram_df = self.parse_ngrams(fasta_df)
            
        # split ngrams and match with position
        ngram_df = pd.DataFrame(ngram_df.apply(lambda x: self.split_ngrams(x), axis=1))
            
        # parse the final ngram df
        ngrams = self.final_ngrams(ngram_df)
            
        # calculate hamming distance between input ref and features
        hamming_list = self.calc_hamDist(ngrams, self.encoded_df)
    
        # convert hamming_list into dataframe and parse
        hamming_df = self.parse_hamming(hamming_list, ngrams)
            
        # separate perfect matches from hd1+
        hd_plus, hd0 = self.parse_hd1(hamming_df)
            
        # filter for feature diversity
        hd0_diversified = self.diversity_filter(hd0, self.diversity_threshold)
            
        # normalize multi-mapped reads and count
        norm_counts = self.locate_multiMapped(hd0_diversified)
            
        # group counts together for each bar code
        bc_counts = self.barcode_counts(norm_counts)
            
        # sum counts for each region
        region_counts = self.regional_counts(bc_counts)
            
        # find top 2 scores for each bar code
        maxes = self.get_top2(region_counts)
            
        # pull out top 2 best options
        top2 = self.find_tops(maxes)
           
        # make FTM calls and save to file
        ftm_calls = top2.groupby("FeatureID").apply(self.decision_tree, bc_counts)
        
        if not ftm_calls.empty:
        
            # format df
            ftm_calls = ftm_calls.drop(["max_count", 
                                        "second_max", 
                                        "id", "pos",
                                        "Target"], axis=1).reset_index(drop=True) 
            ftm_calls["counts"] = ftm_calls.counts.astype(float).round(2)
  
            # output ftm to file
            ftm_file = Path("{}/ftm_calls.tsv".format(self.output_dir))
            ftm_calls.to_csv(ftm_file, sep="\t", index=False)
  
        else: 
            raise SystemExit("No FTM calls can be made on this dataset.")
        
        # save no_calls to a file and add HD1+ back in for sequencing
        all_calls, div_threshold = self.return_calls(ftm_calls, hamming_df)
        
        # filter all counts 
        all_counts = self.filter_allCalls(all_calls, div_threshold)
        
        return all_counts, hamming_df

        
       


        
        