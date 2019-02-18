'''
ftm.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches and hamming distance = self.max_ham_dist matches
- creates voting table and normalized counts 
- returns FTM calls
'''

import sys, os, logging, dask, psutil, gc
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numpy import single
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
from collections import Counter
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
from collections import defaultdict
from itertools import chain      
from dask.array.random import normal
pd.options.mode.chained_assignment = None

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
        self.hamming_weight = hamming_weight
        self.cpus = cpus
        self.kmer_lengths = list(self.encoded_df.bc_length.unique().compute(scheduler='processes', 
                                                                            num_workers=self.cpus))
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
        final_dd = self.client.compute(dd.from_pandas(ngram_df.apply(lambda x: \
                pd.Series(x[0]),axis=1).stack().reset_index(level=1, drop=True),
                npartitions=self.cpus))
        
        final_df =  pd.DataFrame(self.client.gather(final_dd))                                                          
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
        ngram_list = list(set(ngram_df.ngram))
        target_list =encoded_df.Target.compute(scheduler='threads',
                                               num_workers=self.cpus)
        
        hd = lambda x, y: cpy.calc_hamming(x,y, self.max_hamming_dist)
        hamming_list = [hd(x, y) for y in target_list for x in ngram_list]
        
        # check that calls were found
        if len(hamming_list) < 1:
            raise SystemExit("No calls below hamming distance threshold.")
        else:
            
            # remove any tuples that contain "X" for hamming distance
            hamming_list = [i for i in hamming_list if i[2] != "X"]
            
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
        hamming_df = hamming_df.drop_duplicates()
        hamming_df = hamming_df.set_index("ref_match")
        
        assert len(hamming_df) != 0, "No matches were found below hamming distance threshold."

        # match ngrams to their matching Target regions
        ngram_df = ngram_df.set_index("ngram")
        hamming_df = dd.merge(hamming_df, ngram_df, 
                            how="left")
        # get rid of old index, copy bc_feature to retain and set new index
        # for speedier merging
        hamming_df = hamming_df.reset_index(drop=True)
        hamming_df["Target"] = hamming_df["bc_feature"]
        hamming_df = hamming_df.set_index("bc_feature")
        
        # match basecalls to their matching features
        self.encoded_df = self.encoded_df.set_index("Target")
        hamming_df = dd.merge(hamming_df, self.encoded_df, 
                              how='left')
        hamming_df = hamming_df.reset_index(drop=True)
        
        # reorder columns  and subset
        hamming_df = hamming_df[["FeatureID", "id", "region",  
                                 "chrom", "pos", "Target", 
                                 "BC", "cycle", "pool", "hamming"]]
        
        # save raw hamming counts to file and remove from memory
        outfile = os.path.join(self.output_dir, "rawCounts")
        hamming_df.to_parquet(outfile, engine='fastparquet')
        
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
        
        a = input_df.groupby(["FeatureID", "region"])["Target"].nunique().to_frame("feature_div")
        input_df = input_df.join(a, on=["FeatureID", "region"],
                                 npartitions=self.cpus)
        
        # filter out features below diversity threshold
        diversified = input_df[input_df.feature_div >= diversity_threshold]
        
        # check that calls were found
        if len(diversified) > 1:
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
        a = diversified.groupby(["FeatureID", "region", "Target"])["pos"].nunique().to_frame("multi")
        diversified = diversified.join(a, on=["FeatureID", "region", "Target"],
                                 npartitions=self.cpus)

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
        
        return counts
    
    @jit
    def barcode_counts(self, counts):
        '''
        purpose: sum counts for each barcode per position
        input: counts table from locate_MultiMapped
        output: bc_counts of counts aggregated by barcode to each region
        '''
        
        # sum counts per feature/region/pos
        a = counts.groupby(["FeatureID", "Target", "region", "pos"])["counts"].sum().to_frame("bc_count")
        counts = counts.join(a, on=["FeatureID", "Target", "region", "pos"],
                                 npartitions=self.cpus)

        # calculate counts per barcode
        bc_counts = counts.drop(["counts"],axis=1)
        
        # information will be duplicated for each row, drop dups   
        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "region", "Target", "pos"])
        
        return bc_counts
    
    @jit   
    def regional_counts(self, bc_counts):
        '''
        purpose: sum together molecule counts for each region
        input: bc_counts table created in barcode_counts()
        output: regional counts for making ftm calls by region
        '''
               
        # subset and sum counts per region
        a = bc_counts.groupby(["FeatureID", "region"])["bc_count"].sum().to_frame("counts")
        bc_counts = bc_counts.join(a, on=["FeatureID", "region"],
                                 npartitions=self.cpus)
        
        # keep only relevant columns and drop duplicated featureID/gene combos
        regional_counts = bc_counts[["FeatureID", "region", "feature_div", "counts"]]
        regional_counts = regional_counts.drop_duplicates(subset=["FeatureID", "region"])
        
        # filter out featureID/gene combos below covg threshold
        regional_counts = regional_counts[regional_counts.counts >= self.coverage_threshold]
        
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
        a = regional_counts.groupby(["FeatureID"])["counts"].max().to_frame("max_count")
        regional_counts = regional_counts.join(a, on=["FeatureID"],
                                 npartitions=self.cpus)
        
        b = regional_counts.groupby(["FeatureID"])["counts"].apply(lambda x: x.nlargest(2).min()).to_frame("second_max")
        regional_counts = regional_counts.join(b, on=["FeatureID"],
                                 npartitions=self.cpus)
        
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
        
        # assign data types as floats for matching
        tops["counts"] = tops.counts.astype(float)
        tops["max_count"] = tops.max_count.astype(float)
        tops["second_max"] = tops.second_max.astype(float)

        return tops
    
    
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
                
                x["FeatureID"] = x.name
                print(x)

                # pull out relevant Targets
                target_df = bc_counts[(bc_counts.FeatureID.isin([x.FeatureID]))
                                      & (bc_counts.region.isin(x.region))]
                print(target_df.head())
                raise SystemExit()
 

                # create list of unique targets for each gene
                target_df["target_list"] = target_df.groupby(["FeatureID", "region"])["Target"].transform('unique')
                target_df= target_df.drop_duplicates(subset=["FeatureID", "region"],
                                                                keep="first")
                
                # convert target list to sets for quick comparison
                target_df["target_list"] = target_df["target_list"].apply(set)
                
                # calculate Targets unique to each region of interest
                target_df["sym_diff"] =cpy.calc_symmetricDiff(target_df)
        
                target_df = target_df.reset_index(drop=True)

                sym_diff = self.calc_symDiff(x, bc_counts)
                
                group = x.merge(sym_diff[["FeatureID", "region", "sym_diff"]],
                               on=["FeatureID", "region"])
                
                # take top sym_diff score for group
                max_symDiff = group.sym_diff.max()
                
                result = group[group.sym_diff == max_symDiff]
                
                result = result.drop("sym_diff", axis=1)

                # if no decision at this point, no FTM call    
                if len(result) == 1:
                    return result
                else:
                    pass
                    
    def find_multis(self, tops, bc_counts):
        '''
        purpose: located features that have tied counts for best FTM call
        input: count_df created in ftm.sum_counts.py
        output: dataframe of ties and non-tied counts for passing to return_ftm
        '''

        # separate ties to process
        # subset and sum counts per region
        groups = tops.groupby(["FeatureID"]).count().reset_index()
        groups = groups[["FeatureID", "region"]]
        groups.columns = ["FeatureID", "grp_size"]
        tops = tops.assign(grp_size=groups.grp_size)
        
        singles = tops[tops.grp_size == 1]
        multis = tops[tops.grp_size > 1]
        
        if len(multis) > 1:
            
            # process multis
            multi_df = multis.groupby("FeatureID").apply(self.decision_tree, bc_counts).compute()
            
            # concat results with singles
            ftm_counts = pd.concat([singles, multi_df])
        
        else: 
            
            ftm_counts = singles
            
        assert len(ftm_counts) > 0, "No FTM results."
        
        # drop extra columns
        ftm_counts = ftm_counts[['FeatureID', 'region', 'feature_div',
                                 'counts']].reset_index(drop=True)
        
         # output ftm to file
        ftm_file = os.path.join(self.output_dir, "ftm_calls.tsv")
        ftm_counts.to_csv(ftm_file, sep="\t", index=False)
    
        return ftm_counts
    
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
            no_ftm_file = os.path.join(self.output_dir, "no_ftm_calls.tsv")
            no_ftm.to_csv(no_ftm_file, index=False, sep="\t")
           
        # pull out only calls related to FTM called region for HD under threshold   
        hd = hamming_df.drop("id", axis=1)
        
        all_calls = ftm_df.merge(hd, 
                             on=["FeatureID", "region"],
                             how="left")
        
        # format columns
        all_calls = all_calls.drop(["counts"], axis=1)
        
        return all_calls
    
    @jit
    def filter_allCalls(self, all_calls):
           
        # normalize multi-mapped reads and count
        all_norm = self.locate_multiMapped(all_diversified)
           
        # group counts together for each bar code
        all_calls = self.barcode_counts(all_norm)

        return all_calls
    
    @jit
    def format_allCounts(self, all_calls):
        
        # sort output for saving to file
        all_calls = all_calls.sort_values(by=["FeatureID", 
                                        "region",
                                        "pos", 
                                        "Target"], 
                                        axis=0, 
                                        ascending=True, 
                                        inplace=False)
        all_calls.reset_index(inplace=True, drop=True)

        counts_file = os.path.join(self.output_dir, "all_counts.tsv") 
        all_calls.to_csv(counts_file, sep="\t", index=False)

        return all_calls
    
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
        # save basecall normalized counts to file
        # save raw hamming counts to file and remove from memory
        bc_out = os.path.join(self.output_dir, "bc_counts")
        bc_counts.to_parquet(bc_out, engine='fastparquet')
        
        # drop unnecessary columns
        bc_counts = bc_counts.drop(['pool', 'cycle', 'BC'],
                                   axis=1) 
          
        # sum counts for each region
        region_counts = self.regional_counts(bc_counts)
            
        # find top 2 scores for each bar code
        maxes = self.get_top2(region_counts)
            
        # pull out top 2 best options
        top2 = self.find_tops(maxes)
        
        top2.to_parquet("./top2", engine='fastparquet')

        # separate and proces features with more than one 
        # possible ftm call        
        bc_counts2 = bc_counts[["FeatureID", "region", "Target"]]
        bc_counts2.to_parquet("./bc_counts2", engine='fastparquet')
        
        
        ftm_calls = self.find_multis(top2, bc_counts2)
        
        
        
        
        
        
        
        
        
        # save no_calls to a file and add HD1+ back in for sequencing
        all_calls = self.return_calls(ftm_calls, hamming_df)
        
        # filter all counts 
        all_counts = self.filter_allCalls(all_calls)
        
        # format final output
        all_counts = self.format_allCounts(all_counts)

        return all_counts, hamming_df, ftm_calls

        
       


        
        