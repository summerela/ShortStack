'''
align.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches
- creates voting table and counts perfect matches for FTM
- returns table of counting stats that is saved to output folder
- returns voting table from first round of FTM
'''

import pandas as pd
import logging
import re
import scipy.stats as stats
import numpy as np
import time
import pyximport; pyximport.install()
import cython_funcs as cpy
from itertools import count

# import logger
log = logging.getLogger(__name__)

class Align():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, detection_mode, 
                 deltaz_threshold, kmer_size,
                 output_dir, diversity_threshold,
                 qc_outfile, run_info_file):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.detection_mode = detection_mode
        self.deltaz_threshold = deltaz_threshold
        self.kmer_size = kmer_size
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.qc_outfile = qc_outfile
        self.count_file = "{}/normalized_counts.tsv".format(self.output_dir)
        self.run_info_file = run_info_file
        
    def target_count(self):
        '''
        purpose: count number of targets found by position within each feature
        input: encoded_df of feature ID and encoded color call
        output: raw_counts.tsv file of raw counts
        '''
        # create output file path for raw counts
        counts_outfile = "{}/raw_counts.tsv".format(self.output_dir)
        
        # sort encoded dataframe by featureID
        self.encoded_df.sort_values("FeatureID", inplace=True)
        # count number of targets found in each barcode
        feature_counts = self.encoded_df.groupby(["FeatureID", "Target"]).size().reset_index()
        # update columnnames        
        feature_counts.columns = ["FeatureID", "Target", "count"]
        # save raw count to file
        feature_counts.to_csv(counts_outfile, sep="\t", index=None)
           
    def match_perfects(self):
        '''
        purpose: compares each input ref sequence with each basecall
        using ngrams() from cython_funcs
        input: encoded df and fasta df from run_shortstack.py
        output: dataframe of matches and non-matches to encoded_df
        '''
        
        # if supervised, high_fidelity mode, add mutants to fasta_df
        if self.detection_mode.lower() == "high_fidelity":
            fasta_df = pd.concat([self.fasta_df, self.mutant_fasta], axis=0)
        # otherwise run FTM without mutants
        else:
            fasta_df = self.fasta_df

        # create series of targets for faster comparison
        seq_series = pd.Series(fasta_df.seq.values, 
                               index=fasta_df.id.values,
                               dtype="object")
        
        # break apart each sequence into ngrams
        ngram_dict = {}
        for gene, seq in seq_series.iteritems():
            ngram_dict[gene] = cpy.ngrams(seq.strip(), n=self.kmer_size)
        
        # merge ngrams with gene id information    
        ngrams = pd.DataFrame(pd.concat({k: pd.Series(v) for k, v in ngram_dict.items()})).reset_index()
        ngrams.columns = ["gene", "pos", "ngram"]
        
        # set indices on dataframes for faster merging
        ngrams.set_index("ngram", inplace=True)
        self.encoded_df.set_index("Target", inplace=True)
        
        # inner join between kmer seqs and targets to get matches
        matches = self.encoded_df.join(ngrams, how="inner").reset_index().rename(columns={'index' : 'target'})
        
        # pull out non-perfect matches to pass along for variant calling
        self.encoded_df.reset_index(inplace=True)
        self.encoded_df["id"] = self.encoded_df.FeatureID.astype(str) + "_" + self.encoded_df.Target.astype(str)
        self.encoded_df.set_index("id", inplace=True)
        
        matches["id"] = matches.FeatureID.astype(str) + "_" + matches.target.astype(str)
        matches.set_index("id", inplace=True)
        non_perfects = self.encoded_df.loc[self.encoded_df.index ^ matches.index].reset_index()
        
        return non_perfects, matches
    
    def diversity_filter(self, matches_df):
        '''
        purpose: filters out targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only targets per feature that meet threshold
        '''
        
        # group by feature ID and gene
        matches_df["diverse"] = np.where(matches_df.groupby(\
            ["FeatureID", "gene"])["pos"].transform('nunique') > self.diversity_threshold, "T", "")
        
        diversified = matches_df[matches_df.diverse == "T"]
        diversity_filtered = matches_df[matches_df.diverse != "T"]
        diversified.drop("diverse", inplace= True, axis=1)
        diversified.reset_index(inplace=True, drop=True)
        
        if diversity_filtered.shape[0] > 1:
            diversity_filtered.drop("diverse", inplace=True, axis=1)
            diversity_filtered.reset_index(drop=True, inplace=True)
            filter_message = "The following calls were filtered out for being below the diversity threshold of {}:\n {}\n".\
                format(self.diversity_threshold, diversity_filtered)
            with open(self.run_info_file,'a') as f:
                f.writelines(filter_message)
        
        return diversified
        
    
    def normalize_counts(self, diversified_df):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: diversity filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # flag targets that are multi-mapped
        diversified_df['multi_mapped'] = np.where(diversified_df.groupby(["FeatureID", "gene", "target"]).pos\
                                    .transform('nunique') > 1, "T", '')
        
        # separate multi and non multi mapped reads
        non = diversified_df[diversified_df["multi_mapped"] != "T"]\
            .drop("multi_mapped", axis=1)
        multi = diversified_df[diversified_df["multi_mapped"] == "T"]\
            .drop("multi_mapped", axis=1)
        
#         # add counts to non multi mapped reads
        non = non.groupby(["FeatureID", "gene", "target"])\
            .count().reset_index().rename(columns={"pos":"count"})
#         
        # add counts to multi-mapped reads with normaliztion 
        multi["count"] = multi.groupby(["FeatureID", "gene", "target"])\
              .transform(lambda x: 1/x.nunique())
        multi.drop("pos", axis=1, inplace=True)
        
        # join the multi and non back together
        counts = pd.concat([multi, non], axis=0).sort_values("FeatureID")
        # save raw counts to file
        counts.to_csv(self.count_file, sep="\t", index=False, header=True)
        
        counts = counts.drop("target", axis=1).reset_index(drop=True)

        return counts
    
    def score_matches(self, counts):
        '''
        purpose: calculate zscore between results and deltaZ between top 2 results
        note: zscore calc output identical to scipy.stats.zscore, but numpy is faster
        input: filtered, normalized counts dataframe created in align.normalize_counts()
        output: pass_z = perfect match call, zscores and deltaZ scores per target
        '''
        # group matches by feature ID and gene, then sum counts for each group
        count_df = counts.groupby(["FeatureID", "gene"]).sum().reset_index()

        # formula for calculating zscore
        zscore = lambda x: (x - np.mean(x)) / np.std(x)
        
        # add zscore for each feature count
        count_df["zscore"] =  count_df.groupby("FeatureID")["count"].apply(zscore)
        
        # filter out NaN zscores as they represent a tie
        count_df.dropna(subset=["zscore"], inplace=True)
                         
        # select top two genes with highest counts
        top2 = count_df.groupby(["FeatureID"], as_index=False)\
             .apply(lambda x:x.nlargest(2, columns=['zscore'])).reset_index(drop=True)
             
        # calculate delta zscore
        deltas = top2.groupby("FeatureID")\
            .transform(np.diff).rename(columns={"zscore":"deltaZ"})\
            .drop("count", axis=1).reset_index(drop=True)
        deltas["deltaZ"] = abs(deltas["deltaZ"])
         
        # add zscores to counts table and keep max
        scored_df = pd.concat([top2, deltas], axis=1)
        ftm = scored_df.sort_values('count', ascending=False)\
            .drop_duplicates(['FeatureID'])
        ftm_df = ftm.sort_values("FeatureID", ascending=True).reset_index(drop=True)
        
        # filter by zscore threshold
        pass_z = ftm_df[ftm_df.deltaZ.astype(float) >= float(self.deltaz_threshold)]
        under_z = ftm_df[ftm_df.deltaZ.astype(float) < float(self.deltaz_threshold)]\
            .reset_index(drop=True)
        
        # add calls below zscore threshold to run_info.txt  
        if under_z.shape[0] > 0:
            filter_message = "The following calls were filtered for being below the detlaZ threshold of {}:\n {}\n"\
            .format(self.deltaz_threshold, under_z)
            with open(self.run_info_file, 'a') as f:
                f.writelines(filter_message)
        
        ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
        pass_z.to_csv(ftm_calls, sep="\t", index=False)

        return pass_z


    def main(self):
        self.target_count()
        nonperfects, perfects = self.match_perfects()
        diversified_matches = self.diversity_filter(perfects)
        normalized_counts = self.normalize_counts(diversified_matches)
        ftm =self.score_matches(normalized_counts)
        return ftm, nonperfects
        

        