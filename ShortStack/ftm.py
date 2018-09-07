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
import concurrent.futures as cf

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
        self.hamming_file = "{}/hamming_dist_all.tsv".format(self.output_dir)
        self.count_file = "{}/normCounts.tsv".format(self.output_dir)
        self.run_info_file = run_info_file
        self.num_cores = num_cores 
        
        # remove before release
        self.div_file = "{}/diversified_hammingDist_all.tsv".format(self.output_dir)
        
    def parallelize_dataframe(self, df, func):
        
        chunks = round(df.shape[0]/10)
        df_split = np.array_split(df, chunks)
        
        with cf.ProcessPoolExecutor(self.num_cores) as pool:
            df = pd.concat(pool.map(func, df_split))
        
        return df
    
    @jit    
    def target_count(self):
        '''
        purpose: count number of targets found by position within each feature
        input: encoded_df of feature ID and encoded color call
        output: raw_counts.tsv file of raw counts
        '''
        # create output file path for raw counts
        counts_outfile = "{}/bc_counts.tsv".format(self.output_dir)
        
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
                                        'target', 'BC', 'Category', 
                                        'PoolID', 'Qual', 
                                        'pos']])
        # add hamming distance for later joins with snv dataframe
        matches["hamming"] = 0
        
        return matches
    
    @jit 
    def diversity_filter(self, input_df):
        '''
        purpose: filters out targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only targets per feature that meet threshold
        '''
        
        # group by feature ID and gene
#         input_df["diverse"] = np.where(input_df.groupby(\
#             ["FeatureID", "gene"])["pos"].transform('nunique') > self.diversity_threshold, "T", "")
        
        # count number of unique read positions matched for each FeatureID/gene
        input_df["feature_div"] = input_df.groupby(\
            ["FeatureID", "gene"])["pos"].transform('nunique')
            
        # filter out FeatureID/gene combos that are below diversity threshold
        diversified = input_df[input_df.feature_div >= self.diversity_threshold]
        diversified.reset_index(inplace=True, drop=True)
               
        return diversified

    @jit
    def locate_multiMapped(self, diversified):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: diversity filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # flag targets that are multi-mapped
        diversified['multi_mapped'] = np.where(diversified.groupby(["FeatureID", "gene", "BC"]).pos\
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
        multi["count"] = multi.groupby(["FeatureID", "gene", "BC"])["pos"]\
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
        
        # sort counts for nicer output
        counts = counts.sort_values(by=["FeatureID", "gene", "target", "pos"], axis=0, ascending=True, inplace=False)
        counts.reset_index(inplace=True, drop=True)

        # save raw counts to file
        counts.to_csv(self.count_file, sep="\t", 
                      index=False, 
                      header=True)
        
        # sum counts per gene
        counts = counts[["FeatureID", "gene", "feature_div", "count"]]
        counts["gene_count"] = counts.groupby(['FeatureID', 'gene'])['count'].transform("sum")
       
        # keep only relevant columns
        count_df = counts.drop_duplicates(subset=["FeatureID", "gene"],
                                                   keep="first")
        
        # filter out featureID/gene combos below covg threshold
        count_df = count_df[count_df.gene_count >= self.coverage_threshold]
        
        return count_df
     
    @jit
    def return_ftm(self, count_df):
        
        ### locate rows with max count ###
         
        # create column containing max count 
        count_df["max_count"] = count_df.groupby(["FeatureID"])["gene_count"].transform("max")
         
        # return all rows that match max zscore 
        max_df = count_df[count_df['gene_count'] == count_df['max_count']]
        max_df = max_df.drop("max_count", axis=1).reset_index(drop=True)
        
        # seperate ties to process
        dups = max_df[max_df.FeatureID.duplicated(keep=False)]
        non = max_df[~max_df.FeatureID.duplicated(keep=False)]
        
        ### process ties ###
        
        # create column containing max diversity 
        dups["max_div"] = dups.groupby(["FeatureID"])["feature_div"].transform("max")
         
        # return all rows that match max diversity 
        div_df = dups[dups['feature_div'] == dups['max_div']]
        
        # drop any duplicated featureID as uncallable
        div_df.drop_duplicates(subset="FeatureID", keep=False, inplace=True)
        
        # remove extraneous columns
        div_df = div_df.drop(["max_div", "count"], axis=1).reset_index(drop=True) 
        non = non.drop(["count"], axis=1).reset_index(drop=True)
       
        # concat dataframe back together
        ftm = pd.concat([div_df, non])
       
        # format ftm df for output to file
        ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
        ftm.to_csv(ftm_calls, sep="\t", index=False)
        
        return ftm
        
            
    
#     def calc_Zscore(self, norm_counts):
#         '''
#         purpose: calculate zscore between results 
#         note: zscore calc output identical to scipy.stats.zscore, but numpy is faster
#         input: filtered, normalized counts dataframe created in align.normalize_counts()
#         output: df of top2 calls with zscores
#         '''
#         
#         # sum counts per gene
#         count_df = norm_counts.groupby(['FeatureID', 'gene'])['count'].sum().reset_index()
#         
#         # formula for calculating zscore
# #         zscore = lambda x: (x - np.mean(x)) / np.std(x)
#         # add zscore for each feature count
#         count_df["zscore"] = count_df.groupby("FeatureID")["count"].apply(cpy.calc_zscore)
#         raise SystemExit(count_df.head())
#         return count_df

#     def return_ftm(self, count_df):
#     
#         ### locate rows with max zscore ###
#         
#         # create column containing max zscore 
#         max_z = count_df.groupby(['FeatureID']).agg({'zscore':'max'}).reset_index()
#         
#         # merge max zscore column back with original results, many to one
#         max_df = pd.merge(count_df, max_z, how='left', on=['FeatureID'])
#         # rename columns 
#         max_df = max_df.rename(columns={'zscore_x':'zscore', 'zscore_y':'max_z'})
#         
#         # return all rows that match max zscore 
#         max_df = max_df[max_df['zscore'] == max_df['max_z']]
#         max_df = max_df.drop("max_z", axis=1)
#         
#         # remove features with ties
#         max_df.drop_duplicates(subset="FeatureID", keep=False, inplace=True)
#         
#         # calculate cumulative distribution function of the zscore
#         max_df["cdf"] = max_df["zscore"].swifter.apply(lambda x: stats.norm.cdf(x))
#         
#         # format ftm df for output to file
#         ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
#         max_df.to_csv(ftm_calls, sep="\t", index=False)
#         
#         # subset ftm dataframe for memory efficiency to pass to variant graph
#         ftm_df = max_df[["FeatureID", "gene"]]
#         ftm_df.columns = ["FeatureID", "FTM_call"]
#         ftm_df.reset_index(inplace=True, drop=True)
# 
#         return ftm_df

    def main(self):
        
        # create raw counts file
        print("Counting reads...\n")
        self.target_count()
        
        # break reference seq into kmers
        print("Breaking up reference sequence into kmers...\n")
        ngrams = self.create_ngrams(self.fasta_df, self.mutant_fasta)        
        
        # match targets with basecalls
        print("Finding perfectly matched kmers...\n")
        perfects = self.match_targets(ngrams)

        # filter for feature diversity
        print("Filtering results for feature diversity...\n")
        diversified_hamming = self.diversity_filter(perfects)
        
        # locate multi-mappped reads
        print("Normalizing counts...\n")
        multi, non = self.locate_multiMapped(diversified_hamming)
        
        # normalize multi-mapped reads
        multi = self.normalize_multiMapped(multi)

        # finalize normalized counts
        norm_counts = self.finalize_counts(multi, non)
        
        # return ftm calls
        ftm = self.return_ftm(norm_counts)

        # return ftm matches and diversity filtered non-perfects
        return ngrams, normalized_counts, ftm
        

        