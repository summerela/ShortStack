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

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, detection_mode, 
                 deltaz_threshold, kmer_size,max_hamming_dist,
                 output_dir, diversity_threshold,
                 qc_outfile, run_info_file):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.mutant_fasta = mutant_fasta
        self.detection_mode = detection_mode
        self.max_hamming_dist = max_hamming_dist
        self.deltaz_threshold = deltaz_threshold
        self.kmer_size = kmer_size
        self.output_dir = output_dir
        self.diversity_threshold = diversity_threshold
        self.qc_outfile = qc_outfile
        self.hamming_file = "{}/hamming_dist_all.tsv".format(self.output_dir)
        self.count_file = "{}/perfects_normCounts.tsv".format(self.output_dir)
        self.run_info_file = run_info_file
    
    @jit    
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

    def create_ngrams(self):
        '''
        purpose: breaks apart reference sequence into self.kmer_length kmers
        input: fasta_df, mutation_df if provided
        output: ngrams dataframe with kmers and associated gene and position on ref
        '''
        
        # if supervised, add mutants to fasta_df
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
        
        # merge ngrams with gene id and starting pos of kmer    
        ngrams = pd.DataFrame(pd.concat({k: pd.Series(v) for k, v in ngram_dict.items()})).reset_index()
        ngrams.columns = ["gene", "pos", "ngram"]
        ngrams.reset_index()

        return ngrams
    
    def calc_hamming(self, ngrams):
        '''
        purpose: calculate hamming distance between basecalls and targets
        input: ngrams created in ftm.create_ngrams(), 
        output: dataframe of potential vars's with hamming <= max_hamming_dist
        '''

        # calc hamming distance for non-perfect matches
        hamming_df = self.encoded_df.Target.apply(lambda bc: ngrams.ngram.\
                                apply(lambda x: cpy.calc_hamming(bc, x)))
        
        # add labels for columns and index so we retain information
        hamming_df.columns = ngrams.ngram.astype(str) + \
                         ":" + ngrams.gene.astype(str) + \
                         ":" + ngrams.pos.astype(str)       
        hamming_df.index = self.encoded_df.FeatureID.astype(str) + ":" +  \
            self.encoded_df.Target.astype(str)  
        
        # reshape dataframe for parsing    
        hamming2 = hamming_df.stack()  
        hamming2 = hamming2.reset_index()
        hamming2.columns = ["Target", "Ref", "Hamming"]     

        return hamming2
    
    @jit
    def parse_vars(self, hamming_df):
        '''
        purpose: reformat the hamming distance df 
        input: dataframe of reads with hamming distance <= self.max_hamming_dist
        output: reshaped hamming_df 
        '''
                    
        # split hamming_df index into columns to add back into dataframe
        targets = pd.DataFrame(hamming_df.Target.str.split(':',1).tolist())
        targets.columns = ["FeatureID", "BC"]
        matches =  pd.DataFrame(hamming_df.Ref.str.split(':',1).tolist())
        matches.columns = ["Target_Match", "Ref"]
        genes = pd.DataFrame(matches.Ref.str.split(':',1).tolist())
        genes.columns = ["Ref", "Pos"]
        matches.drop("Ref", axis=1, inplace=True)
        hamming = hamming_df.Hamming
        
        # concatenate dataframe back together
        kmers = pd.concat([targets, matches, genes, hamming], axis=1)
    
        # join kmers with fasta_df to get ref position info
        hamming_df = kmers.merge(self.fasta_df[["chrom", "start", "id"]], \
                                 left_on="Ref", right_on="id")     
        
        # calculate starting position of match on reference, subtract 1 for 0 based indexing
        hamming_df["pos"] = (hamming_df["start"].astype(int) \
                             - hamming_df["Pos"].astype(int)) - 1
        
        # drop unnecessary columns
        hamming_df.drop(["id", "Pos", "start"], inplace=True, axis=1) 
        hamming_df.reset_index(inplace=True, drop=True)
        
        # save full hamming_df to file
        hamming_df.to_csv(self.hamming_file, sep="\t", 
                      index=False, 
                      header=True)
        
        # filter hamming df by self.max_hamming_dist
        hamming_df = hamming_df[hamming_df.Hamming <= self.max_hamming_dist]

        return hamming_df
    
#     @jit             
#     def match_targets(self, ngrams):
#         '''
#         purpose: compares each input ref sequence with each basecall
#         using ref_ngrams() from cython_funcs
#         input: encoded targets, fasta_df from run_shortstack.py
#         output: dataframe of matches and non-matches to encoded_df
#         '''
#         # set indices on dataframes for faster merging
#         ngrams.set_index("ngram", inplace=True)
#         self.encoded_df.set_index("Target", inplace=True)
#         
#         # inner join between kmer seqs and targets to get matches
#         matches = self.encoded_df.join(ngrams, how="inner")\
#             .reset_index().rename(columns={'index' : 'target'})
#        
#         # return unmatched for hamming measurement
#         self.encoded_df.reset_index(inplace=True)
#         self.encoded_df["id"] = self.encoded_df.FeatureID.astype(str) + "_" + self.encoded_df.Target.astype(str)
#         matches["id"] = matches.FeatureID.astype(str) + "_" + matches.target.astype(str)
#         
#         unmatched = self.encoded_df[~self.encoded_df['id'].isin(matches['id'])]
#         
#         self.encoded_df.drop(["id"], axis=1, inplace=True)
#         matches.drop(["id"], axis=1, inplace=True)
#         unmatched.drop(["id"], axis=1, inplace=True)
#         
#         return unmatched, matches
    
    def diversity_filter(self, input_df):
        '''
        purpose: filters out targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only targets per feature that meet threshold
        '''

        # group by feature ID and gene
        input_df["diverse"] = np.where(input_df.groupby(\
            ["FeatureID", "Ref"])["pos"].transform('nunique') > self.diversity_threshold, "T", "")
        
        # filter out targets that do not meet diversity threshold
        diversified = input_df[input_df.diverse == "T"]
        diversity_filtered = input_df[input_df.diverse != "T"]
        diversified.drop("diverse", inplace= True, axis=1)
        diversified.reset_index(inplace=True, drop=True)
        
         # find all variants that overlap with max_hamming_dist
        perfects = diversified[diversified.Hamming == 0]
        
        # save diversified hamming_df temporarily for Nicole
        diversified.to_csv("./output/diversified_hamming_all.tsv", sep="\t", 
                      index=False, 
                      header=True)
                
        return diversified, perfects
    
    def normalize_counts(self, diversified_df):
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: diversity filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''

        # flag targets that are multi-mapped
        diversified_df['multi_mapped'] = np.where(diversified_df.groupby(["FeatureID", "Ref", "BC"]).pos\
                                    .transform('nunique') > 1, "T", '')
        
        # separate multi and non multi mapped reads
        non = diversified_df[diversified_df["multi_mapped"] != "T"]\
            .drop("multi_mapped", axis=1)   
        multi = diversified_df[diversified_df["multi_mapped"] == "T"]\
            .drop("multi_mapped", axis=1)
        
         # add counts to non multi mapped reads
        non["count"] = non.groupby(["FeatureID", "Ref", "BC"])["pos"]\
            .transform("count")
    
        # add counts to multi-mapped reads with normaliztion 
        multi["count"] = multi.groupby(["FeatureID", "Ref", "BC"])["pos"]\
              .transform(lambda x: 1/x.nunique())
        
        # join the multi and non back together
        counts = pd.concat([multi, non], axis=0).sort_values("FeatureID")       
        counts["count"] = counts["count"].round(2)
        counts.drop(["Target_Match"], axis=1, inplace=True)
        counts.reset_index(inplace=True, drop=True)

        # save raw counts to file
        counts.to_csv(self.count_file, sep="\t", 
                      index=False, 
                      header=True)
        return counts
    
    
    def calc_Zscore(self, counts):
        '''
        purpose: calculate zscore between results 
        note: zscore calc output identical to scipy.stats.zscore, but numpy is faster
        input: filtered, normalized counts dataframe created in align.normalize_counts()
        output: df of top2 calls with zscores
        '''
        # group matches by feature ID and gene, then sum counts for each group
        count_df = counts.groupby(["FeatureID", "Ref"]).sum().reset_index()

        # formula for calculating zscore
        zscore = lambda x: (x - np.mean(x)) / np.std(x)
        
        # add zscore for each feature count
        count_df["zscore"] =  count_df.groupby("FeatureID")["count"].apply(zscore)
        # filter out NaN zscores as they represent a tie
        count_df.dropna(subset=["zscore"], inplace=True)
        
        return count_df

    def return_ftm(self, count_df):
    
        ### locate rows with max zscore ###
        
        # create column containing max zscore and return rows that match that score
        # note: this is faster than the one-line transform function
        max_z = count_df.groupby(['FeatureID']).agg({'zscore':'max'}).reset_index()
        max_df = pd.merge(count_df, max_z, how='left', on=['FeatureID'])
        max_df = max_df.rename(columns={'zscore_x':'zscore', 'zscore_y':'max_z'})
        # this will return all rows with max zscore so we can filter out ties
        max_df = max_df[max_df['zscore'] == max_df['max_z']]
        max_df = max_df.drop("max_z", axis=1)
        
        # remove features with ties
        max_df.drop_duplicates(subset="FeatureID", keep=False, inplace=True)
        
        # calculate cumulative distribution function of the zscore
        max_df["cdf"] = max_df["zscore"].swifter.apply(lambda x: stats.norm.cdf(x))
        
        ftm_calls = "{}/ftm_calls.tsv".format(self.output_dir)
        max_df.to_csv(ftm_calls, sep="\t", index=False)
        
        return max_df

    def main(self):
        
        # create raw counts file
        print("Counting reads...\n")
        self.target_count()
        
        # break reference seq into kmers
        print("Breaking up reference sequence into kmers...\n")
        ngrams = self.create_ngrams()

        # match targets with basecalls
        print("Calculating hamming distance...\n")
        hamming_df = self.calc_hamming(ngrams)
        
        # parse results of hamming calculation
        print("Parsing results...\n")
        hamming_df = self.parse_vars(hamming_df)
        
        # filter for feature diversity
        print("Filtering results for feature diversity...\n")
        diversified_hamming, perfects = self.diversity_filter(hamming_df)
        
        # normalize multi-mapped reads
        print("Normalizing multi-mapped reads...\n")
        normalized_counts = self.normalize_counts(perfects)

        # calc zscore for each feature
        print("Calculating Zscores...\n")
        zscored =self.calc_Zscore(normalized_counts)
        
        # calc delta zscore between all targets and select best match
        print("Running FTM...\n")
        ftm =self.return_ftm(zscored)

        # return ftm matches and diversity filtered non-perfects
        return ngrams, ftm, diversified_hamming
        

        