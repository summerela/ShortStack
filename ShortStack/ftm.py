'''
ftm.py
- compare basecalls with reference sequence to create voting table for FTM
- bins perfect matches and hamming distance = self.max_ham_dist matches
- creates voting table and normalized counts 
- returns FTM calls
'''

import sys, os, logging, shutil
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import cython_funcs as cpy
from numba import jit
import pandas as pd
import dask.dataframe as dd
pd.options.mode.chained_assignment = None
import functools
import operator

# import logger
log = logging.getLogger(__name__)

class FTM():
    
    def __init__(self, fasta_df, encoded_df, 
                 mutant_fasta, prefix,
                 coverage_threshold, max_hamming_dist,
                 output_dir, diversity_threshold,
                 hamming_weight, ftm_HD0_only, cpus, client):
        self.fasta_df = fasta_df
        self.encoded_df = encoded_df
        self.file_prefix = prefix
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
        
    @jit(parallel=True)
    def create_fastaDF(self, input_fasta, input_vcf):
        print("Merging any input mutations with reference info.")

        # if supervised, add mutants to fasta_df
        if len(input_vcf) > 0:
            fasta_df = pd.concat([input_fasta, input_vcf], axis=0, sort=True)
        # otherwise run FTM without mutants
        else:
            fasta_df = input_fasta
            
        fasta_df = fasta_df.reset_index(drop=True)
        
        return fasta_df
    
    @jit(parallel=True)
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
            ngram_list.append([size, ngram])
         
        return ngram_list

    def parse_ngrams(self, fasta_df):
        print("Parsing reference ngrams.")
        # unravel list to dataframe
        ngram_df = fasta_df.groupby(["id", "region", "chrom", "start"])\
           .apply(lambda x: pd.DataFrame(x['ngram_list'].tolist()[0], columns=['t_length','ngrams']))\
           .reset_index(drop=False).reset_index(drop=True)
        ngram_df = ngram_df.drop("level_4", axis=1)
        
        return ngram_df
    
    def split_ngrams(self, row):  
        
        # match edge grams with position
        ngram_list = [(row.id, 
                row.region,
                row.chrom,
                row.t_length,
                str(int(row.start) + i), 
                c) for i, c in enumerate(row.ngrams)]

        return ngram_list
    
    def final_ngrams(self, ngram_df):
        
        # break apart dataframe into one row per tuple
        final_dd = self.client.compute(dd.from_pandas(ngram_df.apply(lambda x: \
                pd.Series(x[0]),axis=1).stack().reset_index(level=1, drop=True),
                npartitions=self.cpus,
                sort=False))
        
        final_df =  pd.DataFrame(self.client.gather(final_dd))                                                          
        final_df.columns = ["tups"]
                      
        # break apart rows of tuples to final df
        final_df[['id', 'region', 'chrom', 't_length', 'pos', 'ngram']] = final_df['tups'].apply(pd.Series)
        final_df = final_df.drop("tups", axis=1).reset_index(drop=True)
          
        # group together by region and drop duplicates to avoid double counting
        final_df = final_df.drop_duplicates(subset=["chrom", "pos", "ngram"]).reset_index(drop=True)
        
        return final_df
    
    def calc_hamDist(self, ngram_df, encoded_df):
        print("Calculating hamming distance.")
        '''
        purpose: match ngrams to targets 
        input: ngram_df craeted in ftm.parse_ngrams()
        output: hamming_list of tuples (ref, target match, hamming distance)
        '''
        
        hamming_list = []
        
        for size in self.kmer_lengths:
            
            # subset dataframes for size
            ngrams = ngram_df[ngram_df.t_length == size]
            targets = encoded_df[encoded_df.bc_length == size]
            
            # get unique set from each list
            ngram_list = list(set(ngrams.ngram))
            target_list = list(set(targets.Target.compute(scheduler='threads',
                                                   num_workers=self.cpus)))
 
            hd = lambda x, y, z: cpy.calc_hamming(x,y,z, self.max_hamming_dist)
            hd_list = [hd(x, y, size) for y in target_list for x in ngram_list]
            hamming_list.append(hd_list)
         
        # check that calls were found
        if len(hamming_list) < 1:
            raise SystemExit("No calls below hamming distance threshold.")
        else:

            # flatten list of lists in case of more than one kmer length
            hams = functools.reduce(operator.iconcat, hamming_list, [])

            # remove any tuples that contain "X" for hamming distance
            hamming_list = [i for i in hams if i[2] != "X"]

            return hamming_list
        

    def parse_hamming(self, hamming_list, ngram_df):
        print("Parsing hamming dataframe.")
        '''
        purpose: combine hamming distance information with feature level data
        input: hamming_list: created in match_targets(), list of all ngrams, basecalls and hd
            ngram_df: specifies which gene and position a target came from
            encoded_df: provided feature level information
        output: hamming_df with featureID, gene, group, target, pos
        '''
        
        # create dataframe from list of matches
        hamming_df = pd.DataFrame(hamming_list, columns=["ref_match", "bc_feature", "hamming"])
        hamming_df = dd.from_pandas(hamming_df, 
                                    sort=False,
                                    npartitions=self.cpus)
        hamming_df = hamming_df.drop_duplicates()
        hamming_df = hamming_df.set_index("ref_match")
        hamming_df = self.client.persist(hamming_df)
        assert len(hamming_df) != 0, "No matches were found below hamming distance threshold."

        # match ngrams to their matching Target regions
        ngram_df = ngram_df.set_index("ngram")
        hamming_df = dd.merge(hamming_df, ngram_df, 
                            how="left")
        
        # get rid of old index, copy bc_feature to retain and set new index
        # for speedier merging
        hamming_df = hamming_df.reset_index(drop=True)
        hamming_df["Target"] = hamming_df["bc_feature"]
        
        # match basecalls to their matching features
        encoded_df = self.client.persist(self.encoded_df)
        
        hamming_df = dd.merge(hamming_df, encoded_df, 
                              how='left',
                              on='Target')
        hamming_df = hamming_df.reset_index(drop=True)
        
        # reorder columns  and subset
        hamming_df = hamming_df[["FeatureID", "id", "region",  
                                 "chrom", "pos", "Target", 
                                 "BC", "cycle", "pool", "hamming"]]
        
        # save raw hamming counts to file
        outfile = os.path.join(self.output_dir, self.file_prefix + "_rawCounts")
        if os.path.exists(outfile):
            shutil.rmtree(outfile, ignore_errors=True)
        hamming_df.to_parquet(outfile, 
                              engine='fastparquet',
                              append=False)
        
        return hamming_df
    
    @jit(parallel=True)
    def parse_hd1(self, hamming_df):
        print("Parsing imperfect matches.")
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

    @jit(parallel=True)
    def diversity_filter(self, input_df, diversity_threshold):
        print("Filtering results by diversity threshold.")
        '''
        purpose: filters out Targets for a feature that are below self.diversity_threshold
        input: matches_df built in align.match_perfects()
        output: diversified dataframe filtered for only Targets per feature that meet threshold
        '''   

        # get unique barcode counts
        uniques = input_df[["FeatureID", "region", "Target"]].drop_duplicates()
        a = uniques.groupby(["FeatureID", "region"])["Target"].count().to_frame("feature_div")

        # join results with main dataframe as feature_divv
        input_df = input_df.join(a, on=["FeatureID", "region"],
                                 npartitions=self.cpus)
        
        # filter out features below diversity threshold
        diversified = input_df[input_df.feature_div >= diversity_threshold]
        undiversified = input_df[input_df.feature_div < diversity_threshold]

        if len(undiversified)> 1:
            undiversified = undiversified[["FeatureID", "BC", "cycle", "pool"]]
            undiversified["invalid_reason"] = "div_filter"

        # check that calls were found
        if len(diversified) > 1:
            return diversified, undiversified
        else:
            raise SystemExit("No calls pass diversity threshold.")
    
    @jit(parallel=True)
    def locate_multiMapped(self, diversified):
        print("Normalizing multi-mapped barcodes.")
        '''
        purpose: normalize multi-mapped reads as total count in gene/total count all genes
        input: feature_div filtered match dataframe build in align.diversity_filter()
        output: counts normalized counts dataframe to pass to algin.score_matches()
            raw counts saved to tsv file
        '''
        
        # find multi-mapped barcodes
        positions = diversified[["FeatureID", "region", "Target", "pos"]].drop_duplicates()
        a = positions.groupby(["FeatureID", "region", "Target"])["pos"].count().to_frame("multi")
        
        # count multi-mapped reads
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
    
    @jit(parallel=True)
    def barcode_counts(self, counts):
        print("Counting barcodes.")
        '''
        purpose: sum counts for each barcode per position
        input: counts table from locate_MultiMapped
        output: bc_counts of counts aggregated by barcode to each region
        '''
        
        # sum counts per feature/region/pos
        a = counts.groupby(["FeatureID", "Target", "region", "pos"])["counts"].sum().to_frame("bc_count")
        counts = counts.join(a, on=["FeatureID", "Target", "region", "pos"],
                                 npartitions=self.cpus)
        bc_counts = counts.drop(["counts"],axis=1)
        
        # information will be duplicated for each row, drop dups   
        bc_counts = bc_counts.drop_duplicates(subset=["FeatureID", "region", "Target", "pos"])

        return bc_counts
    
    @jit(parallel=True)  
    def regional_counts(self, bc_counts):
        print("Compiling counts for each region.")
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
        regional_counts = bc_counts.drop_duplicates(subset=["FeatureID", "region"])

        # store regions below covg threshold
        below_covg = regional_counts[regional_counts.counts < self.coverage_threshold]
        if len(below_covg) > 1:
            below_covg = below_covg[["FeatureID", "BC", "cycle", "pool"]]
            below_covg["invalid_reason"] = "covg_filter"
        
        # filter out featureID/gene combos below covg threshold
        regional_counts = regional_counts[["FeatureID", "region", "feature_div", "counts"]]
        regional_counts = regional_counts[regional_counts.counts >= self.coverage_threshold]
        
        assert len(regional_counts) != 0, "No matches found above coverage threshold." 
        
        return regional_counts, below_covg
    
    def get_top2(self, regional_counts):
        print("Pulling out top two regions for each molecule.")
        '''
        purpose: locate top 2 target counts for perfect matches
        input: perfects built in ftm.sum_counts()
        output: hamming_df= calls within hamming threshold
            perfect matches with columns denoting max and second max counts
        '''
        
        top2 = regional_counts.groupby('FeatureID')[['FeatureID', 'region', 'feature_div', 'counts']]\
            .apply(lambda x: x.nlargest(2, columns=['counts'])).reset_index(drop=True)
        top2.index.name = ''                

        return top2
                
    @jit(parallel=True)         
    def find_multis(self, tops):
        print("Finding ties.")
        '''
        purpose: located features that have tied counts for best FTM call
        input: count_df created in ftm.sum_counts.py
        output: dataframe of ties and non-tied counts for passing to return_ftm
        '''
        
        # get group size for each feature
        a = tops.groupby(["FeatureID"]).count().reset_index()
        a = a[["FeatureID", "counts"]]
        a.columns = ["FeatureID", "grp_size"]
        a = a.drop_duplicates()
        
        # add counts back to tops df
        tops = dd.merge(tops, a, on=["FeatureID"],
                          how='left',
                          npartitions=self.cpus)
        
        # separate ties to process
        singles = tops[tops.grp_size == 1]
        multis = tops[tops.grp_size > 1]

        # get rid of grp_size column
        singles = singles.drop(["grp_size"], axis=1)
        multis = multis.drop(["grp_size"], axis=1)
    
        return singles, multis
    
    @jit(parallel=True)
    def calc_symDiff(self, group, bc_counts2):
        print("Calculating symmetrical difference.")
        
        # subset bc_counts to only features with FTM calls
        # subset to relevant rows and features for speed 
        target_df = bc_counts2[["FeatureID", "region", "Target"]].merge(group[["FeatureID","region"]])

        # convert target list to sets for quick comparison
        target_df = target_df.groupby(["FeatureID", "region"])["Target"].apply(set).reset_index()
        target_df = target_df.compute(n_workers=self.cpus*2, 
                             threads_per_worker=self.cpus)
        # calculate Targets unique to each region of interest
        target_df["sym_diff"] = cpy.calc_symmetricDiff(target_df)

        target_df = target_df.reset_index(drop=True)
        
        return target_df
    
    @jit(parallel=True)   
    def decision_tree(self, x, bc_counts2):

        '''
        purpose: check coverage. If one has more than 3x coverage, keep. 
            Else check feature diversity scores, if one score is 2x or greater, keep
            else, keep score with highest sym_diff
        input: multi_df
            x = feature's top two rows
        output: used in return_ftm to return ties that pass this filtering logic
        '''

        second_div = x.feature_div.min()
        result = x[x.feature_div > second_div]

        # if there is only one result now, return it
        if len(result) == 1:
            return result

        # otherwise no call can be made
        else:
            pass

        ### tie-breaking logic deprecated ###
        
        # # find second highest count
        # second_max = x.counts.min()
        #
        # # check if result with max count >= 3x second count
        # result = x[x.counts > (3 * second_max)]
        #
        # # if there is only one result, return it
        # if len(result) == 1:
        #     return result
        # # otherwise check feature div
        # else:
        #     second_div = x.feature_div.min()
        #     result = x[x.feature_div > (2 * second_div)]
        #
        #     # if there is only one result now, return it
        #     if len(result) == 1:
        #         return result
        #
        #     # otherwise check symmetrical difference
        #     else:
        #
        #         # pull out targets for each region
        #         group = dd.merge(x, bc_counts2,
        #                          on=["FeatureID", "region"],
        #                          how='left').drop_duplicates()
        #         group = self.client.compute(group)
        #         group = self.client.gather(group)
        #
        #         # create a list of targest for each region
        #         target_df = group.groupby(["FeatureID", "region"])["Target"].apply(set).reset_index()
        #
        #         # calculate Targets unique to each region of interest
        #         target_df["sym_diff"] = cpy.calc_symmetricDiff(target_df)
        #
        #         target_df = target_df.reset_index(drop=True)
        #
        #         # take top sym_diff score for group
        #         max_symDiff = target_df.sym_diff.max()
        #         result = target_df[target_df.sym_diff == max_symDiff]
        #
        #         if len(result) == 1:
        #
        #             result = result.drop(["sym_diff", "Target"], axis=1)
        #             result = x.merge(result,
        #                         on=["FeatureID", "region"])
        #         else:
        #             pass
    
    @jit(parallel=True)
    def process_multis(self, multi_df, bc_counts2):
        
        ## need to figure out how to process this in dask ##
        multis = self.client.compute(multi_df)
        multi_df = self.client.gather(multis)
        
        multi_df = multi_df.groupby("FeatureID").apply(self.decision_tree,
            bc_counts2[["FeatureID", "Target", "region"]]).reset_index(drop=True)
        return multi_df
    
    @jit(parallel=True)
    def return_calls(self, ftm_df, hamming_df):
        print("Processing FTM calls.")
        '''
        purpose: save features where no ftm call can be made to a file
            subset bc_counts to contain only features where an FTM call can be made
        input: ftm_df and encoded_df
        output: no_calls = featureID's and basecalls for features that were not able to 
            be assigned an FTM call to be run through hamming.py
            perfect_calls = bc_counts of per feature/target counts for ftm called features
        '''
        
        # store information on no calls 
        calls = dd.merge(hamming_df, ftm_df, on=['FeatureID', 'region'], 
                   how='left', indicator=True)
        
        # pull out only calls related to FTM called region for HD under threshold   
        all_calls = calls[calls._merge == "both"].drop(["counts", "id",
                                                                  "feature_div",
                                                                  "_merge"],
                                                                 axis=1).drop_duplicates()
        if len(all_calls) == 0:
            raise SystemExit("No FTM calls were made.")     
                                                            
        no_calls = calls[calls._merge == "left_only"].drop(["counts", 
                                                            "feature_div",
                                                            "_merge"],
                                                            axis=1).drop_duplicates()
                  
        if len(no_calls) > 0:
           
            no_calls = no_calls[["FeatureID", "BC", "cycle", "pool"]]
            no_calls["invalid_reason"] = "no_ftm_call"

        return all_calls, no_calls
    
    @jit(parallel=True)
    def filter_allCalls(self, all_calls):
        print("Adding in HD1+.")   
        # normalize multi-mapped reads and count
        all_norm = self.locate_multiMapped(all_calls)
          
        # group counts together for each bar code
        all_calls = self.barcode_counts(all_norm)
        all_calls = all_calls.repartition(npartitions=self.cpus)
        
        counts_file = os.path.join(self.output_dir, self.file_prefix + "_all_counts")
        if os.path.exists(counts_file):
            shutil.rmtree(counts_file, ignore_errors=True)
        all_calls.to_parquet(counts_file, 
                             append=False,
                             engine='fastparquet')

        return all_calls
    
    @jit(parallel=True)
    def main(self):
        
        # create fasta dataframe and combine with mutations if provided
        fasta_df = self.create_fastaDF(self.fasta_df, self.mutant_fasta)
                    
        # break reference seq into kmers
        fasta_df["ngram_list"] = fasta_df.apply(self.create_ngrams, 
                              axis=1)
              
        # unravel dataframe with lists of ngrams
        ngram_df = self.parse_ngrams(fasta_df)
                   
        # split ngrams and match with position
        ngram_df = pd.DataFrame(ngram_df.apply(self.split_ngrams, 
                                               axis=1))           
        # parse the final ngram df
        ngrams = self.final_ngrams(ngram_df)
       
        # calculate hamming distance between input ref and features
        hamming_list = self.calc_hamDist(ngrams, self.encoded_df)
              
        # convert hamming_list into dataframe and parse
        hamming_df = self.parse_hamming(hamming_list, ngrams)
                    
        # separate perfect matches from hd1+
        hd_plus, hd0 = self.parse_hd1(hamming_df)
        
        # filter for feature diversity
        hd0_diversified, undiversified = self.diversity_filter(hd0, self.diversity_threshold)
             
        # normalize multi-mapped reads and count
        norm_counts = self.locate_multiMapped(hd0_diversified)
             
        # group counts together for each bar code
        bc_counts = self.barcode_counts(norm_counts)
              
        # save basecall normalized counts to file
        # save raw hamming counts to file and remove from memory
        bc_out = os.path.join(self.output_dir, self.file_prefix + "_bc_counts")
        if os.path.exists(bc_out):
            shutil.rmtree(bc_out, ignore_errors=True) 
        bc_counts.to_parquet(bc_out, 
                             append=False,
                             engine='fastparquet')
        bc_counts2 = bc_counts.drop(['pool', 'cycle', 'BC'],axis=1)  
           
        # sum counts for each region
        region_counts, below_covg = self.regional_counts(bc_counts)
      
        # find top 2 scores for each bar code
        top2 = self.get_top2(region_counts)    
      
        singles, multis = self.find_multis(top2)
        
        multi_df = self.process_multis(multis, bc_counts2)
        
        # concat results for df's with results
        single_length = len(singles)
        multi_length = len(multi_df)
        if (single_length > 0) and (multi_length > 0):
            ftm_counts = dd.concat([singles, multi_df])
        elif single_length > 0:
            ftm_counts = singles
            ftm_counts = dd.from_pandas(ftm_counts,
                                    sort=False,
                                    npartitions=self.cpus)
        elif multi_length > 0:
            ftm_counts = multi_df
            ftm_counts = dd.from_pandas(ftm_counts,
                                    sort=False,
                                    npartitions=self.cpus)
        else:
            raise SystemExit("No FTM calls can be made on this dataset.")
        
        # output ftm to file
        ftm_file = os.path.join(self.output_dir, self.file_prefix + "_ftm_calls/")
        # remove any existing directories
        if os.path.exists(ftm_file):
            shutil.rmtree(ftm_file, ignore_errors=True)
        ftm_counts.to_parquet(ftm_file, 
                              append=False,
                              engine='fastparquet')

        # save no_calls to a file and add HD1+ back in for sequencing
        all_calls, no_calls = self.return_calls(ftm_counts, hamming_df)
        
        # filter all counts 
        all_counts = self.filter_allCalls(all_calls)

        # find invalid barcodes
        invalid_dfs = [undiversified, below_covg, no_calls]
        concat_list = []
        for df in invalid_dfs:
            if len(df) > 1:
                concat_list.append(df)

        # concatenate invalids
        if len(concat_list) > 1:
            invalid_df = dd.concat(concat_list)
        elif len(concat_list) == 1:
            invalid_df = concat_list[0]
        else:
            invalid_df = ""
        
        return all_counts, invalid_df

        
       


        
        