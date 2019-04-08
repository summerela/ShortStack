#!/usr/bin/env python

'''
Main script to run ShortStack

Dependencies:
biopython==1.72
configparser==3.5.0
Cython==0.28.2
dask==0.19.1
fastparquet
ipywidgets==7.4.2
networkx==2.1
numba==0.38.1
numpy==1.14.2
pandas==0.22.0
pandasql==0.7.3
psutil==5.4.6
pytest==4.0.2
scikit-allel==1.1.10
seqlog==0.3.9
swifter==0.223
ujson==1.35



Required Input: 
- s6 file
    must be in JSON format
- encoding.txt 
    must be tsv
    must contain following columns
    - PoolID: int specifying which pool barcode was run in
    - Target: string of target nucleotides for each input probe
    - BC: matching base call for target
    - bc_length: int specifying length of target to minimize downstream length calc
 
- target.fa
    fasta header must be colon delimited
    header must contain columns as follows
    id:chrom:start:stop:build:strand(+ or -)

Optional Input:
- predefined mutations in vcf4.3 format

To run, type the following from the command line: 
python run_shortstack.py -c path/to/config.txt

'''
#### import packages required to load modules ####
import logging, logging.config     # required for seqlog formatting
import sys, os, time, psutil, gc, dask, glob, shutil
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore") # ignore dask connection warnings 
from logging.handlers import RotatingFileHandler    # set max log size before rollover
from logging.handlers import TimedRotatingFileHandler    # set time limit of 31 days on log files
import configparser as cp   # parse config files 
import argparse    #parse user arguments
import pandas as pd
from numba import jit
import pyximport; pyximport.install()
from dask.distributed import Client
import dask.dataframe as dd
pd.options.mode.chained_assignment = None

# set runtime options
dask.config.set(shuffle='tasks')

# import shortstack modules
import parse_input 
import encoder
import parse_mutations as mut
import ftm
import sequencer as seq

# get path to config file
parser = argparse.ArgumentParser(description='Run ShortStack program.')
parser.add_argument('-c','--config', help='Path to shortstack config file',required=True)
args = parser.parse_args()

class ShortStack():
    
    def __init__(self, 
                 config_path, 
                 input_s6, 
                 target_fa, 
                 mutation_vcf=None,
                 output_dir=os.path.curdir,
                 encoding_table="./encoding.txt",  
                 log_path = "./",
                 qc_threshold=7,
                 diversity_threshold=2,
                 covg_threshold=2,
                 max_hamming_dist=1,
                 hamming_weight=1,
                 ftm_HD0_only=True,
                 ftm_only=False):
        
        # gather run options
        self.today = time.strftime("%Y%m%d")
        self.qc_threshold = int(qc_threshold)
        self.covg_threshold = int(covg_threshold)
        self.diversity_threshold = int(diversity_threshold)
        self.max_hamming_dist = int(max_hamming_dist)
        self.hamming_weight = int(hamming_weight)
        self.ftm_HD0_only = ftm_HD0_only
        self.ftm_only = ftm_only
        self.mem_limit = psutil.virtual_memory().available - 1000
        self.cpus = int(psutil.cpu_count())
        self.client = Client(name="HexSembler",
                             memory_limit=self.mem_limit,
                             n_workers=self.cpus-2, 
                             threads_per_worker=self.cpus/2,
                             processes=True)

        # initialize file paths and output dirs
        self.encoding_file = os.path.abspath(encoding_table)
        self.log_path = os.path.abspath(log_path)
        self.output_dir =  os.path.join(output_dir, "output")
        self.config_path = os.path.abspath(config_path)

        # create output dir if not exists
        self.create_outdir(self.output_dir)
        
        # gather input file locations
        self.input_s6 = os.path.abspath(input_s6)
        self.target_fa = os.path.abspath(target_fa)
        self.mutation_vcf = mutation_vcf
        
        if self.mutation_vcf.lower() != "none": 
            self.mutation_vcf = os.path.abspath(mutation_vcf)
            self.file_check(self.mutation_vcf)
        else:
            self.mutation_vcf = "none"
        
        ### Setup logging ###
        now = time.strftime("%Y%d%m_%H:%M:%S")
        today_file = "{}_HexSembler.log".format(self.today)
        log_file = os.path.join(self.log_path, today_file)
        FORMAT = '{"@t":%(asctime)s, "@l":%(levelname)s, "@ln":%(lineno)s, "@f":%(funcName)s}, "@mt":%(message)s'
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w',
                            format=FORMAT)
        
        self.log = logging.getLogger(__name__)
        
        ## setup rotating log handlers
        size_handler = RotatingFileHandler(
                      log_file, maxBytes=10000000 , backupCount=10)
        
        time_handler = TimedRotatingFileHandler(
                       log_file, when='D', interval=31, backupCount=10
            )
        # add handlers for size and time limits
        self.log.addHandler(size_handler)
        self.log.addHandler(time_handler)    
        
        ###Log input config file options
        run_string = ''' \n ShortStack Run: {now}
        ***Input*** \n
        Input S6 file: {s6} \n
        Target fasta: {fasta} \n
        Encoding table: {colors} \n
        Predefined Mutations: {mutations}\n
        Configuration file: {config} \n
        
        ***Parameters*** \n
        Min Image QC Score: {qc_thresh} \n
        Minimum Feature Diversity: {div_thresh} \n
        Minimum Coverage: {min_cov} \n
        Max Hamming Distance: {ham_dist} \n
        Hamming Weight: {ham_weight}\n
        FTM with HD0 only: {ftm_perfects}\n
        
        ***Results*** \n
        Results output to: {output}\n
        '''.format(now=now, s6=self.input_s6,  
                   fasta=self.target_fa, 
                   colors=self.encoding_file, 
                   mutations=self.mutation_vcf,
                   qc_thresh = self.qc_threshold,
                   div_thresh = self.diversity_threshold,
                   config=self.config_path, 
                   output=self.output_dir,
                   min_cov=self.covg_threshold,
                   ham_dist=self.max_hamming_dist,
                   ham_weight=self.hamming_weight,
                   ftm_perfects = self.ftm_HD0_only)

        # write run info to log
        self.log.info(run_string)

    @staticmethod
    def create_outdir(output_dir):
        '''
        Check if a directory exists, and if not, create it
        :param output_dir: path to directory
        :return: directory created
        '''
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except AssertionError as e:
            error_message = "Unable to create output dir at: {}".format(output_dir, e)
            log.error(error_message)
            raise SystemExit(error_msg)
    
    @jit(parallel=True) 
    def file_check(self, input_file):
        '''
        Purpose: check that input file paths exist and are not empty
        input: file path
        output: assertion error if file not found or is empty
        '''
        error_message = "Check that {} exists and is not empty.".format(input_file)
        input_file = os.path.abspath(input_file)
        assert (os.path.isfile(input_file)) and (os.path.getsize(input_file) > 0), error_message
    
    def main(self):
        '''
        purpose: main function to run shortstack
        Runs in the following order: 
        - parse_input.py: parse input files
        - encoder.py: match basecalls with sequences from encoding file
        - align.py: run first round of FTM
        
        '''
        print("Parsing input...\n")
        # check that file paths are valid
        self.file_check(self.encoding_file)
        self.file_check(self.target_fa)


        #########################
        ####   Parse Input   ####
        #########################
        # instantiate parsing class from parse_input.py
        parse = parse_input.Parse_files(self.output_dir,
                                        self.target_fa,
                                        self.mutation_vcf,
                                        self.encoding_file,
                                        self.cpus,
                                        self.client)

        mutation_df, fasta_df, encoding_df = parse.main_parser()

        # read in s6 parquet
        glob_path = '{}/*.parquet'.format(self.input_s6)
        files = glob.glob(glob_path)
        if len(files) < 1:
            raise SystemExit("Check path to parquet S6 directory.")
        s6_df = dd.read_parquet(files)

        ########################
        ####   Encode S6    ####
        ########################
        print("Encoding targets...\n")
        # instantiate encoder class from encoder.py
        encode = encoder.Encode_files(s6_df,
                                      encoding_df,
                                      self.output_dir,
                                      self.cpus,
                                      self.client)

        # return dataframe of targets found for each molecule
        encoded_df, parity_df = encode.main(encoding_df, s6_df)

        # cleanup encoding_df
        del encoding_df
        gc.collect()

        ###################################
        ####   Assemble Mutations    #####
        ###################################
        ## Supervised mode only ##
        # if mutations are provided, assemble mutation seqs from mutation_vcf
        if self.mutation_vcf != "none":
            print("Assembling mutations...\n")
            self.log.info("Mutations assembled from:\n {}".format(self.mutation_vcf))
            # instantiate aligner module
            mutations = mut.AssembleMutations(fasta_df,
                                          mutation_df,
                                          s6_df)
            # add mutated reference sequences to fasta_df
            mutant_fasta = mutations.main()
        # no mutations provided = unsupervised mode and mutant_fasta is empty
        else:
            mut_message = "No mutations provided."
            self.log.info(mut_message)
            mutant_fasta = pd.DataFrame()

        ###############
        ###   FTM   ###
        ###############
        print("Running FTM...\n")

        # instantiate FTM module from ftm.py
        run_ftm = ftm.FTM(fasta_df,
                              encoded_df,
                              mutant_fasta,
                              self.covg_threshold,
                              self.max_hamming_dist,
                              self.output_dir,
                              self.diversity_threshold,
                              self.hamming_weight,
                              self.ftm_HD0_only,
                              self.cpus,
                              self.client
                              )
        # run FTM
        all_counts, hamming_df, ftm_calls = run_ftm.main()
        all_counts_dir = os.path.join(self.output_dir, "all_counts")
        if os.path.exists(all_counts_dir):
            shutil.rmtree(all_counts_dir, ignore_errors=True)
        all_counts.to_parquet(all_counts_dir,
                              append=False,
                              engine="fastparquet")

        # cleanup
        del encoded_df, mutant_fasta
        gc.collect()
#
#         #############################
#         ###   valid off targets   ###
#         #############################
# #         # save valid barcodes that are off target
# #         print("Saving off-target barcode information...\n")
# #
# #         @jit(parallel=True)
# #         def save_validOffTarget(s6_df, parity_df, hamming_df, ftm_calls):
# #
# #             # filter s6 for valid 3spotters not found in encoding file
# #             no_invalids = dd.merge(s6_df, parity_df.drop_duplicates(),
# #                                 on=['FeatureID','BC', 'pool', 'cycle'],
# #                                 how='left', indicator=True)
# #
# #             # pull out feature id's/basecalls that are only in s6_df and not in invalids
# #             no_invalids = no_invalids[no_invalids._merge == "left_only"]
# #             no_invalids = no_invalids.drop(["_merge", "idx"], axis=1)
# #
# #
# #             # prep hamming_df for merge
# #             hamming_df = hamming_df[["FeatureID", "BC", "cycle", "pool"]]
# #             hamming_df = hamming_df.drop_duplicates().reset_index(drop=True)
# #
# #             # pull out featureID/BC that are in s6 but not in hamming=not hitting targets
# #             valid_offTargets = dd.merge(no_invalids, hamming_df,
# #                        how='left', indicator=True)
# #
# #             valid_offTargets = valid_offTargets[valid_offTargets._merge == "left_only"]
# #             valid_offTargets = valid_offTargets.drop(["_merge"], axis=1)
# #
# #
# #             # pull out features with ftm votes
# #             final_offTargets = dd.merge(valid_offTargets,
# #                                         ftm_calls[["FeatureID"]],
# #                                         on=["FeatureID"],
# #                                         how='left',
# #                                         indicator=True)
# #             final_offTargets = final_offTargets[final_offTargets._merge == "left_only"]
# #             final_offTargets = final_offTargets.drop(["_merge"], axis=1)
# #             raise SystemExit(final_offTargets.head())
# #             # compute data frame
# #             final_offTargets = self.client.compute(final_offTargets)
# #             final_offTargets = self.client.gather(final_offTargets)
# #
# #             # sort output
# #             final_offTargets = final_offTargets.sort_values(by=["FeatureID", "pool",
# #                                                                 "cycle", "BC"])
# #
# #             # save to file
# #             valids_off_out = os.path.join(self.output_dir, "valid_offTargets.tsv")
# #             final_offTargets.to_csv(valids_off_out, sep="\t", index=False)
# #
# #         save_validOffTarget(s6_df, parity_df, hamming_df, ftm_calls)
# #
# #         # clean up
# #         del parity_df, s6_df, hamming_df
# #         gc.collect()
# #
# #         # check for ftm_only option
# #         if self.ftm_only:
# #             ftm_message = "FTM complete. Update ftm_only to run sequencing and consensus pipelines."
# #             raise SystemExit(ftm_message)
# #         else:
# #             print("FTM complete. Sequencing results...\n")
# #
        # ####################
        ###   Sequence   ###
        ####################
        print("Sequencing...\n")
        # instantiate molecule sequencing module from sequencer.py
        sequence = seq.Sequencer(all_counts,
                                 fasta_df,
                                 self.output_dir,
                                 self.cpus,
                                 self.client)

        molecule_df = sequence.main()

        # saving output for consensus sequencing
        fasta_out = os.path.join(self.output_dir, "molecule_info/fasta")
        fasta_df.to_parquet(fasta_out, append=False, engine="fastparquet", compression='snappy')

        molecule_out = os.path.join(self.output_dir, "molecule_info/molecules")
        molecule_df.to_parquet(molecule_out, append=False, engine="fastparquet", compression='snappy')

        print('Molecule parquet files saved to {}/molecule_info/molecules'.format(self.output_dir))
        print('Fasta_df parquet files saved to {}/molecule_info/fasta'.format(self.output_dir))

        # close dask tasks
        self.client.close()

        # cleanup temp files and directories
        dask_folder = os.path.join(os.path.curdir, "dask-worker-space")
        shutil.rmtree(dask_folder)
        
        
if __name__ == "__main__":
    
    # parse config file with default options
    config = cp.ConfigParser()
    configFilePath = args.config
    config.read(configFilePath)

    sStack = ShortStack(config_path=args.config,
                output_dir=config.get("user_facing_options","output_dir"),
                input_s6=config.get("user_facing_options", "input_s6"),
                target_fa=config.get("user_facing_options","target_fa"),
                mutation_vcf=config.get("user_facing_options", "mutation_vcf"),
                encoding_table=config.get("user_facing_options", "encoding_table"),
                log_path=config.get("internal_options", "logFile_path"),
                covg_threshold=config.getint("internal_options","covg_threshold"),
                qc_threshold=config.getint("internal_options","qc_threshold"),
                diversity_threshold=config.getint("internal_options", "diversity_threshold"),
                max_hamming_dist=config.getint("internal_options", "max_hamming_dist"),
                hamming_weight=config.getint("internal_options", "hamming_weight"),
                ftm_HD0_only=config.getboolean("internal_options", "ftm_HD0_only"),
                ftm_only=config.getboolean("user_facing_options", "ftm_only"))

    sStack.main()
