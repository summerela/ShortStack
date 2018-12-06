
#!/usr/bin/env python

'''
Main script to run ShortStack

Dependencies:
- python 3.6
- matplotlib
- pandas
- scipy
- numpy
- ipywidgets
- numba
- swifter
- cython

Required Input: 
- s6 file
    must be in JSON format
- barcode.txt 
    must be tsv
    must contain following columns
    PoolID  Target  BC
 
- target.fa
    fasta header must be colon delimited
    header must contain columns as follows
    id:chrom:start:stop:build

Optional Input:
- predefined mutations in vcf4.3 format

To run: 
python run_shortstack.py -c path/to/config.txt

'''

#### import packages required to load modules ####
import logging    # create log file
import logging.config     # required for seqlog formatting
import os    # traverse/operations on directories and files
import time    # time/date stamp for log
from logging.handlers import RotatingFileHandler    # set max log size before rollover
from logging.handlers import TimedRotatingFileHandler    # set time limit of 31 days on log files

### Setup logging ###
today = time.strftime("%Y%m%d")
now = time.strftime("%Y%d%m_%H:%M:%S")

log_file = "{}_ShortStack.log".format(today)
FORMAT = '{"@t":%(asctime)s, "@mt":%(message)s,"@l":%(levelname)s, "@f":%(funcName)s}, "@ln":%(lineno)s'
logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w',
                    format=FORMAT)
log = logging.getLogger(__name__)

## setup rotating log handlers
size_handler = RotatingFileHandler(
              log_file, maxBytes=10000000 , backupCount=10)

time_handler = TimedRotatingFileHandler(
               log_file, when='D', interval=31, backupCount=10
    )
# add handlers for size and time limits
log.addHandler(size_handler)
log.addHandler(time_handler)

# import remaining packages
import configparser as cp   # parse config files 
import argparse    #parse user arguments
import pandas as pd
import numpy as np
import psutil
import concurrent.futures as cf
from numba import jit

# import shortstack modules
import parse_input 
import encoder
import parse_mutations as mut
import ftm
import sequencer as seq

import pyximport; pyximport.install()

# get path to config file
parser = argparse.ArgumentParser(description='Run ShortStack program.')
parser.add_argument('-c','--config', help='Path to shortstack config file',required=True)
args = parser.parse_args()

class ShortStack():
    
    def __init__(self, 
                 config_path, 
                 input_s6, 
                 target_fa, 
                 mutation_vcf='none',
                 output_dir=os.getcwd(),
                 encoding_table="./base_files/encoding.txt",  
                 kmer_length=6, 
                 qc_threshold=7,
                 num_cores=6,
                 diversity_threshold=2,
                 covg_threshold=2,
                 max_hamming_dist=1,
                 hamming_weight=1,
                 all_fov=True):
        
        # gather run options
        self.kmer_length = int(kmer_length)
        self.qc_threshold = int(qc_threshold)
        self.num_cores= int(psutil.cpu_count() - 3) #leave at least 3 cores
        self.encoding_file = encoding_table
        self.covg_threshold = int(covg_threshold)
        self.diversity_threshold = int(diversity_threshold)
        self.max_hamming_dist = int(max_hamming_dist)
        self.hamming_weight = int(hamming_weight)
        self.all_fov = all_fov

        
        # initialize file paths and output dirs
        self.output_dir = "{}/output/".format(output_dir)
        self.running_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.abspath(config_path)
        
        # create output dir if not exists
        self.create_outdir(self.output_dir)
        self.qc_out_file = "{}/imageQC_filtered.tsv".format(self.output_dir)
        self.run_info_file = "{}/run_info.txt".format(self.output_dir)
        self.base_dir = "{rd}/base_files/".format(rd=self.running_dir)
            
        # gather input file locations
        self.input_s6 = os.path.abspath(input_s6)
        self.target_fa = os.path.abspath(target_fa)
        
        if mutation_vcf.lower() != "none": 
            self.mutation_vcf = os.path.abspath(mutation_vcf)
        else:
            # temporarily disabling unsupervised mode
            raise SystemExit("\nUnsupervised mode not yet implemented. Please provide VCF file to continue.\n")      
        
        ###Log input config file options
        run_string = ''' \n ShortStack Run: {now}
        ***Input*** \n
        Input S6 file: {s6} \n
        Target fasta: {fasta} \n
        Encoding table: {colors} \n
        Predefined Mutations: {mutations}\n
        Configuration file: {config} \n
        
        ***Parameters*** \n
        Kmer Length: {kmer_len} \n
        Min Image QC Score: {qc_thresh} \n
        Minimum Feature Diversity: {div_thresh} \n
        Minimum Coverage: {min_cov} \n
        Max Hamming Distance: {ham_dist} \n
        Hamming Weight: {ham_weight}\n
        
        ***Results*** \n
        Results output to: {output}\n
        Run Info and Warnings output to: {qc_file}\n
        QC Stats output to: {qc_stats}\n
        '''.format(now=now, s6=self.input_s6,  
                   fasta=self.target_fa, 
                   colors=self.encoding_file, 
                   mutations=self.mutation_vcf,
                   kmer_len = self.kmer_length,
                   qc_thresh = self.qc_threshold,
                   div_thresh = self.diversity_threshold,
                   config=self.config_path, 
                   output=self.output_dir,
                   qc_file = self.run_info_file,
                   qc_stats=self.qc_out_file,
                   min_cov=self.covg_threshold,
                   ham_dist=self.max_hamming_dist,
                   ham_weight=self.hamming_weight)
        
        print(run_string)
        # write run info to wd/output/run_info.txt
        with open(self.run_info_file, "w+") as f:
            f.writelines(run_string)

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
            
    @staticmethod
    def clean_up(dir, pattern):
        '''
        Clean up temp files and dirs
        :param dir: Directory to clean
        :param pattern: file name or wild-card style file name: ex. slurm*.out
        :return: removal of specified files in dir
        '''
        for file in glob.glob("{}/{}".format(dir, pattern)):
            os.remove(file)
            log.info("Removed temp file: {} \n".format(file))
            
    @staticmethod
    def parallelize_dataframe(self, df, func):
    
        chunks = round(df.shape[0]/10)
        df_split = np.array_split(df, chunks)
        
        with cf.ProcessPoolExecutor(self.num_cores) as pool:
            df = pd.concat(pool.map(func, df_split))
        
        return df
    
    @jit 
    def file_check(self, input_file):
        '''
        Purpose: check that input file paths exist and are not empty
        input: file path
        output: assertion error if file not found or is empty
        '''
        error_message = "Check that {} exists and is not empty.".format(input_file)
        print("Checking {}".format(input_file))
        assert (os.path.isfile(input_file)) and (os.path.getsize(input_file) > 0), error_message

    def main(self):
        '''
        purpose: main function to run shortstack
        Runs in the following order: 
        - parse_input.py: parse input files
        - encoder.py: match basecalls with sequences from encoding file
        - align.py: run first round of FTM
        
        '''
        
        # check that file paths are valid
        self.file_check(self.encoding_file)
        self.file_check(self.input_s6)
        self.file_check(self.target_fa)
        self.file_check(self.mutation_vcf)

        
        #########################
        ####   Parse Input   ####
        #########################
        # instantiate parsing class from parse_input.py 
        parse = parse_input.Parse_files(self.input_s6, 
                                        self.output_dir,
                                        self.target_fa,
                                        self.mutation_vcf, 
                                        self.encoding_file,
                                        self.qc_threshold,
                                        self.all_fov)
         
        s6_df, mutation_df, encoding_df, fasta_df = parse.main_parser()
      
        ########################
        ####   Encode S6    ####
        ########################
        log.info("Reads encoded using file:\n {}".format(self.encoding_file))  
        # instantiate encoder class from encoder.py
        encode = encoder.Encode_files(s6_df, encoding_df, self.output_dir)
        # return dataframe of targets found for each molecule   
        encoded_df = encode.main(encoding_df,  s6_df)


        ###################################
        ####   Assemble Mutations    #####
        ###################################
        ## Supervised mode only ##
        # if mutations are provided, assemble mutation seqs from mutation_vcf
        if self.mutation_vcf != "none":
            print("Assembling input variants.\n")
            log.info("Mutations assembled from:\n {}".format(self.mutation_vcf))
            # instantiate aligner module
            mutations = mut.AssembleMutations(fasta_df,
                                          mutation_df, 
                                          self.run_info_file,
                                          s6_df)  
            # add mutated reference sequences to fasta_df        
            mutant_fasta = mutations.main()
        # no mutations provided = unsupervised mode and mutant_fasta is empty
        else:
            mut_message = "No mutations provided. Entering unsupervised mode."
            print(mut_message)
            log.info(mut_message)
            mutant_fasta = ""
       
        ###############
        ###   FTM   ###
        ###############
        align_message = "Running FTM...\n"
        print(align_message)
        log.info(align_message)
 
        # instantiate FTM module from ftm.py
        run_ftm = ftm.FTM(fasta_df,
                              encoded_df, 
                              mutant_fasta,
                              self.covg_threshold,
                              self.kmer_length,
                              self.max_hamming_dist,
                              self.output_dir,
                              self.diversity_threshold,
                              self.qc_out_file,
                              self.run_info_file,
                              self.num_cores,
                              self.hamming_weight
                              )
        # run FTM
        all_counts = run_ftm.main()

        ####################
        ###   Sequence   ###
        ####################
        seq_message = "Determining consensus sequence...\n"
        print(seq_message)
        log.info(seq_message) 
           
        # instantiate sequencing module from sequencer.py
        sequence = seq.Sequencer(all_counts,
                                 fasta_df,
                                 self.output_dir)
         
         
        sequence.main()
         


if __name__ == "__main__":
    
    '''
    Instantiate ShortStack instance with options from config.txt
    '''
    try: 
        
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
                    kmer_length=config.getint("internal_options","kmer_length"),
                    covg_threshold=config.getint("internal_options","covg_threshold"),
                    qc_threshold=config.getint("internal_options","qc_threshold"),
                    diversity_threshold=config.getint("internal_options", "diversity_threshold"),
                    max_hamming_dist=config.getint("internal_options", "max_hamming_dist"),
                    hamming_weight=config.getint("internal_options", "hamming_weight"),
                    all_fov=config.getboolean("internal_options", "all_fov"))
    
    
        sStack.main()
    except Exception as e:
        print(e)
        
