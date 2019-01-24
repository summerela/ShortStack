'''
module for parsing input files for shortstack

'''

import sys, re, io, os, logging, allel, swifter, psutil
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
import pyximport; pyximport.install()
import cython_funcs as cpy
from numba import jit
from Bio.GenBank.Record import Feature
from dask import dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
import numpy as np



# import logger
log = logging.getLogger(__name__)

class Parse_files():
    
    # instance parameters
    def __init__(self, 
                 input_s6,
                 output_dir, 
                 target_fa, 
                 mutation_file, 
                 encoder_file,
                 cpus,
                 client):
        
        self.input_s6 = input_s6
        self.output_dir = output_dir
        self.target_fa = target_fa
        self.mutation_file = mutation_file
        self.encoder_file = encoder_file
        self.cpus = cpus
        self.client = client
    
    @jit        
    def test_cols(self, input_df, df_name, required_cols):
        '''
        purpose: test that required columns are present in an input dataframe
        input: input_df, name of dataframe to use in error message as str, list of required_cols 
        output: assertion error for missing columns
        '''
        error_message = "Required column not found in {}".format(df_name)
        for x in required_cols:
            assert x in input_df.columns, error_message
 
       
    @jit                       
    def parse_mutations(self):
        '''
        purpose: parse input mutation vcf file
        input: vcf file or gz vcf file, one alternate per line
        format: vcf 4.0 standard format
        output: mutation dataframe with mutation id and genomic position
        '''
            
        # read in mutation file, truncate to only one mutation per line
        mutation_df = allel.vcf_to_dataframe(str(self.mutation_file), 
                                             fields=['CHROM', 'POS', 'ID', 
                                                     'REF', 'ALT', 
                                                     'variants/STRAND',
                                                     'variants/svlen'], 
                                             alt_number=1,
                                             types={'CHROM':'object', 'POS':'int32',
                                                    'ID':'object', 'REF':'object',
                                                    'ALT':'object', 'STRAND':'S1',
                                                    'variants/svlen':int},
                                             numbers={"ALT":1, "STRAND":1})
        
        # test that required columns are present
        self.test_cols(mutation_df, "mutation vcf", ["CHROM", "POS", "ID", "REF", "ALT", "STRAND"]) 
        mutation_df.rename(columns={"CHROM":"chrom", "POS":"pos", "ID":"id", "REF":"ref", \
                             "ALT":"alt", "STRAND":"strand"}, inplace=True)
        
        # test that no two mutation ID's are the same
        assert mutation_df["id"].nunique() == mutation_df.shape[0], "Check for duplicated mutation id's."
        
        # check that mutations do not fall on the same chrom/region
        dups = mutation_df[mutation_df.duplicated(['chrom','pos', 'ref', 'alt', 'strand'],keep=False)]
        assert dups.empty, "Check that mutation chrom, pos, ref, alt and strand are not duplicated."

        # drop any identical mutations
        mutation_df.drop_duplicates(["chrom", "pos", "ref", "alt", "strand"], inplace=True)
                                  
        # convert mutation lengths to mutation types
        mutation_df['mut_type'] = ""
        mutation_df['mut_type'][mutation_df["svlen"] == 0] = "SNV"   
        mutation_df['mut_type'][mutation_df["svlen"] < 0] = "DEL"  
        mutation_df['mut_type'][mutation_df["svlen"] > 0] = "INS"    
        mutation_df["id"] = mutation_df.id.astype(str) + "_" + mutation_df.mut_type.astype(str)
        mutation_df["svlen"] = abs(mutation_df["svlen"]) # convert deletion length to pos int
        
        return mutation_df
    
    @jit    
    def parse_encoding(self):
        '''
        purpose: parse barcode encoding file
        input: either user specified or default barcode encoding file
        format: tsv containing at least columns: Pool |Target | Color Index
        output: barcode dataframe
        '''

        required_cols = ["PoolID", "Target", "BC", "bc_length"]
        encoding = pd.read_csv(self.encoder_file, sep="\t", header=0,
                               usecols=required_cols,
                               dtype={"PoolID":'object',
                                       "Target":'object', 
                                       "BC":'object',
                                       "bc_length":np.uint8},
                                      comment='#',
                                      engine="c")
        
        # test that required columns are present
        self.test_cols(encoding, "encoding file", required_cols)
        
        # sort alphabetically by pool for faster matching
        encoding = encoding.sort_values(by=["PoolID", "BC"]).reset_index(drop=True)

        return encoding
    
    @jit
    def split_fasta(self):
        '''
        purpose: split out fasta headers and sequences
        input: self.fasta_df
        output: list of fasta headers and seqs to be fed into parse_fasta()
        '''

        # read in fasta using cython_funcs.split_fasta()
        info_list, seq_list = cpy.split_fasta(self.target_fa)
        
        return info_list, seq_list

    def parse_fasta(self):
        '''
        purpose: parse input fasta files containing target sequences 
        input: fasta file in standard fasta format
        format: header must have format: id:chrom:start:stop:build
        output: dataframe of sequences and relevant info
        '''
                
        # split fasta into header and seq
        info_list, seq_list = self.split_fasta()
            
        # zip together sequence info from header and sequence
        fasta_list = list(zip(info_list, seq_list))
        
        # zip into pandas dataframe
        fasta_df = pd.DataFrame(fasta_list, 
                                columns=["info", "seq"],
                                dtype='object')
        
        # break apart header into dataframe columns
        fasta_df["id"],fasta_df["chrom"], \
        fasta_df["start"],fasta_df["stop"],\
        fasta_df["build"],fasta_df["strand"] = list(zip(*fasta_df['info'].swifter.apply(lambda x: x.split(":"))))

        fasta_df = fasta_df.drop("info", axis=1)
        # test that the fasta contains information
        assert not fasta_df.empty, "FASTA does not contain any information"
        fasta_df["chrom"] = fasta_df["chrom"].str.replace("chrom", '').str.replace("chr", '')

        # test that no two mutation ID's are the same
        assert fasta_df["id"].nunique() == fasta_df.shape[0], "FASTA contains duplicates."
        
        # assume that fasta only contains one wt per region
        fasta_df["region"] = fasta_df["id"]

        fasta_df = fasta_df.reset_index(drop=True)

        return fasta_df

    @jit
    def read_s6(self, input_s6):
        
        # specify S6 datatypes
        dtypes = {'Features':'object',
          'fov': 'object',
          'x': 'object',
          'y': 'object'}
        
        # read in S6 file and create feature id's
        df = dd.read_csv(input_s6, dtype=dtypes, blocksize=1024*1024)
        
        # Remove cheeky comma column, if it exists
        df = df.loc[:,~df.columns.str.contains('^Unnamed')]
        # Remove whitespace from column headers
        df.columns = df.columns.str.strip()
        
        df["FeatureID"] = df["fov"].astype(str) + "_" + df["x"].astype(str) + "_" + df["y"].astype(str)
        df= df.drop(["Features", "fov", "x", "y"], axis=1)

        return df
    

    def melt(self, frame, id_vars=None, value_vars=None, var_name=None,
         value_name='value', col_level=None):

        from dask.dataframe.core import no_default
    
        return frame.map_partitions(pd.melt, meta=no_default, id_vars=id_vars,
                                    value_vars=value_vars,
                                    var_name=var_name, value_name=value_name,
                                    col_level=col_level, token='melt')

    @jit
    def pivot_s6(self, input_s6):
        
        # expand basecalls to one row per feature
        s6_df = self.melt(input_s6, id_vars="FeatureID")
        
        # split up pool and cycle info
        s6_df["cycle"] = s6_df['variable'].str.partition('P')[0]
        s6_df["cycle"] = s6_df["cycle"].str.partition("C")[2]
        s6_df["pool"] = s6_df['variable'].str.partition('P')[2]

        # drop variable column
        s6_df = s6_df.drop("variable", axis=1)
        
        col_map = {"FeatureID":"FeatureID",
                   "value":"BC",
                   "cycle":"cycle",
                   "pool":"pool"}
        
        s6_df = s6_df.rename(columns=col_map)
        s6_df["BC"] = s6_df.BC.astype('str')

        return s6_df
    
    @jit
    def filter_s6(self, input_s6):

        # filter out rows where basecall contains uncalled bases of 0 
        pass_calls = input_s6[input_s6.BC.str.contains("0") == False]
        # filter out rows with missing digits (non 3 spotters)
        s6_df = pass_calls[pass_calls.BC.astype(int) > 111111].compute()
        
        # save raw call data to file
        s6_df_outfile = os.path.join(self.output_dir, "all3spotters.tsv")
        s6_df.to_csv(s6_df_outfile, sep="\t", index=False)

        return s6_df

    def main_parser(self):
        
        try:
        
            # parse mutation file if provided
            if self.mutation_file != "none": 
                mutation_df = self.parse_mutations()
            else: 
                mutation_df = "none"

            # parse encoding file    
            encoding_df = self.parse_encoding()
            
            # parse input fasta file
            fasta_df = self.parse_fasta()
            
            # read in and parse S6
            s6_rows = self.read_s6(self.input_s6)
            s6_df = self.pivot_s6(s6_rows)
            
            # filtering S6 dataframe
            s6_df = self.filter_s6(s6_df)
      
            return mutation_df, s6_df, fasta_df, encoding_df
        except Exception as e:
            log.error(e)
            raise SystemExit(e)

    
            