'''
module for parsing input files for shortstack

'''

import pandas as pd
import numpy as np
import re, io, os, sys, logging, inspect,gzip
from _codecs import encode
import subprocess as sp
from pandas.io.json import json_normalize
import multiprocessing as mp
import allel
import pyximport; pyximport.install()
import cython_funcs as cpy
import ujson
from numba import jit

# import logger
log = logging.getLogger(__name__)

class Parse_files():
    
    # instance parameters
    def __init__(self, input_s6, output_dir, target_fa, mutation_file, encoder_file, 
                 qc_threshold=7, kmer_length=6, num_cores=4):
        self.input_s6= input_s6
        self.output_dir = output_dir
        self.target_fa = target_fa
        self.mutation_file = mutation_file
        self.encoder_file = encoder_file
        self.qc_threshold = int(qc_threshold)
        self.kmer_length = int(kmer_length)
        self.num_cores=num_cores
    
    def file_check(self, input_file):
        '''
        Purpose: check that input file paths exist and are not empty
        input: file path
        output: assertion error if file not found or is empty
        '''
        try:
            print("Checking {}".format(input_file))
            assert (os.path.isfile(input_file)) and (os.path.getsize(input_file) > 0)
        except AssertionError as e:
            error_message = "Check that {} exists and is not empty.".format(input_file, e)
            log.error(error_message)
            raise SystemExit(error_msg)
     
    def test_cols(self, input_df, df_name, required_cols):
        '''
        purpose: test that required columns are present in an input dataframe
        input: input_df, name of dataframe to use in error message as str, list of required_cols 
        output: assertion error for missing columns
        '''
        for x in required_cols:
            try:
                assert x in input_df.columns 
            except AssertionError as e:
                error_message = "{}\n{} column not found in {}".format(e, x, df_name)
                log.error(error_message)
                raise SystemExit(error_msg)  
                   
    def parse_s6(self):
        '''
         purpose: parse input s6 json file
         input: s6.json from imaging 
         output: s6 dataframe filtered for uncalled bases and qc score
                 qc dataframe with containing reads that were filtered out        
         '''
        
        # file handle to input s6.json
        print("Parsing S6 file:{}".format(self.input_s6))
        log.info("Parsing S6 file:{}".format(self.input_s6))
        
        # check that file exists and is not empty 
        self.file_check(self.input_s6)
        
        try:
            
            # read in json with C ultrafast json
            # this function will be parallelized when json file not nested improperly
            json_data = pd.read_json(self.input_s6)    
            feature_df = json_normalize(json_data["Features"], record_path=["Cycles","Pools"],
                                        meta=["FeatureID"])
             
             # filter out rows where basecall contains uncalled bases of 0 
            pass_calls = feature_df[feature_df.BC.str.contains("0") == False]
            uncalled_df = feature_df[feature_df.BC.str.contains("0")]
            uncalled_df["filter"] = "UC"
            
            # filter out rows where the Qual score falls below self.qc_threshold
            qc_string = "|".join(str(x) for x in range(1,self.qc_threshold))
            s6_df = pass_calls[pass_calls.Qual.str.contains(qc_string) == False].reset_index(drop=True)
            below_qc = feature_df[feature_df.Qual.str.contains(qc_string)]
            below_qc["filter"] = "Qual"
             
            # create qc dataframe
            qc_df = pd.concat([uncalled_df, below_qc], axis=0).reset_index(drop=True)
             
            # check that there are calls left after filtering
            try:
                assert s6_df.shape[0] > 0
            except Exception as e:
                error_msg = "No basecalls passed filtering from S6: \n{}".format(e)
                log.error(error_msg)
                raise SystemExit(error_msg)

            return s6_df, qc_df
        
        except Exception as e:
            error_msg = "Error parsing S6 file: \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)
                
    def parse_mutations(self):
        '''
        purpose: parse input mutation vcf file
        input: vcf file or gz vcf file, one alternate per line
        format: vcf 4.0 standard format
        output: mutation dataframe with mutation id and genomic position
        '''
        print("Parsing mutations file:{}".format(self.mutation_file))
        log.info("Parsing mutations file:{}".format(self.mutation_file))
        
        # check that file exists and is not empty 
        self.file_check(self.mutation_file)
        
        try:
            
            # read in mutation file, truncate to only one mutation per line
            mutation_df = allel.vcf_to_dataframe(self.mutation_file, 
                                                 fields=['CHROM', 'POS', 'ID', 
                                                         'REF', 'ALT', 
                                                         'variants/STRAND', 
                                                         'is_snp'], 
                                                 alt_number=1,
                                                 types={'CHROM':'object', 'POS':'int32',
                                                        'ID':'object', 'REF':'object',
                                                        'ALT':'object', 'STRAND':'S1',
                                                        'is_snp':'object'},
                                                 numbers={"ALT":1, "STRAND":1})

            # test that required columns are present
            self.test_cols(mutation_df, "mutation vcf", ["CHROM", "POS", "ID", "REF", "ALT", "STRAND"]) 
            mutation_df.rename(columns={"CHROM":"chrom", "POS":"pos", "ID":"id", "REF":"ref", \
                                 "ALT":"alt", "STRAND":"strand"}, inplace=True)

            # test that no two mutation ID's are the same
            assert mutation_df["id"].nunique() == mutation_df.shape[0]

            # drop any identical mutations
            mutation_df.drop_duplicates(["chrom", "pos", "ref", "alt", "strand"], inplace=True)
                                      
            # label mutation by length for bucketing
            conditions = [
                (mutation_df['ref'].str.len() <  mutation_df['alt'].str.len()),
                (mutation_df['ref'].str.len() >  mutation_df['alt'].str.len()),
                (mutation_df['is_snp'])
                ]
            choices = ['ins', 'del', 'snv']
             
            # add column with mutation type
            mutation_df['mut_type'] = np.select(conditions, choices, default='')      
            mutation_df.drop("is_snp", inplace=True, axis=1)      
            return mutation_df
        
        except Exception as e:
            error_msg = "Error parsing mutation vcf file: \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)
      
    def parse_encoding(self):
        '''
        purpose: parse barcode encoding file
        input: either user specified or default barcode encoding file
        format: tsv containing at least columns: Pool |Target | Color Index
        output: barcode dataframe
        '''
        print("Reading in encoding file from: {}".format(self.encoder_file))
        log.info("Reading in encoding file from: {}".format(self.encoder_file))
        # check that file exists and is not empty 
        self.file_check(self.encoder_file)
        
        try:
        
            required_cols = ["PoolID", "Target", "BC"]
            encoding = pd.read_csv(self.encoder_file, sep="\t", header=0,
                                   usecols=required_cols,
                                   dtype={"PoolID":int,
                                           "Target":str, 
                                           "BC":str})  
            
            # test that required columns are present
            self.test_cols(encoding, "encoding file", required_cols)
            
            # sort alphabetically by pool for faster matching
            encoding = encoding.sort_values(by=["PoolID", "BC"]).reset_index(drop=True)
           
            return encoding
        except Exception as e:
            error_msg = "Error parsing encoding file: \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)
 
    def split_fasta(self):
        '''
        purpose: split out fasta headers and sequences
        input: self.fasta_df
        output: list of fasta headers and seqs to be fed into parse_fasta()
        '''
    
        print("Parsing fasta file: {}".format(self.target_fa))
        log.info("Parsing fasta file: {}".format(self.target_fa))
        # check that file exists and is not empty 
        self.file_check(self.target_fa)

        # read in fasta using cython_funcs.split_fasta()
        try:
            info_list, seq_list = cpy.split_fasta(self.target_fa)

        # check that file exists                
        except IOError:                    
            print("{} does not exist".format(self.target_fa))
        
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
        fasta_df = pd.DataFrame(fasta_list, columns=["info", "seq"])
        
        # break apart header into dataframe columns
        try:
            fasta_df["id"],fasta_df["chrom"], \
            fasta_df["start"],fasta_df["stop"],\
            fasta_df["build"],fasta_df["strand"] = \
            list(zip(*fasta_df['info'].apply(lambda x: x.split(":"))))
        except Exception as e:
            raise SystemExit("Error: {} \n Ensure your fasta headers are in the format: id:chrom:start:stop:build:strand".format(e))    
        fasta_df.drop("info", axis=1, inplace=True)
        
        # test that the fasta contains information
        assert fasta_df.shape[0] > 0, "FASTA does not contain any information"
        fasta_df["chrom"] = fasta_df["chrom"].str.replace("chrom", '').str.replace("chr", '')
        
        # test that no two mutation ID's are the same
        assert fasta_df["id"].nunique() == fasta_df.shape[0]
        return fasta_df.reset_index(drop=True)
 
    def s6_to_json(self):
        '''
        Purpose: Check if s6 file csv or tsv, convert to json format
        input: s6 file (full path) in csv or tsv format
        format: s6 file with Feature/fov/x/y column header. Should be able to handle simulation s6 and imaging file s6 csvs.
        output: json s6 file generated in same directory as s6 file. Only generates json based on first FOV in s6 file. 
        '''    
        #Read in CSV
        if os.path.splitext(self.input_s6)[1] == '.csv':
            s6DF = pd.read_csv(self.input_s6)
        elif os.path.splitext(self.input_s6)[1] == '.tsv':
            s6DF = pd.read_table(self.input_s6)
            
        #Remove cheeky comma column, if it exists
        s6DF = s6DF.loc[:,~s6DF.columns.str.contains('^Unnamed')]
        #Remove whitespace from column headers
        s6DF.rename(columns=lambda x: x.strip(),inplace = True)
        #Get list of cycle numbers
        s6DF.insert(loc=1, column='FeatureID',
                       value=s6DF['fov'].astype(str) + '_' + s6DF['x'].astype(str) + '_' + s6DF['y'].astype(str))
        #Only get header values with Cycle/pool information. 
        Header = s6DF.columns.values.tolist()[5:]

        #Establish default Quality/Category metrics
        Qual = "999"
        Category = "000"
        #Only grabbing first FOV
        fovcheck = int(s6DF['fov'][0])
        #Make name of json file
        filename = os.path.splitext(os.path.basename(self.input_s6))[0]
        jsonname = "FOVID_" + str(fovcheck) + "_" + filename +'.json'
        jsonfile = os.path.join(os.path.dirname(self.input_s6), jsonname)
        self.input_s6 = jsonfile
        TotalDict = {'FovID':fovcheck,'Features':[]}
        #Iterate over entries in s6DF, construct dictionary of values for passing to json. Only grabbing first FOV in file.
        for rowindex, row in enumerate(s6DF.loc[s6DF['fov'] == fovcheck].values):
            #Grab values for FeatureID, X, Y values
            TotalDict['Features'].append({'FeatureID':row[1],"X":row[3],"Y":row[4],'Cycles':[]})
            #Counter that increases for every cycle seen to index list of cycles 
            cycleCount = -1
            #Iterate through barcode values to add them to total dictionary
            for index, BC in enumerate(row[5:]):
                #Get cycle and pool information from matching index in Header list. 
                column = Header[index]
                cycle = int(re.search('C(.*)P', column).group(1))
                pools = int(column.split('P')[1])
                #Check if Cycle exists in feature dictionary, if not create cycleID 
                #and insert BC information. Otherwise, add pool information to existing cycleID.
                if not any(d.get('CycleID', None) == cycle for d in TotalDict['Features'][rowindex]['Cycles']):
                    cycleCount += 1
                    TotalDict['Features'][rowindex]['Cycles'].append({"CycleID":cycle,"Pools":[{"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category}]})
                else:
                    TotalDict['Features'][rowindex]['Cycles'][cycleCount]['Pools'].append\
                    ({"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category})

        # Write JSON file
        with io.open(jsonfile, 'w', encoding='utf8') as outfile:
            outfile.write(ujson.dumps(TotalDict,indent = 4))
    
    def s6FileCheck(self):
        '''
        Purpose: Check if s6 file is .json, otherwise run converter method. 
        input: s6 filepath
        format: s6 of either json, csv, or tsv. Will break otherwise. 
        output: Alters s6 file or returns same s6 file. 
        '''
        if os.path.splitext(self.input_s6)[1] == '.json':
            self.input_s6 = self.input_s6
        else:
            self.s6_to_json()
       

    
    def main_parser(self):
        self.s6FileCheck()
        
        s6_df, qc_df = self.parse_s6()
        if self.mutation_file.lower() != "none": 
            mutation_df = self.parse_mutations()
        else: 
            mutation_df = "none"
            
        encoding_df = self.parse_encoding()
        fasta_df = self.parse_fasta()
        return s6_df, qc_df, mutation_df, encoding_df, fasta_df
            