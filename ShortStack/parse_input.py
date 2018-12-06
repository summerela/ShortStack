'''
module for parsing input files for shortstack

'''

import pandas as pd
import re, io, os, logging
from pandas.io.json import json_normalize
import allel
import pyximport; pyximport.install()
import cython_funcs as cpy
import ujson
from numba import jit
import swifter
from Bio.GenBank.Record import Feature


# import logger
log = logging.getLogger(__name__)

class Parse_files():
    
    # instance parameters
    def __init__(self, input_s6, output_dir, target_fa, mutation_file, encoder_file, 
                 qc_threshold=7, all_fov="true"):
        self.input_s6= input_s6
        self.output_dir = output_dir
        self.target_fa = target_fa
        self.mutation_file = mutation_file
        self.encoder_file = encoder_file
        self.qc_threshold = int(qc_threshold)
        self.qc_string = "|".join(str(x) for x in range(1,self.qc_threshold))
        self.all_fov = all_fov
    
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
    def read_s6(self):
        '''
        purpose: read in s6 file in JSON format
        input: file path to s6 file as self.input_s6
        output: feature_df of parsed JSON S6 file
        '''
        # file handle to input s6.json
        print("Reading in S6 file:{}".format(self.input_s6))
            
        # this function will be parallelized when json file not nested improperly
        print("Parsing JSON format...\n")
        json_data = pd.read_json(self.input_s6)
    
        return json_data
    
    @jit
    def convert_JSON(self, json_obj):   
        feature_df = json_normalize(json_obj["Features"], record_path=["Cycles","Pools"],
                                    meta=["FeatureID"])  
        
        return feature_df
    
    @jit           
    def parse_s6(self, feature_df):
        '''
         purpose: parse input s6 json file
         input: s6.json from imaging 
         output: s6 dataframe filtered for uncalled bases and qc score
                 qc dataframe with containing reads that were filtered out        
         '''
        print("Parsing S6 file")
    
        # filter out rows where basecall contains uncalled bases of 0 
        pass_calls = feature_df[feature_df.BC.str.contains("0") == False]
        # filter out rows with missing digits (non 3 spotters)
        pass_calls = pass_calls[pass_calls.BC.astype(int) > 111111]
        
        # filter out rows where the Qual score falls below self.qc_threshold
        s6_df = pass_calls[pass_calls.Qual.str.contains(self.qc_string) == False].reset_index(drop=True)
        
        # save raw call data to file
        s6_df_outfile = "{}/all3spotters.tsv".format(self.output_dir)
        s6_df.to_csv(s6_df_outfile, sep="\t", index=False)
        
        return s6_df
    
    @jit   
    def check_s6(self, s6_df):
        '''
        purpose: check that basecalls remain after qc filtering 
        input: s6_df created in parse_s6
        output: pass or fail assertion
        '''
             
        # check that there are calls left after filtering
        error_msg = "No basecalls passed filtering from S6: \n{}".format(self.input_s6)
        assert s6_df.shape[0] > 0, error_msg
       
    @jit                       
    def parse_mutations(self):
        '''
        purpose: parse input mutation vcf file
        input: vcf file or gz vcf file, one alternate per line
        format: vcf 4.0 standard format
        output: mutation dataframe with mutation id and genomic position
        '''
        print("Parsing mutations file:{}".format(self.mutation_file))
        log.info("Parsing mutations file:{}".format(self.mutation_file))
            
        # read in mutation file, truncate to only one mutation per line
        mutation_df = allel.vcf_to_dataframe(self.mutation_file, 
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
        assert mutation_df["id"].nunique() == mutation_df.shape[0]

        # drop any identical mutations
        mutation_df.drop_duplicates(["chrom", "pos", "ref", "alt", "strand"], inplace=True)
                                  
        # convert mutation lengths to mutation types
        mutation_df['mut_type'] = ""
        mutation_df['mut_type'][mutation_df["svlen"] == 0] = "SNV"   
        mutation_df['mut_type'][mutation_df["svlen"] < 0] = "DEL"  
        mutation_df['mut_type'][mutation_df["svlen"] > 0] = "INS"    
        mutation_df["id"] = mutation_df.id.astype(str) + "_" + mutation_df.mut_type.astype(str)

        return mutation_df
    
    @jit    
    def parse_encoding(self):
        '''
        purpose: parse barcode encoding file
        input: either user specified or default barcode encoding file
        format: tsv containing at least columns: Pool |Target | Color Index
        output: barcode dataframe
        '''
        print("Reading in encoding file from: {}".format(self.encoder_file))
        log.info("Reading in encoding file from: {}".format(self.encoder_file))

        required_cols = ["PoolID", "Target", "BC"]
        encoding = pd.read_csv(self.encoder_file, sep="\t", header=0,
                               usecols=required_cols,
                               dtype={"PoolID":int,
                                       "Target":str, 
                                       "BC":str},
                                      comment='#')
        
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
    
        print("Parsing fasta file: {}".format(self.target_fa))
        log.info("Parsing fasta file: {}".format(self.target_fa))

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
        fasta_df = pd.DataFrame(fasta_list, columns=["info", "seq"])
        
        # break apart header into dataframe columns
        fasta_df["id"],fasta_df["chrom"], \
        fasta_df["start"],fasta_df["stop"],\
        fasta_df["build"],fasta_df["strand"] = list(zip(*fasta_df['info'].swifter.apply(lambda x: x.split(":"))))

        fasta_df.drop("info", axis=1, inplace=True)
        # test that the fasta contains information
        assert fasta_df.shape[0] > 0, "FASTA does not contain any information"
        fasta_df["chrom"] = fasta_df["chrom"].str.replace("chrom", '').str.replace("chr", '')
        
        # test that no two mutation ID's are the same
        assert fasta_df["id"].nunique() == fasta_df.shape[0], "FASTA contains duplicates."
        fasta_df["groupID"] = fasta_df["id"]

        return fasta_df.reset_index(drop=True)
    
    @jit
    def s6FileCheck(self):
        '''
        Purpose: Check if s6 file is .json, otherwise run converter method. 
        input: s6 filepath
        format: s6 of either json, csv, or tsv. Will break otherwise. 
        output: Alters s6 file or returns same s6 file. 
        '''
        print(os.path.splitext(self.input_s6)[1].lower())
        if os.path.splitext(self.input_s6)[1].lower() != '.json':
            
            print("Converting {} to JSON format.".format(self.input_s6))
            if self.all_fov == False:
                self.s6_to_json()
            else:
                self.s6_to_json_all()
    
    def s6_to_json(self):
        '''
        Purpose: Check if s6 file csv or tsv, convert to json format
        input: s6 file (full path) in csv or tsv format
        format: s6 file with Feature/fov/x/y column header. Should be able to handle simulation s6 and imaging file s6 csvs.
        output: json s6 file generated in same directory as s6 file. Only generates json based on first FOV in s6 file. 
        '''    
        #Read in CSV
        if os.path.splitext(self.input_s6, )[1] == '.csv':
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
                    TotalDict['Features'][rowindex]['Cycles'].append({"CycleID":cycle,"Pools":[{"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category, "CycleID":cycle}]})
                else:
                    TotalDict['Features'][rowindex]['Cycles'][cycleCount]['Pools'].append\
                    ({"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category, "CycleID":cycle})

        # Write JSON file
        with io.open(jsonfile, 'w', encoding='utf8') as outfile:
            outfile.write(ujson.dumps(TotalDict,indent = 4))
        
    def s6_to_json_all(self):
        '''
        Purpose: Check if s6 file csv or tsv, convert to json format
        input: s6 file (full path) in csv or tsv format
        format: s6 file with Feature/fov/x/y column header. Should be able to handle simulation s6 and imaging file s6 csvs.
        output: json s6 file generated in same directory as s6 file. Only generates json based on first FOV in s6 file. 
        '''    
        #Read in CSV
        if os.path.splitext(self.input_s6, )[1] == '.csv':
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
        #Indicating that this is all FOVs
        fovcheck = "All"
        #Make name of json file
        filename = os.path.splitext(os.path.basename(self.input_s6))[0]
        jsonname = 'All_FOV_' + filename + '.json'
        jsonfile = os.path.join(os.path.dirname(self.input_s6), jsonname)
        self.input_s6 = jsonfile
        TotalDict = {'FovID':fovcheck,'Features':[]}
        #Iterate over entries in s6DF, construct dictionary of values for passing to json.
        for rowindex, row in s6DF.iterrows():
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
                    TotalDict['Features'][rowindex]['Cycles'].append({"CycleID":cycle,"Pools":[{"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category, "CycleID":cycle}]})
                else:
                    TotalDict['Features'][rowindex]['Cycles'][cycleCount]['Pools'].append\
                    ({"PoolID":pools,'BC':str(BC),'Qual':Qual,"Category":Category, "CycleID":cycle})

        # Write JSON file
        with io.open(jsonfile, 'w', encoding='utf8') as outfile:
            outfile.write(ujson.dumps(TotalDict,indent = 4))
    
    
    def main_parser(self):
        
        # check for CSV vs JSON
        self.s6FileCheck()

        # read in s6 file
        json_obj = self.read_s6()
        feature_df = self.convert_JSON(json_obj)
        
        # parse s6 file and return qc_df
        s6_df = self.parse_s6(feature_df)
        self.check_s6(s6_df)
        
        # parse mutation file if provided
        if self.mutation_file != "none": 
            mutation_df = self.parse_mutations()
        else: 
            mutation_df = "none"
        
        # parse encoding file    
        encoding_df = self.parse_encoding()
        
        # parse input fasta file
        fasta_df = self.parse_fasta()

        return s6_df, mutation_df, encoding_df, fasta_df
            