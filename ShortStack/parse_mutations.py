'''
module for aligning ShortStack basecalls

vcf assumptions:
- follow vcf 4.3 format
- variants broken into one alt per line
- variants left normalized
- ref and alt strings must include base before the event

'''
import logging, os, sys, swifter, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
import pandasql as psql
import numpy.testing as npt
import pyximport; pyximport.install()
import cython_funcs as cpy
from numba import jit

# import logger
log = logging.getLogger(__name__)

class AssembleMutations():
    
    def __init__(self, fasta_df, mutation_df, s6_df):
        self.fasta_df = fasta_df.reset_index(drop=True)
        self.mutation_df = mutation_df.reset_index(drop=True)
        self.s6_df = s6_df

    def mutation_checker(self):
        print("Verifying mutations.")
        '''
        purpose: check that input mutations are within genomic range of fasta_df
        input: mutation_df and fasta_df
        output: adds column to mutation df with ref_ids that mutation falls in  
        '''

        # alias dataframes for lazy typing
        m_df = self.mutation_df
        f_df = self.fasta_df
        
        # check that mutations are within chrom, start, stop, strand of any ref seq
        query = '''select m_df.chrom, m_df.pos,
            m_df.id as var_id, m_df.ref, m_df.alt,
            f_df.id as region, m_df.mut_type, m_df.strand, 
            f_df.seq as ref_seq, f_df.start as ref_start, f_df.build
            from m_df  
            inner join f_df  
            on 
            m_df.chrom = f_df.chrom 
            where m_df.pos between f_df.start and f_df.stop
            and m_df.strand = f_df.strand
        '''
        # pull out mutations that are valid
        # note, pandas query not supporting merge 'where pos between start and stop'
        valid_mutations = psql.sqldf(query)  

        # check that there are any valid mutations
        error_message = "No valid mutations found within wildtype regions."
        if valid_mutations.empty:
            log.error(error_message)
            raise SystemExit(error_message)
        
        return valid_mutations
    
    @jit(parallel=True)
    def find_invalids(self, valids): 
        print("Parsing mutations.")
        '''
        purpose: checks if there are mutations listed in vcf file that do not 
            fall within the range of any input reference sequences
        input: valids dataframe created in parse_mutations.mutation_checker(),
            mutation_df
        output: filters out invalid mutations that do not fall within the genomic 
            range of any input reference sequences
        '''
        
        # calculate variant starting position on reference
        valids["var_start"] = (valids["pos"].astype(int) - valids["ref_start"].astype(int))
        
        # pull out the invalids to add to log
        invalids = self.mutation_df[~self.mutation_df['id'].isin(valids['var_id'])]

        # if there are invalid mutations, write to log file
        if not invalids.empty:
            
            # write info to log
            invalid_list = list(invalids.id)

            var_line = '''
            \n The following variants were not located within a ref seq and were omitted:\n''' +\
             ','.join(invalid_list) + "\n"
                
            log.info(var_line)   
         
        return valids
    
    def check_valids(self, valid_mutations):
        '''
        purpose: checks that the specified reference allele from the vcf file
            is at the position specified in the reference sequence
        input: valid mutation dataframe
        output: will throw an error if mutations listed in vcf are not found in ref sequence
        '''

        # check that first ref position of mutation matches wt sequence
        npt.assert_array_equal(valid_mutations.apply(lambda x: x['ref_seq'][x['var_start']], axis=1),
        valid_mutations.apply(lambda x: x['ref'][0], axis=1),
         "Ensure mutations in vcf are found within the reference sequence.")
   
    def process_mutations(self, input_df):
        print("Encoding mutation type.")
        '''
        purpose: create mutated sequences from input vcf file
        input: mutation_df from assemble_mutations
        output: fasta_df of combined reference and alternate sequences for assembly
        '''
    
        # process deletions
        input_df.alt_seq[(input_df.mut_type == 'DEL')] = \
            [seq[0:n] for n, seq in zip((input_df.var_start-input_df.mut_length), input_df.ref_seq)] +  \
            input_df["alt"] + \
            [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
                
        # process insertions
        input_df.alt_seq[(input_df['mut_type'] == 'INS')] = \
            [seq[0:n] for n, seq in zip((input_df.var_start), input_df.ref_seq)] + \
            input_df["alt"] + \
            [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
            
        # process snvs
        input_df.alt_seq[(input_df['mut_type'] == 'SNV')] = \
            [seq[0:n] for n, seq in zip((input_df.var_start), input_df.ref_seq)] + \
            input_df["alt"]+ \
            [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
            
        input_df.alt_seq = input_df.alt_seq.str.strip()

        return input_df

    @jit(parallel=True)
    def create_mutation_ref(self, valid_mutations):
        print("Creating reference mutations.")
        '''
        purpose: Create mutation reference sequence from input vcf file 
        for guided,high-qual mode
        input: valid_muations df created by assembly.mutation_checker() 
               and fasta_df imported on init from run_shortstack.py
        output: adds var seq as an entry to fasta_df with var id
        '''
         
        # calculate length of mutation
        valid_mutations['mut_length'] = 1 # initialize column with 1 so we don't have to calc snv's
        
        # calculate length of deletions
        valid_mutations.mut_length[(valid_mutations['mut_type'].str.lower() == 'del')] = \
            valid_mutations["ref"].str.len() - valid_mutations["alt"].str.len()
        
        # calculate length of insertions
        valid_mutations.mut_length[(valid_mutations['mut_type'].str.lower() == 'ins')] = \
            valid_mutations["alt"].str.len() - valid_mutations["ref"].str.len()
        
        # add column to store alternate seq as string
        valid_mutations["alt_seq"] = ""
                                                                     
        # create mutated sequences
        valid_mutations = self.process_mutations(valid_mutations)

        # reshape valid_mutations to concat with fasta_df
        valid_dict = {"id":valid_mutations["var_id"]+"_mut",
                      "chrom":valid_mutations["chrom"],
                      "start":valid_mutations["ref_start"],
                      "stop":(valid_mutations["ref_start"].astype(int) + valid_mutations["mut_length"].astype(int)),
                      "seq":valid_mutations["alt_seq"],
                      "strand":valid_mutations["strand"],
                      "region":valid_mutations["region"]
                      }
        
        # convert to pandas dataframe 
        mutant_fasta = pd.DataFrame(valid_dict, columns=valid_dict.keys())

        return mutant_fasta
         
    def main(self):
        '''
        Purpose: run all functions to assemble mutated sequences
        '''
        
        try:

            # filter out mutations from vcf not located within any fasta seq
            valid_mutations = self.mutation_checker()
            
            # parse returned mutations
            mutations_df = self.find_invalids(valid_mutations)
            self.check_valids(mutations_df)
            
            # add mutation sequences to s6_df if vcf provided
            mutant_fasta = self.create_mutation_ref(mutations_df)
            
            return mutant_fasta
        except Exception as e:
            log.error(e)
            raise SystemExit(e)
        


        