'''
module for aligning ShortStack basecalls

vcf assumptions:
- follow vcf 4.3 format
- variants broken into one alt per line
- variants left normalized
- ref and alt strings must include base before the event

'''
import logging
import pandas as pd
import pandasql as psql
import numpy.testing as npt
import pyximport; pyximport.install()
import cython_funcs as cpy

# import logger
log = logging.getLogger(__name__)

class AssembleMutations():
    
    def __init__(self, fasta_df, mutation_df, run_info_file, s6_df):
        self.fasta_df = fasta_df
        self.mutation_df = mutation_df
        self.run_info_file = run_info_file
        self.s6_df = s6_df

    def mutation_checker(self):
        '''
        purpose: check that input mutations are within genomic range of fasta_df
        input: mutation_df and fasta_df
        output: adds column to mutation df with ref_ids that mutation falls in
                removes mutations that are not found within any reference and 
                lists them in run_info.txt       
        '''
        
        print("Checking that input mutations are within range of input fasta file.")
        
        # alias dataframes for lazy typing
        m_df = self.mutation_df.reset_index(drop=True)
        f_df = self.fasta_df.reset_index(drop=True)
        
        # check that mutations are within chrom, start, stop, strand of any ref seq
        query = '''select m_df.chrom, m_df.pos,
            m_df.id as var_id, m_df.ref, m_df.alt,
            f_df.id as ref_match, m_df.mut_type, m_df.strand,
            f_df.seq as ref_seq, f_df.start as ref_start, f_df.build
            from m_df  
            inner join f_df  
            on 
            m_df.chrom = f_df.chrom 
            where m_df.pos between f_df.start and f_df.stop
            and m_df.strand = f_df.strand
        '''
        # check that there is at least one valid mutation left to assemble
        try:
            # pull out mutations that are valid
            # note, pandas query not supporting merge 'where pos between start and stop'
            valid_mutations = psql.sqldf(query)        

            # check that there are any valid mutations
            assert valid_mutations.shape[0] > 0
            
            # pull out the invalids to add to run_info.txt
            invalids = m_df[~m_df['id'].isin(valid_mutations['var_id'])]

            # if there are invalid mutations, write to self.run_info_file 
            if invalids.shape[0] > 0:
                
                # write info to run_info.txt
                invalid_list = list(invalids.id)

                var_line = '''
                \n The following variants were not located within a ref seq and were omitted:\n''' +\
                 ','.join(invalid_list) + "\n"
                with open(self.run_info_file, 'a+') as f:
                    f.writelines(var_line)
                    
                log.info(var_line)    
            return valid_mutations
        
        except Exception as e:
            error_msg = "No valid mutations to assemble. Check run_info.txt. \n{}".format(e)
            log.error(error_msg)
            raise SystemExit(error_msg)
        
    def create_mutation_ref(self, valid_mutations):
        '''
        purpose: Create mutation reference sequence from input vcf file 
        for guided,high-qual mode
        input: valid_muations df created by assembly.mutation_checker() 
               and fasta_df imported on init from run_shortstack.py
        output: adds var seq as an entry to fasta_df with var id
        '''
        print("Creating variant sequences from input VCF file.")
        # change length of column output for testing
        pd.set_option('max_colwidth',600)

        # calculate variant starting position on reference, subtract 1 for 0 based indexing
        valid_mutations["var_start"] = (valid_mutations["pos"].astype(int) - valid_mutations["ref_start"].astype(int)) - 1
        
        # check that mutation ref allele matches position on reference
        npt.assert_array_equal( valid_mutations.apply(lambda x: x['ref_seq'][x['var_start']], 1),
                                 valid_mutations.apply(lambda x: x['ref'][-1], 1), \
                               "Check that mutations found in vcf are found in ref seq.")

        # calculate length of mutation
        valid_mutations['mut_length'] = 1 # initialize column with 1 so we don't have to calc snv's
        valid_mutations.mut_length[(valid_mutations['mut_type'] == 'del')] = \
            valid_mutations["ref"].str.len() - valid_mutations["alt"].str.len()
        valid_mutations.mut_length[(valid_mutations['mut_type'] == 'ins')] = \
            valid_mutations["alt"].str.len() - valid_mutations["ref"].str.len()
        
        # add column to store alternate seq as string
        valid_mutations["alt_seq"] = ""
                                                                       
        # create mutated sequences
        valid_mutations = cpy.process_mutations(valid_mutations)
        
        # reshape valid_mutations to concat with fasta_df
        valid_dict = {"id":valid_mutations["var_id"]+"_mut",
                      "chrom":valid_mutations["chrom"],
                      "start":valid_mutations["ref_start"],
                      "stop":(valid_mutations["ref_start"].astype(int) + valid_mutations["mut_length"].astype(int)),
                      "seq":valid_mutations["alt_seq"],
                      "strand":valid_mutations["strand"]
                      }
        mutant_fasta = pd.DataFrame(valid_dict, columns=valid_dict.keys()) 
        mutant_fasta.reset_index(drop=True, inplace=True)    
        return mutant_fasta
                   
    def main(self):
        '''
        Purpose: run all functions to assemble mutated sequences
        '''

        # filter out mutations from vcf not located within any fasta seq
        valid_mutations = self.mutation_checker()
        # add mutation sequences to s6_df if vcf provided
        mutant_fasta = self.create_mutation_ref(valid_mutations)
        return mutant_fasta
        


        