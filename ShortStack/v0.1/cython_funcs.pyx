'''
functions in this script use cython to accelerate speed
'''

from itertools import chain
import re
 
def split_fasta(input_file):
    '''
    purpose: parses input fasta, splitting header and sequence
        used in parse_input.split_fasta()
    input: input fasta reference file
    output: lists of headers and sequences that will be passed to 
        parse_input.parse_fasta()
    '''
    info_list = []
    seq_list = []
      
    f = open(input_file, 'r')

    # read in fasta and test line for > or erroneous <
    for line in f:
        if line.startswith(">") or line.startswith("<"):
            # strip >:_ and spaces from from lines
            info_list.append(line.replace(">", "").replace("<", "").strip())
            # add sequences
        else:
            seq_list.append(line)
            
    return info_list, seq_list

def process_mutations(input_df):
    '''
    purpose: create mutated sequences from input vcf file
    input: mutation_df from assemble_mutations
    output: fasta_df of combined reference and alternate sequences for assembly
    '''
    # process deletions
    input_df.alt_seq[(input_df.mut_type == 'del')] = \
        [seq[0:n] for n, seq in zip((input_df.var_start-input_df.mut_length), input_df.ref_seq)] +  \
        input_df["alt"] + \
        [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
            
    # process insertions
    input_df.alt_seq[(input_df['mut_type'] == 'ins')] = \
        [seq[0:n] for n, seq in zip((input_df.var_start), input_df.ref_seq)] + \
        input_df["alt"] + \
        [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
        
    # process snvs
    input_df.alt_seq[(input_df['mut_type'] == 'snv')] = \
        [seq[0:n] for n, seq in zip((input_df.var_start), input_df.ref_seq)] + \
        input_df["alt"]+ \
        [seq[n:] for n, seq in zip((input_df.var_start +1), input_df.ref_seq)]
        
    input_df.alt_seq = input_df.alt_seq.str.strip()
            
    return input_df

def ngrams(str string, int n):
    '''
    purpose: break up input reference sequence into kmers
        used in align.match_perfects()
    input: fasta_df 
    output: all possible kmers for each input sequence in fasta_df
    '''
    cdef ngrams = []
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram)for ngram in ngrams]

def match_basecall(str pattern, str seq):
    '''
    purpose: compare reference sequence with each basecall
    to find positions of perfect matches
    input: basecall, sequence
    output: list of positions
    ### NOT USED ##
    '''

    cdef list matches = []
    
    # find positions where target maps to seq 
    matches = list(chain.from_iterable(map(lambda x: [x.start()]\
                                 ,re.finditer(pattern, seq))))
    return matches