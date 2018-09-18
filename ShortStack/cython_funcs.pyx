'''
functions in this script use cython to accelerate speed
'''

from itertools import chain
import re
import numpy as np
 
def split_fasta(input_file):
    '''
    purpose: parses input fasta, splitting header and sequence
        used in parse_input.split_fasta()
    input: input fasta reference file
    output: lists of headers and sequences that will be passed to 
        parse_input.parse_fasta()
    '''
    cdef list info_list = []
    cdef list seq_list = []
      
    f = open(input_file, 'r')

    # read in fasta and test line for > or erroneous <
    for line in f:
        # skip header comments
        if not line.startswith("#"):
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

def calc_symmetricDiff(x):
    # create list of target sets for all potential targets
    cdef list targets = []
    cdef set u = set()
    
    #  take the set of each target list for the feature
    targets = [set(i) for i in x.target_list.values] 
    
    # calc intersection
    u = set.intersection(*targets)

    # symmetric difference = num unique targets in gene - intersection all 
    symDiff = x.feature_div - len(u)

    return symDiff

def calc_hamming(str a, str b):
    '''
    purpose: calculate hamming distance between two strings
    input: ngrams and targets
    output: matrix of hamming distances between each target and ref seq ngram
    '''
    cdef int k, l, c
    c = 0
    l = len(a)
    for k from 0 <= k < l:
        if a[k] != b[k]:
            c += 1
    return c

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