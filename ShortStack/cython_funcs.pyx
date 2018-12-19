'''
functions in this script use cython to accelerate speed
'''
from ipywidgets.widgets.interaction import _get_min_max_value
from collections import Counter
 
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

def calc_hamming(str a, str b, int maxHD):
    '''
    purpose: calculate hamming distance between two strings
    input: ngrams and targets
    output: matrix of hamming distances between each target and ref seq ngram
    '''
    cdef int l, k
    c = 0
    l = len(a)
    for k from 0 <= k < l:
        if a[k] != b[k]:
            if (c < maxHD):
                c += 1
            else:
                c = "X"
                break
    return (a,b, c)

# def calc_seq_hamming(str a, str b):
#     '''
#     purpose: calculate hamming distance between two strings
#     input: ngrams and targets
#     output: matrix of hamming distances between each target and ref seq ngram
#     '''
#     cdef int l, k, c
#     c = 0
#     l = len(a)
#     nuc_list = []
#     for k from 0 <= k < l:
#         if a[k] != b[k]:
#             mismatch = "{}:{}:{}".format(b[k],a[k],k)
#             nuc_list.append(mismatch)
#             c += 1
#     
#     if len(nuc_list) == 0:
#         nuc_list= ""
#     return c, nuc_list

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
    '''
    purpose: find reads that are unique to a target/feature
    input: dataframe of perfect matches
    output: 
    '''
    # create list of target sets for all potential targets
    cdef list targets, result
    cdef set u 
    
    #  get a list of target sets for each feature id 
    targets = list(x.target_list.values)
     
    result = []
 
    for set_element in targets:
        result.append(len(set_element.difference(set.union(*[x for x in targets if x != set_element]))))
 
    return result

