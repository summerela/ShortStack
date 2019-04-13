'''
sequencer.py

'''
import sys, warnings, logging, os, swifter, dask
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numba import jit
import pandas as pd
from collections import defaultdict, Counter
from Bio import pairwise2
dask.config.set(scheduler='tasks')
pd.set_option('display.max_columns', 100)


# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 counts,
                 fasta_df,
                 out_dir,
                 prefix,
                 cpus,
                 client):
        
        self.cpus = cpus
        self.counts = counts[["FeatureID", "region", "pos", "Target", "bc_count"]].compute(ncores=self.cpus)
        self.counts.pos = self.counts.pos.astype(int)
        self.counts.bc_count = self.counts.bc_count.astype(float)
        self.fasta = fasta_df
        self.fasta.columns = ['ref_seq', 'id', 'chrom', 'start', 'stop', 'build', 'strand', 'region']
        self.output_dir = out_dir
        self.prefix = prefix
        self.client = client
    
    @jit(parallel=True)   
    def update_weight_for_row(self, row, graph):
        
        # enumerate targets to break hexamers into bases        
        for pos, letter in enumerate(row.Target):
            base_pos = row.pos + pos
            graph[base_pos][letter] += row.bc_count
    
    @jit(parallel=True)
    def get_path(self, grp):

        # setup dictionary by position
        graph = defaultdict(Counter)
        
        # update weights for each position observed
        grp.apply(self.update_weight_for_row, graph=graph,
                  axis=1)
        
        # coerce results to dataframe         
        base_df = pd.DataFrame.from_dict(graph, orient='index')

        # ensure that dataframe contains all possible nuc columns
        col_list = ["A", "T", "G", "C", "-"]
        for col in col_list:
            if col not in base_df:
                base_df[col] = 0.0

        # replace NaN with 0.0
        base_df = base_df.fillna(0.0)

        # pull out nuc with max count
        base_df["nuc"] = base_df.idxmax(axis=1)

        # add regional info
        base_df["region"] = grp.region.unique()[0]
        
        return base_df
    
    @jit(parallel=True)
    def join_seq(self, grp):
    
        seq_list = []
        
        # sort group by position
        grp = grp.sort_values("pos", ascending=True)
        
        # join together molecule sequence
        molecule_seq = ''.join(grp.max_nuc.tolist()).strip()
    
        # append the sequence to seq_list
        seq_tup = (grp.FeatureID.unique()[0], grp.region.unique()[0], molecule_seq)
        seq_list.append(seq_tup)
        
        return seq_list
    
    @jit(parallel=True)
    def align_seqs(self, x):    
        
        # parameters may need to be experimentally adjusted
        query = x.feature_seq
        target = x.ref_seq
        
        # create pairwise global alignment object
        alignments = pairwise2.align.globalms(target,
                        query,
                        1, 0, -3, -.1, # recommended penalties to favor snv over indel
                        one_alignment_only=True) # returns only best score
        
        # for each alignment in alignment object, return aligned sequence only
        for a in alignments:
            query, alignment, score, start, align_len = a
            return alignment
    
   
    @jit(parallel=True)
    def main(self):
        
        print("Counting reads per base...\n")
        # split reads up by base
        base_df = self.counts.groupby(["FeatureID"]).apply(self.get_path).reset_index(drop=False)
        base_df.columns = ["FeatureID", "pos", "A", "C", "G", "T", "-", "max_nuc", "region"]

        # save to a file
        base_out = os.path.join(self.output_dir, self.prefix + "_base_counts.tsv")
        base_df.to_csv(base_out, index=False, sep="\t")

        print("Determining consensus sequence...\n")
        ## return consensus sequence
        seq_list = base_df.groupby("FeatureID").apply(self.join_seq)
        df = pd.DataFrame(seq_list).reset_index(drop=True)
        df1 = pd.DataFrame(df[0].tolist(), index=df.index) 
        seq_df = pd.DataFrame(df1[0].tolist(), index=df1.index) 
        seq_df.columns = ["FeatureID", "region", "feature_seq"]
        
        print("Adding reference sequences to align...\n")
        ## add reference sequence for each feature
        seq_df = seq_df.merge(self.fasta,
                     on=["region"],
                     how='left')  
        # strip new line characters from sequences
        seq_df["ref_seq"] = seq_df.ref_seq.str.strip()
        seq_df["feature_seq"] = seq_df.feature_seq.str.strip()
        
        print("Aligning consensus sequences...\n")
        seq_df["alignment"] = seq_df.apply(self.align_seqs,
                                           axis=1)

        return seq_df
        
        