'''
sequencer.py

'''
import sys, warnings, logging, dask, os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numba import jit
import pandas as pd
from collections import defaultdict, Counter
from Bio import pairwise2
dask.config.set(scheduler='tasks')
from operator import itemgetter

# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 counts,
                 fasta_df,
                 out_dir,
                 prefix,
                 cpus,
                 client,
                 align_params):
        
        self.cpus = cpus
        self.counts = counts[["FeatureID", "region", "pos", "Target", "bc_count"]].compute(ncores=self.cpus)
        self.counts.pos = self.counts.pos.astype(int)
        self.counts.bc_count = self.counts.bc_count.astype(float)
        self.fasta = fasta_df
        self.fasta = self.fasta.rename(columns={"seq":"ref_seq"})
        self.output_dir = out_dir
        self.prefix = prefix
        self.client = client
        self.align_params = align_params
    
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
        col_list = ["A", "T", "G", "C", "U", "-"]
        for col in col_list:
            if col not in base_df:
                base_df[col] = 0.0

        # replace NaN with 0.0
        base_df = base_df.fillna(0.0)

        # pull out nuc with max count
        base_df["nuc"] = base_df[["A", "T", "G", "C", "U", "-"]].idxmax(axis=1)

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

    def align_seqs(self, x):

        # pull out attributes for feature
        query = ''.join(x.feature_seq.unique())
        target = x.ref_seq.to_list()
        featureID = x.name
        chrom = ''.join(x.chrom.unique())
        xStart = ''.join(x.start.unique())
        region = ''.join(x.region.unique())
        refSeq = ''.join(x.ref_seq.unique())

        align_list = []
        for ref in target:

            if self.align_params == "supervised":
                alignment = pairwise2.align.localms(ref,
                                                      query,
                                                      1, 0, -3, -.1,  # recommended penalties to favor snv over indel
                                                      one_alignment_only=True)  # returns only best score
            else:
                alignment = pairwise2.align.localms(ref,
                                                      query,
                                                      1, -1, -2, 0,  # recommended penalties to favor snv over indel
                                                      one_alignment_only=True)  # returns only best score
            align_list.append(alignment)

        # find alignment with max score
        flat_list = [item for sublist in align_list for item in sublist]

        if len(flat_list) > 1:
            max_alignment = max(flat_list, key=itemgetter(2))
        else:
            max_alignment = [item for sublist in flat_list for item in sublist]

        # return formatted max alignment
        alignment = max_alignment[1]
        score = max_alignment[2]
        align_len = max_alignment[4]
        pct_sim = round(score/align_len*100, 2)

        aligned = [featureID, chrom, xStart, region, alignment, refSeq, pct_sim]

        return aligned

    @jit(parallel=True)
    def main(self):

        print("Counting reads per base...\n")
        # split reads up by base
        base_df = self.counts.groupby(["FeatureID"]).apply(self.get_path).reset_index(drop=False)
        base_df.columns = ["FeatureID", "pos", "A", "C", "T", "G", "U", "-", "max_nuc", "region"]

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
        alignments = pd.DataFrame(seq_df.groupby("FeatureID").apply(self.align_seqs).reset_index(drop=True))
        alignments.columns = ["align_list"]
        alignments[['FeatureID', 'chrom', 'start_pos', 'region', 'alignment', 'ref_seq', 'pct_sim']] = pd.DataFrame(
            alignments.align_list.values.tolist(),
            index=alignments.index).reset_index(drop=True)
        alignments = alignments.drop("align_list", axis=1)

        return alignments
