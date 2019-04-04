'''
sequencer.py

'''
import sys, warnings, logging, re, os, swifter, dask, psutil
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
                 cpus,
                 client):
        
        self.cpus = cpus
        self.counts = counts[["FeatureID", "region", "pos", "Target", "bc_count"]].compute(ncores=self.cpus)
        self.counts.pos = self.counts.pos.astype(int)
        self.counts.bc_count = self.counts.bc_count.astype(float)
        self.fasta = fasta_df
        self.fasta.columns = ['ref_seq', 'id', 'chrom', 'start', 'stop', 'build', 'strand', 'region']
        self.output_dir = out_dir
        
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
        alignments = pairwise2.align.globalms(query, 
                        target, 
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
        base_df.columns = ["FeatureID", "pos", "A", "C", "G", "T", "max_nuc", "region"]

        # save to a file
        base_out = os.path.join(self.output_dir, "base_counts.tsv")
        base_df.to_csv(base_out, index=False, sep="\t")

        print("Determining consensus sequence...\n")
        ## return consensus sequence
        seq_list = base_df.groupby("FeatureID").apply(self.join_seq)
        df = pd.DataFrame(seq_list).reset_index(drop=True)
        df1 = pd.DataFrame(df[0].tolist(), index=df.index) 
        seq_df = pd.DataFrame(df1[0].tolist(), index=df1.index) 
        seq_df.columns = ["FeatureID", "region", "feature_seq"]

        seq_df.to_csv("/home/selasady/ShortStack/ShortStack/feasibility/seq_df.tsv", sep="\t", index=False)
        self.fasta.to_csv("/home/selasady/ShortStack/ShortStack/feasibility/fasta.tsv", sep="\t", index=False)

        print(self.fasta.head())
        print(self.fasta.dtypes)
        print(seq_df.head())
        print(seq_df.dtypes)
        
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
#         seq_df = seq_df.drop("ref_seq", axis=1) # uncomment after testing

        return seq_df
    
        

### graph solution ###

# class Sequencer():
#     
#     def __init__(self, 
#                  counts,
#                  fasta_df,
#                  out_dir,
#                  cpus,
#                  client):
#     
#         self.counts = counts
#         self.fasta_df = fasta_df
#         self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
#         self.output_dir = out_dir
#         self.cpus = cpus
#         self.client = client
#      
#     @jit   
#     def match_fasta(self, counts):
#         '''
#         purpose: split target reads into individual bases
#         input: all_ftm_counts created in ftm.return_all_counts()
#         output: individual bases, positions and counts per base
#         '''   
#         
#         # convert counts to dask 
#         counts = dd.from_pandas(counts, npartitions=self.cpus)
#         
#         # pull start position from fasta_df
#         nuc_df = dd.merge(counts, self.tiny_fasta, left_on="groupID", right_on="groupID") 
#         nuc_df["pos"] = nuc_df.pos.astype(int) + nuc_df.start.astype(int)
#         nuc_df = nuc_df.drop("start", axis=1) 
#         
#         fasta_list = list(set(nuc_df.groupID.values.compute(scheduler='processes', num_workers=self.cpus)))
# 
#         return nuc_df, fasta_list
#     
#     def build_baseGraph(self, row): 
#         '''
#         purpose: create a graph with one node for every position in the sequence
#             so that genomic position is retained even when there is no coverage 
#         input: ftm_fasta, the subset fasta_df for regions in fasta_list
#         output: list of graphs objects with graph.name = groupID
#         '''
#         
#         #initiate graph
#         node_list = []
# 
#         # build graph with nodes of N and weight 0 to retain positional info
#         for pos in range(int(row.start), int(row.stop)):
#             node_line = '"{}:N", "{}:N", weight=0'.format(int(pos), int(pos)+1)
#             node_list.append(node_line)
#         
#         return node_list
# 
#     def create_ngrams(self, df):
#         '''
#         purpose: use ngrams function to break each kmer into 2bp edges
#         input: nuc_df created in sequencer.match_fasta
#         output: df containing a column with a list of edges
#         '''
#         
#         # use cython ngrams function to break apart kmers
#         df["nuc"] = df.Target.map(lambda x: cpy.ngrams(x, 2))
#         df["idx"] = df.FeatureID.astype(str) + ":" + \
#                     df.groupID.astype(str)
#         df = df.set_index("idx")
#         
#         return df
#     
#     def create_edges(self, df):
#         '''
#         purpose: break apart edges into start_pos:nuc, stop_pos:nuc, count tuples
#         input: df created in create_ngrams containing list of edges in nuc column
#         output: edge dataframe edge_df containing one row for each featureID
#         '''
#         
#         edge_df = df.apply(lambda row: [(str(row.pos + i) + ":" + ''.join(c)[0],
#                     str(row.pos + (1 + i)) + ":" + ''.join(c)[1],
#                     row.bc_count) for i, c in enumerate(row.nuc)],
#                     axis=1, 
#                     meta='object').compute(scheduler='processes', num_workers=self.cpus)
#            
#         return edge_df
#     
#     @jit
#     def split_edges(self, edge_df):
#         '''
#         purpose: break apart rows of edges into edge dataframe
#         input: edge_df from create_edges
#         output: dataframe of start stop and count for each edge
#         '''
#         
#         # unravel list of edges to one row per edge
#         df = pd.DataFrame(edge_df.values.tolist(), index=edge_df.index)
#         df = df.reset_index(drop=False)
#         
#         # unravel tuples to dataframe
#         melted = pd.melt(df, id_vars="idx", value_name="edge").dropna()
#         melted = melted.drop("variable", axis=1)
#         melted = melted.set_index("idx")
#         
#         # parse final edge dataframe output
#         result = pd.DataFrame(melted['edge'].values.tolist(), index=melted.index)
#         result = result.reset_index(drop=False)
#         
#         result.columns = ["id", "edge1", "edge2", "count"]
# 
#         return result
#     
#     @jit
#     def sum_edge_weights(self, edge_df):
#         '''
#         purpose:sum the egde weights for each position
#         input: result dataframe created in split_deges
#         output: weighted edges to feed to graph in edge_df
#         '''
#         
#         # sum edge weights
#         edge_df["weight"] = edge_df.groupby(["id", "edge1", "edge2"])["count"].transform("sum")
#         edge_df.drop("count", axis=1, inplace=True)
#         edge_df = edge_df.drop_duplicates()
#         
#         return edge_df
#     
#     def get_path(self, edge_df):
#         '''
#         purpose: get best path through graph for each feature
#         input: edge df containing edges and weights = counts
#         output: best path and graph object for each feature
#         '''
#          
#         G = nx.from_pandas_edgelist(
#             edge_df,
#             source="edge1",
#             target="edge2",
#             edge_attr=["weight"],
#             create_using=nx.OrderedDiGraph()
#         )
#  
#         # check nodes for ties
#         for node in G.nodes:
#             
#             # create list of all edges for each node
#             edges = G.in_edges(node, data=True)
#             
#             # if there are multiple edges
#             if len(edges) > 1:
#                 
#                 # find max weight
#                 max_weight = max([edge[2]['weight'] for edge in edges])
#                 
#                 tie_check = []
#                 for edge in edges:
#                     # pull out all edges that match max weight
#                     if edge[2]["weight"] == max_weight:
#                         tie_check.append(edge)
#                 
#                 # check if there are ties       
#                 if len(tie_check) > 1:
#                     for x in tie_check:
#                         
#                         # flag node as being a tie
#                         G.node[x[0]]["tie"] = True
#                 
#         # return longest path
#         longest_path = nx.dag_longest_path(G) 
#         
#         return longest_path, G
#     
#     @jit
#     def trim_path(self, longest_path, graph):
#         '''
#         purpose: prune graph for any ties and conver to no calls
#         input: graph and path from get_path()
#         output: final best path through graph
#         '''
#         
#         final_path = [] 
#         
#         for node in longest_path:
#             # check if node has a tie
#             if node in nx.get_node_attributes(graph, "tie"):
#                 # return no call for tie
#                 node = "N"
#             else:
#                 # return just the nucleotide
#                 node = node.split(":")[1]
#             
#             # add node to final path
#             final_path.append(node)
#             
#         return ''.join(final_path)
#     
#     @jit
#     def split_id(self, df):
#         '''
#         purpose: break apart id column into freature and groupID 
#         input: final path dataframe with sequence for each molecule
#         output: final path df with featureID and groupID to pass to conseensus
#         '''
#         
#         df[['featureID', 'region']] = df['id'].str.split(':', n=1, expand=True)
#         df = df.drop("id", axis=1)
#         df = df[["featureID", "region", "seq"]]
#         
#         return df
#         
# 
#     def main(self):
#         
#         # convert target sequence to base and position with count
#         split_targets, fasta_list = self.match_fasta(self.counts)
#         
#         # subset fasta df for regions in fasta_list
#         ftm_fasta = self.fasta_df[["groupID", "start", "stop"]][self.fasta_df['groupID'].isin(fasta_list)]
# 
#         # generate list of base graphs for each ftm called region
# #         graph_list = [ x for x in ftm_fasta.apply(self.build_baseGraph, axis=1)]
#         
#         # slit dataframe into edges
#         ngram_df = self.create_ngrams(split_targets)
#         
#         # split apart ngrams lists
#         edge_list = self.create_edges(ngram_df)
#         
#         # return edge dataframe
#         edge_df = self.split_edges(edge_list)
#         
#         # sum edge weights
#         summed_edge = self.sum_edge_weights(edge_df)
#         
#         print(summed_edge.head())
#         
#         
#         
#         raise SystemExit()
#         
#         # sequence each molecule
#         seq_list = []
#         
#         for id, group in summed_edge.groupby("id"):
# 
#             path, graph = self.get_path(group)
#             seq = self.trim_path(path, graph)
# 
#             # parse sequence data
#             seq_data = "{},{}".format(id, seq)   
#             seq_list.append(seq_data)
#             
#         # save molecule sequences to file
#         seq_outfile = Path("{}/molecule_seqs.tsv".format(self.output_dir))
#         seq_df = pd.DataFrame([sub.split(",") for sub in seq_list], columns=["id", "seq"])
#         
#         # break apart id field into featureID and region
#         seq_df = self.split_id(seq_df)
# 
#         seq_df.to_csv(seq_outfile, sep="\t", index=False)
# 
#         return seq_df
        
        