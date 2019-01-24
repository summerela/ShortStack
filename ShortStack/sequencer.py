'''
sequencer.py

'''
import sys, warnings, logging, re, os, swifter, dask, psutil
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import multiprocessing as mp
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
from dask.bag.core import split
dask.config.set(scheduler='tasks')


# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 counts,
                 fasta_df,
                 out_dir,
                 cpus,
                 client):
    
        self.counts = counts
        self.fasta_df = fasta_df
        self.tiny_fasta = self.fasta_df[["region", "chrom", "start"]]  #subset for faster merging
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
          
    def break_edges(self, df):
         
        # break apart reads into edges
        df = dd.from_pandas(df, npartitions=self.cpus)
        df["nuc"] = df.Target.apply(lambda x: cpy.ngrams(x, 1))
        df = df.drop(["feature_div", "hamming"], axis=1)
        df = df.compute()

        return df
    
    def position_edges(self, x):
        '''
        purpose: break apart edges into start_pos:nuc, stop_pos:nuc, count tuples
        input: df created in create_ngrams containing list of edges in nuc column
        output: edge dataframe edge_df containing one row for each featureID
        '''
        
        edge_list = [(x.FeatureID, 
                      x.region,
                      str(int(x.pos) + i),
                      c,
                      x.bc_count) for i,c in enumerate(x.nuc)]

        return edge_list
     
    @jit
    def split_edges(self, edge_list):
        '''
        purpose: break apart rows of edges into edge dataframe
        input: edge_df from create_edges
        output: dataframe of start stop and count for each edge
        '''
        # convert list to datafame
        df = pd.DataFrame(edge_list.tolist())

        # unravel tuples to dataframe
        melted = pd.melt(df, value_name="edge").dropna()
        melted = melted.drop("variable", axis=1)
        
        # parse final edge dataframe output
        result = pd.DataFrame(melted['edge'].values.tolist())
        result = result.reset_index(drop=True)
        result.columns = ["featureID", "region", "pos", "nuc", "bc_count"]

        return result
    
    @jit
    def sum_edge_weights(self, edge_df):
        '''
        purpose:sum the egde weights for each position
        input: result dataframe created in split_deges
        output: weighted edges to feed to graph in edge_df
        '''
        
        # sum edge weights
        edge_df["weight"] = edge_df.groupby(["featureID", "pos", "nuc"])["bc_count"].transform('sum')
         
        # parse output
        edge_df = edge_df.drop("bc_count", axis=1)
        edge_df = edge_df.drop_duplicates()
        edge_df.sort_values(by=["featureID", "pos", "nuc", "weight"])

        return edge_df
    
    def ref_seqs(self):
        
        # break apart sequences into list of ngrams
        fasta_df = dd.from_pandas(self.fasta_df[["region", "id", "start"]], 
                                  npartitions=self.cpus)
        fasta_df["nucs"] = self.fasta_df.seq.apply(lambda x: cpy.ngrams(x, 1))

        # match edge grams with position
        fasta_df = fasta_df.apply(lambda row: [(row.id, 
                                                row.region,
                                  str(int(row.start) + i), 
                                  c) for i, c in enumerate(row.nucs)],
                                  axis=1,
                                  meta='object').compute()
        
        # convert list to datafame
        df = pd.DataFrame(fasta_df.tolist())

        return df
    
    @jit
    def parse_ref(self, ref_df):

        # unravel tuples to dataframe
        melted = pd.melt(ref_df, value_name="edge").dropna()
        melted = melted.drop("variable", axis=1)
        
        # parse final edge dataframe output
        result = pd.DataFrame(melted['edge'].values.tolist())
        result = result.reset_index(drop=True)

        result.columns = ["id", "region", "pos", "ref_nuc"]

        # clean up new line chars that appear at 1+ end of seq
        ref_df = result[result.ref_nuc != "\n"]

        # sort and add weight of 0 
        ref_df = ref_df.sort_values(by=["region", "pos", "ref_nuc"])

        return ref_df
    
    @jit
    def add_refs(self, group, ref_df):
        
        # pull out identifiers
        feature_id = group.featureID.unique()[0]
        group_region = group.region.unique()[0]
        
        # subset ref_df to region
        region_df = ref_df[ref_df.region == group_region]

        # outer join tables to keep missing bases
        group_df = group.merge(region_df, on=["region", "pos"],
                                   how='outer')
        
        # fill out missing columns for missing bases
        group_df["featureID"] = feature_id
        group_df["weight"][pd.isnull(group_df['weight'])] = 0
        # change base to N for missing bases
        group_df["nuc"][pd.isnull(group_df['nuc'])] = "N"
        
        # sort by position
        group_df = group_df.sort_values(by="pos").reset_index(drop=True)
        
        return group_df
    
    @jit
    def get_max(self, group):
        
        # filter for nuc with max weight at each position
        group["nuc_max"] = group.groupby(['pos'])['weight'].transform(max)
        group = group[group.weight == group.nuc_max]

        return group
    
    def get_seq(self, group):

        # pull out positions that have more than one possible nuc
        no_ties = group.groupby('pos').filter(lambda x: len(x)==1)
        multis = group.groupby('pos').filter(lambda x: len(x)> 1)
        print(multis)
        
        # convert ties to "N" **evntually replace with nuc/nuc**
        if not multis.empty:
            multis["nuc"] = "N"  
            multis = multis.drop_duplicates(subset=["pos", "nuc"], keep='first')   
            final_counts = pd.concat([no_ties, multis])
            final_counts = final_counts.drop("nuc_max", axis=1)
        else:
            final_counts = no_ties.drop("nuc_max", axis=1)
        
        return final_counts
        
    def main(self):

        # split apart ngrams lists
        edge_list_df = self.break_edges(self.counts)
          
        # match each nucleotide with position and count
        edge_list = edge_list_df.apply(lambda x: self.position_edges(x), axis=1)
          
        # parse edge list from lists of tuples to df
        edge_df = self.split_edges(edge_list)
          
        # sum edge weights
        weighted_df = self.sum_edge_weights(edge_df)        
        
        # create reference df
        ref_df = self.ref_seqs()
        
        # parse ref_df
        ref_df = self.parse_ref(ref_df)

        # add ref position of N for bases with no coverage
        final_countdown = weighted_df.groupby("featureID").apply(self.add_refs, ref_df)
        # get featureID back into dataset
        final_countdown.set_index("featureID", inplace=True)
        final_countdown.reset_index(drop=False, inplace=True) 
        
               
        # get max weighted base and create final sequence
        max_df = final_countdown.groupby("featureID").apply(self.get_max)
        # get featureID back into dataset
        max_df.set_index("featureID", inplace=True)
        max_df = max_df.reset_index(drop=False)

         
        # get ref info and save molecule counts to a file
        molecule_df = max_df.groupby('featureID').apply(self.get_seq)
        molecule_df = molecule_df.reset_index(drop=True)
        molecule_df = molecule_df.merge(ref_df, on=["region", "pos"])
        molecule_df = molecule_df.sort_values(by=["featureID", "pos"])
        molecule_count = os.path.join(self.output_dir, "molecule_counts.tsv")
        molecule_df.to_csv(molecule_count, sep="\t", index=False)
        
        # save sequences to file for consensus
        seq_list = []
        for feature, data in molecule_df.groupby("featureID"):
            
            region = ''.join(data.region.unique())
            seq_data = "{},{},{}".format(feature, 
                                             region, 
                                             ''.join(data.nuc.values))
            seq_list.append(seq_data)
              
        # save molecule sequences to file
        seq_outfile = os.path.join(self.output_dir, "molecule_seqs.tsv")
        seq_df = pd.DataFrame([sub.split(",") for sub in seq_list], columns=["FeatureID", "region", "seq"])
        seq_df.to_csv(seq_outfile, sep="\t", index=False)
        
        # coerce ref_df position back to int
        ref_df["pos"] = ref_df.pos.astype(int)

        return seq_df, ref_df
        

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
#         fasta_list = list(set(nuc_df.groupID.values.compute()))
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
#                     meta='object').compute()
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
            
#### non graphical solution #### 

# # import logger
# log = logging.getLogger(__name__)
# 
# class Sequencer():
#     
#     def __init__(self, 
#                  counts,
#                  fasta_df,
#                  out_dir):
#     
#         self.counts = counts
#         self.fasta_df = fasta_df
#         self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
#         self.output_dir = out_dir
#      
#     @jit   
#     def match_fasta(self, counts):
#         '''
#         purpose: split target reads into individual bases
#         input: all_ftm_counts created in ftm.return_all_counts()
#         output: individual bases, positions and counts per base
#         '''   
#         
#         # pull start position from fasta_df
#         nuc_df = counts.merge(self.tiny_fasta, left_on="groupID", right_on="groupID") 
#         nuc_df["pos"] = nuc_df.pos.astype(int) + nuc_df.start.astype(int) 
#         nuc_df.drop(["start", "gene"], axis=1, inplace=True)
# 
#         return nuc_df 
#     
#     def break_edges(self, df):
#         
#         # break apart reads into edges
#         df["edges"] = df.Target.apply(lambda x: cpy.ngrams(x, 1))
#         
#         return df
#     
#     def index_edges(self, x):
#          
#         edge_list = []
#                
#         edge_list.append(x.swifter.apply(
#             lambda row: 
#             [(i + row.pos,
#              c,
#              row.bc_count) for i,c in enumerate(row.edges)],
#             axis=1))
#          
#         # flatten lists of tuples
#         flat_list = [item for sublist in edge_list for item in sublist]
#         flat_edge_list = [item for sublist in flat_list for item in sublist]
#  
#         return flat_edge_list
#      
#     @jit
#     def sum_weights(self, edge_list):
#          
#         nucs = defaultdict(Counter)
#  
#         for key, nuc, weight in edge_list:
#             nucs[key][nuc] += weight
#      
#         return nucs
#      
#     @jit
#     def get_seq(self, nuc_counts):
#          
#         seq_list = []
#          
#         for key, nuc in nuc_counts.items():
#             max_nuc = []
#             max_val = max(nuc.values())
#             for x, y in nuc.items():
#                 if y == max_val:
#                     max_nuc.append(x)
#                      
#             if len(max_nuc) != 1:
#                 max_nuc = "N"
#             else:
#                 max_nuc = ''.join(max_nuc)
#              
#             seq_list.append(max_nuc)
#          
#         sequence = ''.join(seq_list)
#          
#         return sequence
# 
#     
#     def main(self):
#         
#         # maintain info on featureID and region
#         feature_list = self.counts[["FeatureID", "groupID"]].drop_duplicates().reset_index(drop=True)
#         
#         # convert target sequence to base and position with count
#         print("Matching targets with position..\n")
#         split_targets = self.match_fasta(self.counts)
#         edge_df = self.break_edges(split_targets)
#         
#         # group edge dataframe by feature to sequence
#         features = edge_df.groupby("FeatureID")
#         
#         # sequence each molecule
#         seq_list = []
#         for featureID, group in features:
#             groupID = ''.join(group.groupID.unique())
#             edge_list = self.index_edges(group)
#             weights = self.sum_weights(edge_list)
#             seq = self.get_seq(weights)
#             seq_data = "{},{},{}".format(featureID, groupID, seq)
#               
#             seq_list.append(seq_data)
#             
#         # save molecule sequences to file
#         seq_outfile = "{}/molecule_seqs.tsv".format(self.output_dir)
#         seq_df = pd.DataFrame([sub.split(",") for sub in seq_list], columns=["FeatureID", "region", "seq"])
#         seq_df.to_csv(seq_outfile, sep="\t", index=False)
#         
#         return seq_df
        
                

        
      

        
        
        

 
        


        
        
        


        
        
        
