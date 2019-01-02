'''
sequencer.py

'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re
import swifter
import networkx as nx
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing as mp
import dask
import dask.dataframe as dd
from dask.dataframe.io.tests.test_parquet import npartitions
dask.config.set(scheduler='threads')
import psutil
import swifter

# import logger
log = logging.getLogger(__name__)

class Sequencer():
    
    def __init__(self, 
                 counts,
                 fasta_df,
                 out_dir,
                 client):
    
        self.counts = counts
        self.fasta_df = fasta_df
        self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
        self.output_dir = out_dir
        self.cpus = psutil.cpu_count() - 3
        self.client = client
     
    @jit   
    def match_fasta(self, counts):
        '''
        purpose: split target reads into individual bases
        input: all_ftm_counts created in ftm.return_all_counts()
        output: individual bases, positions and counts per base
        '''   
       
        # convert counts to dask 
        counts = dd.from_pandas(counts, npartitions=self.cpus)
        counts = self.client.persist(counts)
        
        # pull start position from fasta_df
        nuc_df = dd.merge(counts, self.tiny_fasta, left_on="groupID", right_on="groupID") 
        nuc_df["pos"] = nuc_df.pos.astype(int) + nuc_df.start.astype(int) 
        nuc_df = nuc_df.drop(["start"], axis=1)

        return nuc_df 
    
    @jit
    def break_edges(self, df):
        '''
        purpose: break apart kmers into 2 base pairs to form edge list for graph
        input: nuc_df created in sequencer.match_fasta()
        output: nuc_df with new "nuc" column containing an edge list for each kmer
        '''
        
        # create ridiculous index so dask doesn't complain 
        df["id"] = df["FeatureID"].astype(str) + \
                   df["groupID"].astype(str) + \
                   df["Target"].astype(str) + \
                   df["pos"].astype(str) + \
                   df["hamming"].astype(str)
                   
        df["id"] = df.id.str.replace("_", "")
        
        df = df.set_index(df["id"],
                          shuffle="tasks",
                          npartitions=self.cpus*2)
        
        # drop id column
        df = df.drop("id", axis=1)
        
        return df

    def create_ngrams(self, df):
        
        # use cython ngrams function to break apart kmers
        df["nuc"] = df.Target.map(lambda x: cpy.ngrams(x, 2))
        
        # persist data frame for downstream analysis
        df = self.client.persist(df)

        return df
    
    def create_nodes(self, x):

        edge_list = []
         
        edge_list.append(x.apply(
             lambda row: 
            [(str(row.pos + i) + ":" + ''.join(c)[0],
             str(row.pos + (1 + i)) + ":" + ''.join(c)[1],
             row.bc_count) for i, c in enumerate(row.nuc)],
            axis=1))
        
        # flatten lists of tuples
        flat_list = [item for sublist in edge_list for item in sublist]
        flat_edge_list = [item for sublist in flat_list for item in sublist]
        
        return flat_edge_list
    
    @jit
    def sum_edge_weights(self, edge_list):
        
        # dataframe for summing totals and adding to graph
        edge_df = pd.DataFrame(edge_list, columns=["edge1", "edge2", "count"])
        
        # sum edge weights
        edge_df["weight"] = edge_df.groupby(["edge1", "edge2"])["count"].transform("sum")
        edge_df.drop("count", axis=1, inplace=True)
 
        return edge_df

        
    def get_path(self, edge_df):
         
        G = nx.from_pandas_edgelist(
            edge_df,
            source="edge1",
            target="edge2",
            edge_attr=["weight"],
            create_using=nx.OrderedDiGraph()
        )
 
        # check nodes for ties
        for node in G.nodes:
            
            # create list of all edges for each node
            edges = G.in_edges(node, data=True)
            
            # if there are multiple edges
            if len(edges) > 1:
                
                # find max weight
                max_weight = max([edge[2]['weight'] for edge in edges])
                
                tie_check = []
                for edge in edges:
                    # pull out all edges that match max weight
                    if edge[2]["weight"] == max_weight:
                        tie_check.append(edge)
                
                # check if there are ties       
                if len(tie_check) > 1:
                    for x in tie_check:
                        
                        # flag node as being a tie
                        G.node[x[0]]["tie"] = True
                
        # return longest path
        longest_path = nx.dag_longest_path(G) 
        
        return longest_path, G
    
    @jit 
    def trim_path(self, longest_path, graph):
        
        final_path = [] 
        
        for node in longest_path:
            # check if node has a tie
            if node in nx.get_node_attributes(graph, "tie"):
                # return no call for tie
                node = "N"
            else:
                # return just the nucleotide
                node = node.split(":")[1]
            
            # add node to final path
            final_path.append(node)
            
        return ''.join(final_path)

    
    def main(self):
        
        
        # maintain info on featureID and region
        feature_list = self.counts[["FeatureID", "groupID"]].drop_duplicates().reset_index(drop=True)
        
        # convert target sequence to base and position with count
        print("Matching targets with position..\n")
        split_targets = self.match_fasta(self.counts)
        
        # create graph edges
        print("Creating graph edges..\n")
        edge_df = self.break_edges(split_targets)
        
        # slit dataframe into edges
        print("Splitting edges by nucleotide...\n")
        ngrams = self.create_ngrams(edge_df)
        ngrams = self.client.compute(ngrams)
        ngrams = ngrams.result()

        # group information by featureID
        features = ngrams.groupby("FeatureID")

        # sequence each molecule
        seq_list = []
        
        print("Creating graph...\n")
        for featureID, group in features:
            groupID = ''.join(group.groupID.unique())
            
            edge_list = self.create_nodes(group)

            edge_df = self.sum_edge_weights(edge_list)

            path, graph = self.get_path(edge_df)
            seq = self.trim_path(path, graph)
            
            # parse sequence data
            seq_data = "{},{},{}".format(featureID, groupID, seq)   
            seq_list.append(seq_data)
            
        # save molecule sequences to file
        print("Saving sequences to file...\n")
        seq_outfile = Path("{}/molecule_seqs.tsv".format(self.output_dir))
        seq_df = pd.DataFrame([sub.split(",") for sub in seq_list], columns=["FeatureID", "region", "seq"])
        seq_df.to_csv(seq_outfile, sep="\t", index=False)

        return seq_df
            
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
        
                

        
      

        
        
        

 
        


        
        
        


        
        
        
