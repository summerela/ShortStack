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
    
        self.counts = counts[['FeatureID', 'region', 'pos', 'Target','bc_count']] # sequencing to region
        self.fasta_df = fasta_df
        self.output_dir = out_dir
        self.cpus = cpus
        self.client = client
        
    def munge_counts(self, count_df):
        '''
        purpose: split bases into (pos, nuc) tuples
        input: all_counts df from ftm.py
        output: df of tuples for each feature
        '''

        # convert counts to dask dataframe
        counts = dd.from_pandas(self.counts, npartitions=self.cpus)
        counts["pos"] = counts.pos.astype(int)

        # split apart targets with position
        counts["nuc"] = counts.apply(lambda row: [(x + row.pos,y) for x,y in enumerate(row.Target)],
                                     axis=1)

        return counts
    
    @jit
    def split_bases(self, count_df):
        '''
        purpose: split feature tuple rows into one per column with 
            index retaining feature/region/count info
        input: munged counts df from munge_counts()
        output: df of one (pos, nuc) tuple per column per feature
        '''
        
        # set index to retain information
        count_df["idx"] = count_df.FeatureID + ":" + count_df.region + ":"  + count_df.bc_count.astype(str)
        count_df = count_df.set_index("idx")
        count_df = count_df.compute()

        # split tuples into columns and return back to dask
        base_df = pd.DataFrame(count_df["nuc"].values.tolist(), index=count_df.index).reset_index()
        base_df = dd.from_pandas(base_df, npartitions=self.cpus)
        
        return base_df
      
    def melt(self, df, id_vars=None, value_vars=None, var_name=None,
         value_name='value', col_level=None):
        '''
        purpose: dask doesn't have custom melt feature, wrapping pandas melt
        '''

        from dask.dataframe.core import no_default
        
        # map pandas melt to each chunk of df
        return df.map_partitions(pd.melt, meta=no_default, id_vars=id_vars,
                                value_vars=value_vars,
                                var_name=var_name, value_name=value_name,
                                col_level=col_level, token='melt')
    
    @jit    
    def melt_edges(self, df):
        '''
        purpose: unravel rows with multipel columns of tuples
            into df of one row per position per feature
        input: base_df created in split_bases
        output: df with one position per row, per feature
        '''
        
        df = self.melt(df, 
                       id_vars='idx',
                       value_name="nuc").drop("variable", axis=1)
        df = df.set_index("idx")
              
        return df
    
    @jit
    def split_tuples(self, df):
        '''
        purpose: split apart tuples and index to create final counts df for 
            base calling
        input: melted df created in melt_edges
        output: df with feature | region | chrom |  pos | nuc  | count 
        '''
        
        # break apart tuples
        df[["pos", "base"]] = pd.DataFrame(df['nuc'].values.tolist(), index=df.index)
        df = df.drop("nuc", axis=1).reset_index(drop=False)

        df = dd.from_pandas(df, npartitions=self.cpus)
        
        # do a lot of extra munging in dask to split idx to cols
        df['FeatureID'] = df["idx"].str.partition(":")[0]
        df['region'] = df['idx'].str.partition(":")[2]
        df['region'] = df['region'].str.partition(":")[0]
        df['counts'] = df['idx'].str.partition(":")[2]
        df['counts'] = df['counts'].str.partition(":")[2]
    
        df = df.drop("idx", axis=1)
        df["counts"] = df.counts.astype(float)

        return df
    
    @jit
    def calc_weights(self, df):
        '''
        purpose:sum the egde weights for each position
        input: result dataframe created in split_deges
        output: df with summed weights for each nuc per position
        '''
        df = df.compute()
        df["weight"] = df.groupby(["FeatureID", "region", "pos", "base"])["counts"].transform('sum')
        df = df.drop("counts", axis=1)
        df = df.drop_duplicates()
        
        return df
    
    def split_refs(self, fasta_df):
        '''
        purpose: split apart wt regions for bases with no coverage and vcf output
        input: fasta_df created in parse_input.py
        output: rows of (pos, nuc) tuples for each wt reference
        '''
        
        # subset fasta to wt windows
        region_fasta = self.fasta_df[self.fasta_df.id == self.fasta_df.region]
        
        # break apart sequences into list of ngrams
        fasta_dd = dd.from_pandas(region_fasta[["region", "chrom",
                                                 "start", "seq"]], 
                                  npartitions=self.cpus)
        
        # coerce start to int
        fasta_dd["start"] = fasta_dd.start.astype(int)
        
        # remove end line character
        fasta_dd["seq"] = fasta_dd.seq.str.strip()
        
        # break apart sequences intob bases
        fasta_dd["nucs"] = fasta_dd.seq.apply(lambda x: cpy.ngrams(x, 1))
        
        # split targets with position
        fasta_dd["bases"] = fasta_dd.apply(lambda row: [(x + row.start,y) for x,y in enumerate(row.nucs)],
                                     axis=1)
        
        return fasta_dd
    
    @jit
    def parse_refs(self, fasta_dd):
        '''
        purpose: pull apart the fasta_dd rows of tuples and parse into df
        input: fasta_dd created in split_refs
        output: fasta_df with one row per position for each input wt ref seq 
        '''
         
        # set index to retain information
        fasta_dd["idx"] = fasta_dd.region + ":" + fasta_dd.chrom
        fasta_dd = fasta_dd.set_index("idx")
        fasta_dd = fasta_dd.compute()
 
        # split tuples into columns 
        fasta_df = pd.DataFrame(fasta_dd["bases"].values.tolist(), index=fasta_dd.index).reset_index()
        fasta_dd = dd.from_pandas(fasta_df, npartitions=self.cpus)
         
        # split tuples per row into columns
        fasta_df = self.melt_edges(fasta_dd).compute()

        # get rid of NaN where some ref seqs were longer than others
        fasta_df = fasta_df.dropna(axis=0, how='any')
        
        fasta_df = pd.DataFrame(fasta_df['nuc'].values.tolist(), index=fasta_df.index).reset_index()
        fasta_df.columns = ["idx", "pos", "ref_base"]
        
        # split out the idx column
        fasta_df[["region", "chrom"]] = fasta_df['idx'].str.split(':',expand=True)
        fasta_df = fasta_df.drop("idx", axis=1)
        
        return fasta_df
    
    @jit
    def add_refs(self, group, ref_df):
        '''
        purpose: add ref base info and keep position with base of N for areas
            with no coverage
        input: fasta_df and count df
        output: count df with matching ref allele and N bases for positions
            with no coverage
        '''
        # pull out identifiers
        feature_id = group.FeatureID.unique()[0]
        group_region = group.region.unique()[0]
        
        # subset ref_df to seq window
        region_df = ref_df[ref_df.region == group_region]
        
        # pull out chrom to fill in blanks
        region_chrom = region_df.chrom.unique()[0]

        # outer join tables to keep missing bases
        group_df = group.merge(region_df, on=["region", "pos"],
                                   how='outer')
        
        # fill out missing columns for missing bases
        group_df["FeatureID"] = feature_id
        
        # change base to N for missing bases
        group_df["base"][pd.isnull(group_df['base'])] = "N"
        group_df["weight"][pd.isnull(group_df['weight'])] = 0
        group_df["chrom"][pd.isnull(group_df['chrom'])] = region_chrom
        
        # sort by position
        group_df = group_df.sort_values(by="pos").reset_index(drop=True)

        return group_df
    
    @jit
    def get_max(self, group):
        
        # filter for nuc with max weight at each position
        group["nuc_max"] = group.groupby(['pos'])['weight'].transform(max)
        group = group[group.weight == group.nuc_max]
        group = group.drop("nuc_max", axis=1)

        return group





    
    @jit     
    def main(self):
         
        # create list of position and base pair for each target
        counts = self.munge_counts(self.counts)
              
        # break bases up into separate columns
        munged_df = self.split_bases(counts)
              
        # melt columns of tuples into dataframes
        nuc_df = self.melt_edges(munged_df)
        nuc_df = nuc_df.compute()
              
        # split tuples of pos, nuc into columns
        base_df = self.split_tuples(nuc_df)
             
        # calculate edge weights
        weighted_df = self.calc_weights(base_df)
             
        # create ref seqs for bases with no coverage
        ref_dd = self.split_refs(self.fasta_df)
        ref_df = self.parse_refs(ref_dd)

        # add ref position of N for bases with no coverage
        full_covg_df = weighted_df.groupby("FeatureID").apply(self.add_refs, ref_df)
        # parse output
        full_covg_df.set_index("FeatureID", inplace=True)
        full_covg_df.reset_index(drop=False, inplace=True) 
        
        # get max weighted base and create final sequence
        max_df = full_covg_df.groupby("FeatureID").apply(self.get_max)
        # get featureID back into dataset
        max_df.set_index("FeatureID", inplace=True)
        max_df = max_df.reset_index(drop=False)
        max_df = max_df [["FeatureID", "region", "chrom", "pos", "base", "ref_base", "weight"]]
        max_df = max_df.sort_values(by=["FeatureID", "pos"])
        
        # save molecule counts to file
        mol_counts = os.path.join(self.output_dir, "molecule_counts.tsv")
        max_df.to_csv(mol_counts, sep="\t", index=False)
        
        return max_df, ref_df
    
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
        
        