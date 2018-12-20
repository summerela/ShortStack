'''
consensus.py

'''

import logging
import cython_funcs as cpy
from numba import jit
import numpy as np
import pandas as pd
import re
import swifter
from collections import defaultdict, Counter
from pathlib import Path


# import logger
log = logging.getLogger(__name__)

class Consensus():
    
    def __init__(self, 
                 molecule_seqs,
                 fasta_df,
                 out_dir):
    
        self.molecule_seqs = molecule_seqs
        self.fasta_df = fasta_df
        self.tiny_fasta = self.fasta_df[["groupID", "start"]]  #subset for faster merging
        self.output_dir = out_dir
    
      
    def count_molecules(self, x):
        
        nucs = defaultdict(Counter)
        
        for edge in x.swifter.apply(
            lambda row: 
            [(i,
             c,
             1) for i,c in enumerate(row.seq)],
            axis=1):
            for tuple in edge:
                key = tuple[0]
                nuc = tuple[1]
                weight = tuple[2]
                
                if nuc.upper() == "N":
                    nuc_weight = 0
                else:
                    nuc_weight = 1
                
                nucs[key][nuc] +=  nuc_weight
                
        return nucs

    @jit
    def get_seq(self, nuc_counts):
        
        seq_list = []
        
        for key, nuc in nuc_counts.items():
            max_nuc = []
            max_val = max(nuc.values())
            for x, y in nuc.items():
                if y == max_val:
                    max_nuc.append(x)
                    
            if len(max_nuc) != 1:
                max_nuc = "N"
            else:
                max_nuc = ''.join(max_nuc)
            
            seq_list.append(max_nuc)
        
        sequence = ''.join(seq_list)
        
        return sequence   
        
    def main(self):
        
        consensus_list = []
        
        # create consensus sequence for each feature
        for feature, data in self.molecule_seqs.groupby("region"):

            groupID = ''.join(data.region.unique())
            
            bases = self.count_molecules(data)
            consensus = self.get_seq(bases)
            
            consensus_info = "{},{}".format(groupID, consensus)
              
            consensus_list.append(consensus_info)
        
        # save molecule sequences to file
        consensus_out = Path("{}/consensus_seqs.tsv".format(self.output_dir))
        consensus_df = pd.DataFrame([sub.split(",") for sub in consensus_list], columns=["region", "seq"])
        consensus_df.to_csv(consensus_out, sep="\t", index=False) 
