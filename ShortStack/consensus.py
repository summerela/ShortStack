'''
consensus.py

'''
import configparser as cp   # parse config files
import argparse    #parse user arguments
from logging.handlers import RotatingFileHandler    # set max log size before rollover
from logging.handlers import TimedRotatingFileHandler    # set time limit of 31 days on log files
import sys, warnings, logging, dask
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from numba import jit
import pandas as pd
from collections import defaultdict, Counter
from Bio import pairwise2
import time, psutil, os, shutil
from dask.distributed import Client
import dask.dataframe as dd

dask.config.set(scheduler='tasks')

# get path to config file
parser = argparse.ArgumentParser(description='HexSembler Consensus Sequencing.')
parser.add_argument('-c','--config', help='Path to HexSembler config file',required=True)
args = parser.parse_args()

class Consensus():

    def __init__(self,
                 config_path,
                 molecules,
                 fasta_df,
                 out_dir=os.path.curdir,
                 log_path = "./",):

        self.mem_limit = psutil.virtual_memory().available - 1000
        self.cpus = int(psutil.cpu_count())
        self.molecules = molecules
        self.fasta = dd.read_parquet(fasta_df, engine='fastparquet')
        raise SystemExit(self.fasta)
        self.fasta.columns = ['ref_seq', 'id', 'chrom', 'start', 'stop', 'build', 'strand', 'region']
        self.out_dir = out_dir
        self.client = Client(name="HexSembler",
                             memory_limit=self.mem_limit,
                             n_workers=self.cpus - 2,
                             threads_per_worker=self.cpus / 2,
                             processes=True)

        # gather run options
        self.today = time.strftime("%Y%m%d")

        # set directory path
        self.log_path = os.path.abspath(log_path)
        self.output_dir = os.path.join(self.out_dir, "consensus")

        # create output dir if not exists
        self.create_outdir(self.output_dir)

        ### Setup logging ###
        now = time.strftime("%Y%d%m_%H:%M:%S")
        today_file = "{}_HexSembler_consensus.log".format(self.today)
        log_file = os.path.join(self.log_path, today_file)
        FORMAT = '{"@t":%(asctime)s, "@l":%(levelname)s, "@ln":%(lineno)s, "@f":%(funcName)s}, "@mt":%(message)s'
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w',
                            format=FORMAT)

        self.log = logging.getLogger(__name__)

        ## setup rotating log handlers
        size_handler = RotatingFileHandler(
            log_file, maxBytes=10000000, backupCount=10)

        time_handler = TimedRotatingFileHandler(
            log_file, when='D', interval=31, backupCount=10
        )
        # add handlers for size and time limits
        self.log.addHandler(size_handler)
        self.log.addHandler(time_handler)

        @staticmethod
        def create_outdir(output_dir):
            '''
            Check if a directory exists, and if not, create it
            :param output_dir: path to directory
            :return: directory created
            '''
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            except AssertionError as e:
                error_message = "Unable to create output dir at: {}".format(output_dir, e)
                self.log.error(error_message)
                raise SystemExit(error_msg)

        @jit(parallel=True)
        def file_check(self, input_file):
            '''
            Purpose: check that input file paths exist and are not empty
            input: file path
            output: assertion error if file not found or is empty
            '''
            error_message = "Check that {} exists and is not empty.".format(input_file)
            input_file = os.path.abspath(input_file)
            assert (os.path.isfile(input_file)) and (os.path.getsize(input_file) > 0), error_message

        @jit(parallel=True)
        def update_weight_for_row(self, row, graph):
            # enumerate targets to break hexamers into bases
            for pos, letter in enumerate(row.alignment):
                base_pos = row.start + pos
                graph[base_pos][letter] += 1  # for each nuc seen, weight of 1

    @jit(parallel=True)
    def get_path(self, grp):
        # setup dictionary by position
        graph = defaultdict(Counter)

        # update weights for each position observed
        grp.apply(self.update_weight_for_row, graph=graph,
                  axis=1)
        #
        # coerce results to dataframe
        base_df = pd.DataFrame.from_dict(graph, orient='index')
        base_df["-"] = 0.0
        base_df = base_df.fillna(0)

        # get nucleotide with highest count
        base_df["nuc"] = base_df.idxmax(axis=1)

        # calc number of features with ftm call to region
        base_df["sample_size"] = len(grp)

        # convert counts to allele frequencies
        base_df["A"] = round(base_df.A / base_df.sample_size * 100, 2)
        base_df["T"] = round(base_df.T / base_df.sample_size * 100, 2)
        base_df["G"] = round(base_df.G / base_df.sample_size * 100, 2)
        base_df["C"] = round(base_df.C / base_df.sample_size * 100, 2)

        return base_df

    @jit(parallel=True)
    def join_seq(self, grp):
        seq_list = []

        # sort group by position
        grp = grp.sort_values("pos", ascending=True)

        # join together molecule sequence
        molecule_seq = ''.join(grp.nuc.tolist()).strip()

        # append the sequence to seq_list
        seq_tup = (grp.region.unique()[0], molecule_seq)
        seq_list.append(seq_tup)
        #
        return seq_list

    @jit(parallel=True)


    def align_seqs(self, x):
        # parameters may need to be experimentally adjusted
        query = x.consensus_seq
        target = x.ref_seq

        # create pairwise global alignment object
        alignments = pairwise2.align.globalms(target,
                                              query,
                                              1, 0, -3, -.1,  # recommended penalties to favor snv over indel
                                              one_alignment_only=True)  # returns only best score

        # for each alignment in alignment object, return aligned sequence only
        for a in alignments:
            query, alignment, score, start, align_len = a
            pct_sim = round(score / align_len * 100, 2)

            return [x.region, alignment, x.ref_seq, pct_sim]


    @jit(parallel=True)
    def main(self):
        print("Counting reads per base...\n")
        # split reads up by base
        base_df = self.molecules.groupby(["region"]).apply(self.get_path).reset_index(drop=False)
        base_df.columns = ["region", "pos", "G", "C", "A", "T", "-", "nuc", "sample_size", "nuc_freq"]

        print("Determining consensus sequence...\n")
        seq_list = base_df.groupby("region").apply(self.join_seq)
        df = pd.DataFrame(seq_list).reset_index(drop=True)
        df1 = pd.DataFrame(df[0].tolist(), index=df.index)
        seq_df = pd.DataFrame(df1[0].tolist(), index=df1.index)
        seq_df.columns = ["region", "consensus_seq"]

        print("Adding reference sequences to align...\n")
        ## add reference sequence for each feature
        seq_df = seq_df.merge(self.fasta,
                              on=["region"],
                              how='left')

        # strip new line characters from sequences
        seq_df["ref_seq"] = seq_df.ref_seq.str.strip()
        seq_df["consensus_seq"] = seq_df.consensus_seq.str.strip()

        print("Aligning consensus sequences...\n")
        alignments = pd.DataFrame(seq_df.apply(self.align_seqs, axis=1))
        alignments.columns = ["align_list"]
        alignments[['region', 'alignment', 'ref_seq', 'pct_sim']] = pd.DataFrame(alignments.align_list.values.tolist(),
                                                                                 index=alignments.index).reset_index(
            drop=True)
        alignments = alignments.drop("align_list", axis=1)

        # delete after demo
        print(base_df.head())
        print(alignments)

        # close dask tasks
        self.client.close()

        # cleanup temp files and directories
        dask_folder = os.path.join(os.path.curdir, "dask-worker-space")
        shutil.rmtree(dask_folder)

        print("ShortStack pipeline complete.")

        # close dask tasks
        self.client.close()

        # cleanup temp files and directories
        dask_folder = os.path.join(os.path.curdir, "dask-worker-space")
        shutil.rmtree(dask_folder)


if __name__ == "__main__":

    # parse config file with default options
    config = cp.ConfigParser()
    configFilePath = args.config
    config.read(configFilePath)

    # instantiate sequencing module from sequencer.py
    consensus = Consensus(config_path=args.config,
                               molecules=config.get("options", "molecule_path"),
                               fasta_df=config.get("options", "fasta_path"),
                               out_dir=config.get("options", "output_dir"),
                               log_path=config.get("options", "log_path"))

    consensus.main()

# import sys, re, swifter, psutil, os
# import warnings
# from dask.dataframe.methods import sample
# from Bio.Nexus.Trees import consensus
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
# import logging
# import cython_funcs as cpy
# from numba import jit
# import numpy as np
# import pandas as pd
# from collections import defaultdict, Counter
# import dask.dataframe as dd
# from dask.dataframe.io.tests.test_parquet import npartitions
#
# # import logger
# log = logging.getLogger(__name__)
#
# class Consensus():
#
#     def __init__(self,
#                  molecule_df,
#                  ref_df,
#                  out_dir,
#                  cpus,
#                  client,
#                  today):
#
#         self.molecule_df = molecule_df
#         self.ref_df = ref_df
#         self.output_dir = out_dir
#         self.cpus = cpus
#         self.client = client
#         self.today = today
#
#     @jit(parallel=True)
#     def weigh_molecules(self, molecule_df):
#         '''
#         purpose: sum weights for each base/position
#         input: molecule_df created as output of sequenceer.py
#         output: molecule_df grouped by region with a sum of weights at eaach pos
#         '''
#
#         # calculate sample size for each region
#         size_df = dd.from_pandas(molecule_df[["FeatureID", "region"]].drop_duplicates(),
#                                  npartitions=self.cpus)
#         sample_sizes = size_df.groupby('region')['FeatureID'].count().reset_index()
#         sample_sizes.columns = ["region", "sample_size"]
#
#         # if molecule weight < 1 then set base to N
#         molecule_df["base"][molecule_df.weight == np.nan] = "N"
#         molecule_df["base"][molecule_df.weight < 1] = "N"
#
#         # set all molecule weights to N
#         molecule_df["weight"] = 1
#
#         # group by region and sum weights
#         molecule_df["base_weight"] = molecule_df.groupby(["region", "pos", "base"])["weight"].transform('sum')
#         molecule_df = molecule_df.drop(["weight", "FeatureID"],  axis=1)
#
#         # divide count by sample size to get frequency
#         molecule_df = dd.merge(molecule_df, sample_sizes,
#                                         on="region",
#                                         how="left")
#
#         return molecule_df
#
#     @jit(parallel=True)
#     def parse_consensus(self, molecule_df):
#         '''
#         purpose: convert format to one row per position for each molecule
#         input: molecule_df
#         output: final consensus output to output_dir/consensus_counts.tsv
#         '''
#
#         molecule_df = self.client.compute(molecule_df)
#         molecule_df = self.client.gather(molecule_df)
#
#         consensus_df = pd.pivot_table(molecule_df,
#                                       values = ['base_weight'],
#                                       index=['region','chrom',
#                                              'pos', 'ref_base',
#                                              'sample_size'],
#                                       columns='base',
#                                       fill_value=0).reset_index()
#
#         # sort and parse columns
#         consensus_df.columns = consensus_df.columns.droplevel(1)
#         consensus_df.columns = ["region", "chrom", "pos", "ref_base", "sample_size", "A", "C", "G", "N", "T"]
#         consensus_df = consensus_df[["region", "chrom", "pos", "ref_base", "A", "T", "G", "C", "N", "sample_size"]]
#         consensus_df = consensus_df.sort_values(by=["region", "pos"])
#
#         # find most frequent allele for each position
#         consensus_df["max_nuc"] = consensus_df[["A", "T", "G", "C", "N"]].idxmax(axis=1)
#
#         # save to a file
#         out_file = os.path.join(self.output_dir, "consensus_counts.tsv")
#         consensus_df.to_csv(out_file, sep="\t", index=False)
#
#         return consensus_df
#
#     @jit(parallel=True)
#     def find_MAF(self, consensus_df):
#
#         # find rows where the max_nuc does not equal ref_base
#         mafs = consensus_df[consensus_df.ref_base != consensus_df.max_nuc]
#
#         # calc row total calls
#         mafs["DP"] = mafs[["A", "T", "G", "C", "N"]].sum(axis=1)
#
#         # add placeholder for QV and NL values
#         mafs["QV"] = 30
#         mafs["NL"] = 5
#
#         # add (num_ref calls, num_alt calls)
#         mafs["AD"] =  tuple(zip(mafs.lookup(mafs.index,mafs.ref_base),
#                                 mafs.lookup(mafs.index,mafs.max_nuc)))
#
#         # calculate variant allele freq
#         mafs["VF"] = round(((mafs.lookup(mafs.index,mafs.max_nuc)/mafs["sample_size"]) * 100),2)
#
#         # add alt nuc
#         mafs["ALT"] = mafs["max_nuc"]
#         mafs["REF"] = mafs["ref_base"]
#         mafs["QUAL"]= "."
#         mafs["FILTER"] = "."
#
#         # parse into vcf format
#         mafs["INFO"] = "AD=" + mafs.AD.astype(str) + ";" + \
#                        "DP=" + mafs.DP.astype(str) + ";" + \
#                        "QV=" + mafs.QV.astype(str) + ";" + \
#                        "NL=" + mafs.NL.astype(str) + ";" + \
#                        "VF=" + mafs.VF.astype(str)
#
#         # remove unnecessary columns
#         mafs = mafs.drop(["DP", "NL", "AD", "VF", "QV",
#                           "A", "T", "G", "C", "N",
#                           "max_nuc", "ref_base"], axis=1)
#
#         # reorder columns
#         mafs  = mafs[["chrom", "pos", "REF", "ALT", "QUAL", "FILTER", "INFO"]]
#
#         return mafs
#
#     def make_vcf(self, maf_df):
#
#         # write VCF header to file
#         today_file = "{}_ShortStack.vcf".format(self.today)
#         output_VCF = os.path.join(self.output_dir, today_file)
#         with open(output_VCF, 'w') as vcf:
#             vcf.write("##fileformat=VCFv4.2\n")
#             vcf.write("##source=ShortStackv0.1.1\n")
#             vcf.write("##reference=GRCh38\n")
#             vcf.write("##referenceMod=file://data/scratch/reference/reference.bin\n")
#             vcf.write("##fileDate:{}\n".format(self.today))
#             vcf.write("##comment='Unreleased dev version. See Summer or Nicole with questions.'\n")
#             vcf.write("##FILTER=<ID=Pass,Description='All filters passed'>\n")
#             vcf.write("##INFO=<ID=GENE, Number=1, Type=String, Description='Gene name'>\n")
#             vcf.write("#CHROM    POS    ID    REF    ALT    QUAL    FILTER    INFO\n")
#
#         # parse dataframe for vcf output
#         maf_df.to_csv(output_VCF, index=False, sep="\t", header=False, mode='a')
#
#
#
#     def main(self):
#
#         # set all molecule weights to 1 and sum
#         consensus_weights = self.weigh_molecules(self.molecule_df)
#
#         # parse results and save to file
#         consensus_df = self.parse_consensus(consensus_weights)
#
#         # find rows where the major allele varies from the ref allele
#         maf_df = self.find_MAF(consensus_df)
#
#         self.make_vcf(maf_df)
#
#


