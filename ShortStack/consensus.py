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
from dask import delayed

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
                 out_dir,
                 output_prefix,
                 log_path):

        self.mem_limit = psutil.virtual_memory().available - 1000
        self.cpus = int(psutil.cpu_count())
        self.molecules = molecules
        self.output_prefix = output_prefix
        self.fasta = pd.read_csv(fasta_df, sep="\t")
        self.out_dir = out_dir
        self.client = Client(name="HexSemblerConsensus",
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
        today_file = "{}_{}_consensus.log".format(self.output_prefix, self.today)
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

    def create_outdir(self, output_dir):
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
            raise SystemExit(error_message)

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
    def get_molecules(self, mol_dir):
        mol_list = []
        for p, d, f in os.walk(mol_dir):
            for file in f:
                if file.endswith('_molecules.tsv'):
                    mol_list.append(os.path.join(p, file))
        return mol_list

    @jit(parallel=True)
    def update_weight_for_row(self, row, graph):
        # enumerate targets to break hexamers into bases
        for pos, letter in enumerate(row.alignment):
            base_pos = row.start + pos
            graph[base_pos][letter] += 1  # for each nuc seen, weight of 1

    @jit(parallel=True)
    def get_path(self, grp):

        # pull out starting position for region
        start_pos = grp.start.unique()[0]

        # setup dictionary by position
        graph = defaultdict(Counter)

        # update weights for each position observed
        grp.apply(self.update_weight_for_row, graph=graph,
                  axis=1)

        # coerce results to dataframe
        base_df = pd.DataFrame.from_dict(graph, orient='index')
        base_df["pos"] = base_df.index.astype(int) + start_pos

        # cover any bases not found
        col_list = ["A", "T", "G", "C", "-"]
        for col in col_list:
            if col not in base_df:
                base_df[col] = 0.0

        # get nucleotide with highest count
        base_df["nuc"] = base_df[["A", "T", "G", "C", "-"]].idxmax(axis=1)

        # calc number of features with ftm call to region
        base_df["sample_size"] = len(grp)
        base_df = base_df.fillna(0)


        # convert counts to allele frequencies
        base_df["A"] = round((base_df["A"] / base_df.sample_size) * 100, 2)
        base_df["T"] = round((base_df["T"] / base_df.sample_size) * 100, 2)
        base_df["G"] = round(base_df["G"] / base_df.sample_size * 100, 2)
        base_df["C"] = round(base_df["C"] / base_df.sample_size * 100, 2)

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

        # read in molecule alignments
        print("Aggregating molecule files...\n")
        mol_file_list= self.get_molecules(self.molecules)
        mol_df = dd.read_csv(mol_file_list, sep="\t").compute(ncores=self.cpus)

        print("Counting reads per base...\n")
        # split reads up by base
        base_df = mol_df.groupby(["region"]).apply(self.get_path).reset_index(drop=False)
        base_df = base_df[["region", "pos", "A", "T", "G", "C", "-", "nuc", "sample_size"]]

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
        alignment_out = os.path.join(self.output_dir,
                        self.output_prefix + "_" + self.today + "_consensus.tsv")
        alignments.to_csv(alignment_out, sep="\t", index=False)

        print("Consensus alignments saved to:\n {}\n".format(alignment_out))

        # close dask tasks
        self.client.close()

        print("Consensus sequencing complete.")


if __name__ == "__main__":

    # parse config file with default options
    config = cp.ConfigParser()
    configFilePath = args.config
    config.read(configFilePath)

    # instantiate sequencing module from sequencer.py
    consensus = Consensus(config_path=args.config,
                               molecules=config.get("options", "molecule_files"),
                               fasta_df=config.get("options", "fasta_dataframe"),
                               out_dir=config.get("options", "output_dir"),
                               output_prefix=config.get("options", "output_prefix"),
                               log_path=config.get("options", "log_path"))

    consensus.main()