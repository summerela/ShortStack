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
import time, psutil, os
from dask.distributed import Client
import dask.dataframe as dd
from pandas.io.json import json_normalize

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
                 log_path,
                 align_type="local",
                 match_points=1,
                 mismatch_penalty=-1,
                 gap_open_penalty=-2,
                 gap_extend_penalty=0):

        self.mem_limit = psutil.virtual_memory().available - 1000
        self.config_path  = config_path
        self.cpus = int(psutil.cpu_count())
        self.molecules = molecules
        self.today = time.strftime("%Y%m%d")
        self.output_prefix = str(self.today) + "_" + output_prefix
        self.out_dir = os.path.join(out_dir, self.output_prefix + "_consensus")
        self.fasta = pd.read_csv(fasta_df, sep="\t")
        self.client = Client(name="HexSemblerConsensus",
                             memory_limit=self.mem_limit,
                             n_workers=self.cpus - 2,
                             threads_per_worker=self.cpus / 2,
                             processes=True)
        self.align_type = align_type
        self.match_points = match_points
        self.mismatch_penalty = mismatch_penalty
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty

        # set directory path
        self.log_path = os.path.abspath(log_path)
        self.output_dir = os.path.join(self.out_dir, "consensus")

        # create output dir if not exists
        self.create_outdir(self.output_dir)

        ### Setup logging ###
        now = time.strftime("%Y%d%m_%H:%M:%S")
        today_file = "{}_consensus.log".format(self.output_prefix)
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
        base_df = base_df.reset_index(drop=False)
        base_df = base_df.rename(columns={"index":"pos"})
        base_df = base_df.fillna(0)

        # cover any bases not found
        col_list = ["A", "T", "G", "C", "-"]
        for col in col_list:
            if col not in base_df:
                base_df[col] = 0.0

        # get nucleotide with highest count
        base_df["nuc"] = base_df[["A", "T", "G", "C", "U", "-"]].idxmax(axis=1)

        # calc number of features with ftm call to region
        base_df["sample_size"] = len(grp)

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

        if self.align_type == "local":

            # create pairwise global alignment object
            alignments = pairwise2.align.localms(x.ref_seq,
                                                  x.consensus_seq,
                                                  self.match_points,
                                                  self.mismatch_penalty,
                                                  self.gap_open_penalty,
                                                  self.gap_extend_penalty,  # recommended penalties to favor snv over indel
                                                  one_alignment_only=True)  # returns only best score

        else:
            # create pairwise global alignment object
            alignments = pairwise2.align.globalms(x.ref_seq,
                                                 x.consensus_seq,
                                                 self.match_points,
                                                 self.mismatch_penalty,
                                                 self.gap_open_penalty,
                                                 self.gap_extend_penalty,  # recommended penalties to favor snv over indel
                                                 one_alignment_only=True)  # returns only best score

        # for each alignment in alignment object, return aligned sequence only
        for a in alignments:
            target, alignment, score, start, align_len = a
            pct_sim = round(score / align_len * 100, 2)

            return [x.chrom, x.start, x.region, alignment, x.ref_seq, pct_sim]

    @jit(parallel=True)
    def find_variants(self, row):
        # create dictionary of variants for each position
        var_graph = defaultdict(defaultdict)
        if row.pct_sim != 100:
            for idx, (alt, ref) in enumerate(zip(row.alignment, row.ref_seq)):
                if alt != ref:
                    var_graph[str(row.chrom)+"$"+str(row.start_pos)+"$"+str(row.region)][idx] = {"ref":ref, "alt":alt}

        return var_graph


    def reshape_graph(self, graph):

        # flatten dictionary levels to create dataframe
        reform = {(level1_key, level2_key): values
            for level1_key, level2_dict in graph.items()
            for level2_key, values in level2_dict.items()}

        result = []
        for k, v in reform.items():
            for x,y in v.items():
                result.append((k[1], x,y))

        df = pd.DataFrame(result)
        df.columns = ["region", "pos", "vars"]

        return df

    def reshape_df(self, df):

        # flatten dictionary into columns for ref and alt, one per row
        df = pd.concat([df, json_normalize(df["vars"])], axis=1)
        df = df.drop("vars", axis=1)
        df[["chrom", "start", "region"]] = df['region'].str.split('$',expand=True)
        df["pos"] = df.pos.astype(int) + df.start.astype(int)
        df = df.drop("start", axis=1)
        df = df.sort_values(["region", "pos"])

        # group together indels by consecutive position within region
        s = df.groupby('region').pos.apply(lambda x: x.diff().fillna(1).ne(1).cumsum())
        var_list = []
        for idx, group in df.groupby(['region', s], sort=False):
            chrom = group.chrom.unique()[0]
            start_pos = min(group.pos)
            region = idx[0]
            ref = ''.join(group.ref.to_list())
            alt = ''.join(group.alt.to_list())
            var_list.append([chrom, start_pos, region, ref, alt])

        var_df = pd.DataFrame(var_list)
        var_df.columns = ["#CHROM", "POS", "ID", "REF", "ALT"]
        # add placeholder for future metrics
        var_df["QUAL"] = "."  # placeholder for real quality score
        var_df["FILTER"] = "."
        var_df["QV"] = "."

        return var_df

    @jit(parallel=True)
    def make_vcf(self, var_df, base_df):

        var_df["ID"] = var_df.ID.astype(str)
        var_df["POS"] = var_df.POS.astype(int)
        base_df["region"] = base_df.region.astype(str)
        base_df["pos"] = base_df.pos.astype(int)

        # pull in count information
        vcf_df = var_df.merge(base_df,
                              left_on=['ID', 'POS'],
                              right_on=['region', 'pos'],
                              how='left')

        # parse into vcf format
        vcf_df["INFO"] = "DP=" + vcf_df.DP.astype(str) + ";" + \
                       "QV=" + vcf_df.QV.astype(str) + ";" + \
                       "VF=" + vcf_df.AF.astype(str)

        # remove unnecessary columns
        vcf_df = vcf_df.drop(["DP", "AF", "QV",
                          "A", "T", "G", "C", "U", "-",
                        'region', 'pos', 'nuc', 'sample_size'], axis=1)
        return vcf_df

    def write_vcf(self, vcf_df):

        vcf_file = os.path.join(self.out_dir, self.output_prefix + ".vcf")

        # delete file if file exists
        if os.path.exists(vcf_file):
            os.remove(vcf_file)

        with open(vcf_file, 'a+') as vcf:
            vcf.write("##fileformat=VCFv4.2\n")
            vcf.write("##source=ShortStackv0.1.1\n")
            vcf.write("##reference=GRCh38\n")
            vcf.write("##referenceMod=file://data/scratch/reference/reference.bin\n")
            vcf.write("##fileDate:{}\n".format(self.today))
            vcf.write("##comment='Command line arg: python3 consensus.py -c path/to/consenus_config.txt.'\n")
            vcf.write("##INFO=<ID=DP,Number=1,Type=Integer,Description='Total Depth'>\n")
            vcf.write("##INFO=<ID=QV,Number=1,Type=Integer,Description='Quality Value'>\n")
            vcf.write("##INFO=<ID=VF,Number=1,Type=Integer,Description='Variant Frequency'>\n")
            vcf.write("##FILTER=<ID=Pass,Description='All filters passed'>\n")
            vcf.write("##INFO=<ID=GENE, Number=1, Type=String, Description='Gene name'>\n")
            vcf.write("#CHROM    POS    ID    REF    ALT    QUAL    FILTER    INFO\n")

        vcf_df.to_csv(vcf_file, index=False, sep="\t", header=False, mode='a')

        print("VCF saved as {}".format(vcf_file))

    @jit(parallel=True)
    def main(self):

        # read in molecule alignments
        print("Aggregating molecule files...\n")
        mol_file_list= self.get_molecules(self.molecules)
        mol_df = dd.read_csv(mol_file_list, sep="\t").compute(ncores=self.cpus)

        print("Counting reads per base...\n")
        # split reads up by base
        base_df = mol_df.groupby(["region"]).apply(self.get_path).reset_index(drop=False)
        base_df = base_df[["region", "pos", "A", "T", "G", "C", "U", "-", "nuc", "sample_size"]]
        # calculate depth at each base
        base_df["DP"] = round(base_df[["A", "T", "G", "C", "U", "-"]].sum(axis=1), 2)
        # calculate allele freq
        base_df["AF"] = round(((base_df.lookup(base_df.index, base_df.nuc) / base_df["DP"]) * 100), 2)

        # write base call counts to file
        base_file = os.path.join(self.out_dir, self.output_prefix + "_counts.tsv")
        base_df.to_csv(base_file, sep="\t", index=False)
        print("Base counts saved to {}".format(base_file))

        print("Determining consensus sequence...\n")
        seq_list = base_df.groupby("region").apply(self.join_seq)
        df = pd.DataFrame(seq_list).reset_index(drop=True)
        df1 = pd.DataFrame(df[0].tolist(), index=df.index)
        seq_df = pd.DataFrame(df1[0].tolist(), index=df1.index)
        seq_df.columns = ["region", "consensus_seq"]

        print("Adding reference sequences to align...\n")
        # add reference sequence for each feature
        seq_df = seq_df.merge(self.fasta,
                              on=["region"],
                              how='left')

        # strip new line characters from sequences
        seq_df["ref_seq"] = seq_df.ref_seq.str.strip()
        seq_df["consensus_seq"] = seq_df.consensus_seq.str.strip()

        print("Aligning consensus sequences...\n")
        alignments = pd.DataFrame(seq_df.apply(self.align_seqs, axis=1))
        alignments.columns = ["align_list"]
        alignments[['chrom', 'start_pos','region', 'alignment', 'ref_seq', 'pct_sim']] = pd.DataFrame(alignments.align_list.values.tolist(),
                            index=alignments.index).reset_index(drop=True)
        alignments = alignments.drop("align_list", axis=1)

        alignment_out = os.path.join(self.output_dir,
                        self.output_prefix + "_" + self.today + "_consensus.tsv")
        alignments.to_csv(alignment_out, sep="\t", index=False)
        print("Consensus alignments saved to:\n {}\n".format(alignment_out))

        print("Generating VCF file...\n")
        var_graph = alignments.apply(self.find_variants, axis=1)
        var_df1 = self.reshape_graph(var_graph)
        var_df = self.reshape_df(var_df1)
        vcf_df = self.make_vcf(var_df, base_df)
        self.write_vcf(vcf_df)

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
                               log_path=config.get("options", "log_path"),
                               align_type=config.get("alignment", "align_type"),
                               match_points=config.getint("alignment", "match_points"),
                               mismatch_penalty=config.getint("alignment", "mismatch_penalty"),
                               gap_open_penalty=config.getint("alignment", "gap_open_penalty"),
                               gap_extend_penalty=config.getint("alignment", "gap_extend_penalty"))

    consensus.main()