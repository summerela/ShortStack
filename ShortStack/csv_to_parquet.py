'''
*** CSV to S6 converter for HexSembler Software ***

purpose: converts imaging CSV results to parquet format
    filtering out non-3spotters and any spots containing 0
input: S6 CSV file from imaging
output: s6_parquet folder containing parquet files compressed with snappy format

To Run: 
python3 csv_to_parquet.py input_csv_path output_parquet_dir

Dependencies:
- to be run only on Freya or servers with fastparquet and snappy installed
- python3.6
- Dask
- Pandas
- Numba

'''
from dask.distributed import Client
from dask.dataframe.core import no_default
import dask.dataframe as dd
import sys, os
from numba import jit
import pandas as pd
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore") # ignore dask warnings
import psutil

# read file in parallel using Dask
@jit(parallel=True)
def read_s6(input_s6):
    print("Reading in S6 file {}.".format(input_s6))
    
    # read in S6 file and create feature id's
    df = dd.read_csv(input_s6, dtype='object',
                     blocksize='100000MB',
                     error_bad_lines=False)
    
    # Remove cheeky comma column, if it exists
    df = df.loc[:,~df.columns.str.contains('^Unnamed')]
    # Remove whitespace from column headers
    df.columns = df.columns.str.strip()

    # pull FOV out of file name, assumes standard output format
    if "fov" in df.columns:
        pass
    else:
        df["fov"] = os.path.basename(input_s6).split("_")[4].split("0")[-1]

    df["FeatureID"] = df["fov"].astype(str) + "_" + df["x"].astype(str) + "_" + df["y"].astype(str)

    # # if these columns are in the file, remove after no longer needed
    drop_list = ["Features", "fov", "x", "y"]
    for col in drop_list:
        if col in df.columns:
            df.drop([col], axis=1)
    
    return df

@jit(parallel=True)
def melt(frame, id_vars=None, value_vars=None, var_name=None,
     value_name='value', col_level=None):
    return frame.map_partitions(pd.melt, meta=no_default, id_vars=id_vars,
                                value_vars=value_vars,
                                var_name=var_name, value_name=value_name,
                                col_level=col_level, token='melt')

@jit(parallel=True)
def parse_s6(input_s6):
    print("Parsing S6 file.")

    # expand basecalls to one row per feature
    s6_df = melt(input_s6,
                 id_vars="FeatureID",
                 value_name='BC')

    s6_df['BC'] = s6_df['BC'].astype('int')

    # filter out invalid reads
    s6_df = s6_df[s6_df.BC != 0]
    s6_df = s6_df[s6_df.BC > 111111]

    s6_df['BC'] = s6_df['BC'].astype('str')

    # filter out rows where basecall contains uncalled bases of 0
    s6_df = s6_df[~s6_df.BC.str.contains("0")]

    # split up pool and cycle info
    s6_df["cycle"] = s6_df['variable'].str.partition('P')[0]
    s6_df["pool"] = s6_df['variable'].str.partition('P')[2]

    # drop variable column
    s6_df = s6_df.drop("variable", axis=1)

    return s6_df



if __name__=="__main__":

    mem_limit = psutil.virtual_memory().free - 1000

    # setup dask distributed cluster
    client = Client(n_workers=30,
                    threads_per_worker=15,
                    memory_limit=mem_limit)

    # check for required args
    arguments = len(sys.argv) - 1
    if arguments < 2:
        raise SystemExit("Please enter the path to the folder containing your csv files(s) and desired output directory.")

    # gather file paths
    csv_dir = sys.argv[1]
    outdir = sys.argv[2]
      
    # read in the csv
    file_counter = 0

    for file in os.listdir(csv_dir):
        if file.endswith("_S6.csv"):
            full_path = os.path.join(csv_dir, file)
            s6_df = read_s6(full_path)

            # # filter and output as parquet
            s6_df = parse_s6(s6_df)
            file_counter += 1
            #
            # # write out to parquet
            file_name = file.split(".")[0] + "_" + str(file_counter)
            outfile = os.path.join(outdir, file_name)

            s6_df.to_parquet(outfile,
                             append=False,
                             engine='fastparquet',
                             compression='snappy')

    print("Conversion complete.")