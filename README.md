# HexSembler Pipeline
NanoString HexSembler pipeline for sequencing and variant calling of single molecule short reads.  

Note: This program has currently only been optimized to run on the freya serever and the linux platform.  

## Versions
Please [check here](https://github.com/summerela/ShortStack/releases) for latest stable release. 

# Getting Started
Make sure you have the following: 
- Path to folder containing one S6 file for each FOV in CSV format. S6 files from other runs should not be in this folder.  
- FASTA file of reference sequences
- Encoding file that maps basecalls to targets for each pool. A standard file can be found in ShortStack/base_files
- Optional VCF file to create mutation reference sequences for supervised sequencing

## Installation

All installation instructions assume you have already installed a working copy of python. 

### Windows users ###
If you are trying to run this on Windows (insert sad face) you may first need to install Visual C++ build tools and add cl.ext to your environmental path variables, as [discussed here](https://stackoverflow.com/questions/41724445/python-pip-on-windows-command-cl-exe-failed/41724634). 

1. [Install Visual C++ build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017)
2. Use the Use the Visual C++ Command Prompt (You can find it in the Start Menu under the Visual Studio folder). Install using pip as detailed below. 
    

### Install from github master branch 
- stable release, but most likely not the most updated version

To install the lastest development version, from your command line type: 

    pip install git+https://github.com/summerela/ShortStack.git@master --user

### Install from github dev branch 
- latest updates, not necessarily tested or stable

To install the lastest development version, from your command line type: 

    pip install git+https://github.com/summerela/ShortStack.git@dev --user

## Required Input: 
### config.txt
Please see required options [here](https://github.com/summerela/ShortStack/blob/dev/ShortStack/example_files/config.txt)
### s6 file:
- must be in CSV format
- CSV files will be filtered and converted to parquet format before running FTM and molecule sequencing
### encoding.txt 
- TSV format
- Must contain following columns:
    - PoolID: int specifying which pool barcode was run in
    - Target: string of target nucleotides for each input probe
    - BC: matching base call for target
    - bc_length: int specifying length of target to minimize downstream length calc
### target.fa
- fasta header must be colon delimited
- header must contain columns as follows: 
     id:chrom:start:stop:build:strand
- strand should be entered as either "+" or "-"
- standard FASTA format

## Optional Input:
### VCF file
- VCF4.3 format
- INFO field must contain STRAND= followed by + or - 

## How to run the pipeline 
### 1. Convert the S6 CSV to parquet: 
  
`python3 csv_to_parquet.py input_csv_path output_parquet_dir`

#### Output = folder of parquet files

### 2. Fill out your config file (usually called config.txt), as follows: 
- do not use quotes around strings
- leave blank for default options
- do not edit the default section

Fill out the following options in the [user_facing_options] section: 
- output_dir = Path to directory where you would like to place output
- file_prefix= prefix that will be added to output results along with today's data
- input_s6= path to file directory containing all of the S6 files you would like to analyze for this run. Results will be aggregated together for consensus variant calling. 
- target_fa= path to your fasta file of target regions
- encoding_table= path to the encoding file that maps your targets to nucleotides
- mutation_vcf= enter path to a VCF file for supervised sequencing, or enter None for unsupervised
- ftm_only= set to True if you only want to run FTM
- logfile_path=path to your log file to document program events

#### output = config file 

### 3. Run Molecule Sequencing
Next just point the HexSembler script to your config file and run molecule sequencing. 

Type the following argument into your command line: 
`python run_shortstack.py -c path/to/config.txt`

 #### Output 
- Bc_counts: Parquet folder contatining counts for all valid HD0s, diversity filtered
- all_counts: parquet folder containing coutns for all valid barclodes up to HD max to FMT called region
- Invalids:  Parquet folder of invalid barcodes (color code doesn’t exist in encoding file)
- Rawcounts: HD0/higher depending on cutoff
    - Contains ontargets, offtargets that have a target, cycle/location information
- No_ftm_calls: Features where the pipeline was unable to determine an ROI
- ftm_calls: Features called for each region 
- base_counts: per base allele frequencies to FTM called region used to crate consensus sequence
- fasta.tsv: fasta file containing both WT and any input variant sequences. Fed to conesnsus.py pipeline. 
- molecules.tsv: per molecule aligned consensus sequence to the FTM called region

### 4. Run Consensus Variant Calling
The final step requires you to create another config file with paths to the folder containing the parquet files for each FOV, as well as the molecule sequence files and the output fasta.tsv from Step 3. 

#### consensus_config.txt
[options]
- molecule_files= path to folder contatining all the molecules.tsv files output by Step 3 for each FOV
- fasta_dataframe= path to a fasta.tsv created in step 3 of the pipeline (there will be one for each FOV, but they should all be - identical, point to any one of them)
- output_prefix= refix that will be added to output results along with today's data
- output_dir= path to directory where you would like to place the output files
- log_path=path to your log file to document program events

To change the alignment parameters, follow the guidlines [here](https://biopython.org/DIST/docs/api/Bio.pairwise2-module.html) and update the consensus_config.txt "align" section as shown below: 
[align]
- align_type=local  
- match_points=1  
- mismatch_penalty=-1  
- gap_open_penalty=-2  
- gap_extend_penalty=0  

#### Output = 
- counts.tsv: per base allele frequencies per ROI 
- .vcf: variant calls
- consensus.tsv: consensus sequences
