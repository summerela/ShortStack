# ShortStack
NanoString ShortStack assembly and variant calling tool for target regions. 

Note: This program has only been optimized to run on the freya serever. 

## Versions
### Current
- 0.1.1 contains only FTM, not complete sequencing and variant calling

# Getting Started
Make sure you have the following files availble: 
- S6 file from imaging in either JSON or CSV format
- FASTA file of reference sequences
- Encoding file that maps basecalls to targets for each pool. A standard file can be found in ShortStack/base_files
- Optional VCF file to create mutation reference sequences if you are searching for a particular set of variants. 

## Installation
For internal use, a current version will be maintained on the bioinformatics SVN at: 
http://svn/repos/BIS/tools/ShortStack

- Copy this directory to whatever server you would like to run ShortStack from. 
- From the command line where you copied the directory, install the requirements by typing: 

     pip3 install -r requirements.txt  
- You should now be able to run the program from the command line

## Required Input: 
### s6 file:
- must be in JSON format
- CSV files will be converted using Joseph Phan's CSV to JSON converter
### barcode.txt 
- TSV format
- Must contain following columns:

    PoolID  Target  BC
- PoolID = FOV_x_y coords as Feature ID
- Target = Target sequence
- BC = nunberical basecall from imaging
### target.fa
- fasta header must be colon delimited
- header must contain columns as follows: 

    id:chrom:start:stop:build:strand
- strand should be entered as either "+" or "-"

## Optional Input:
### VCF file
- VCF4.3 format
- INFO field must contain STRAND= followed by + or - 

## To run: 
    python run_shortstack.py -c path/to/config.txt

## Output
- qc_metrcis.tsv: target reads that have been filtered out for containing:
    - UC – uncalled bases where any position in the 3 spotter contains a 0
    - QC – where any position in the 3 spotter quality score (fake QC score at this point) below qc_threshold
- normalized_counts.tsv = counts per feature of each gene/target, normalized for multi-mapping
- ftm_calls = ftm calls with zscore/deltaZ between top 2
- run_report.txt – contains file paths to all input files and user options
