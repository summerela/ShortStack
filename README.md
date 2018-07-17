# ShortStack
NanoString ShortStack assembly and variant calling tool for target regions

## Required Input: 
### s6 file:
    - must be in JSON format
### barcode.txt 
    - must be tsv
    - must contain following columns
    - PoolID  Target  BC
### target.fa
    - fasta header must be colon delimited
    - header must contain columns as follows
    - id:chrom:start:stop:build:strand
    - strand should be entered as either "+" or "-"

## Optional Input:
### VCF file
    - VCF4.3 format
    - INFO field must contain STRAND= followed by + or - 

## To run: 
python run_shortstack.py -c path/to/config.txt