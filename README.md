# ShortStack
NanoString ShortStack assembly and variant calling tool for target regions

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
