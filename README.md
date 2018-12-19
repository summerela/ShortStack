# ShortStack
NanoString ShortStack pipeline for sequencing and variant calling of per molecule short reads.  

Note: This program has currently only been optimized to run on the freya serever and the linux platform.  

## Versions
Please [check here](https://github.com/summerela/ShortStack/releases) for latest stable release. 

# Getting Started
Make sure you have the following files availble: 
- S6 file from imaging in either JSON or CSV format
- FASTA file of reference sequences
- Encoding file that maps basecalls to targets for each pool. A standard file can be found in ShortStack/base_files
- Optional VCF file to create mutation reference sequences if you are searching for a particular set of variants. 

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
- must be in JSON format
- CSV files will be converted using Joseph Phan's CSV to JSON converter
### encoding.txt 
- TSV format
- Must contain following columns:
    - PoolID  Target  BC
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
### All3spotters
All barcode events
### Bc_counts
Valid HD0s, diversity filtered
### Invalids 
Invalid barcodes (color code doesn’t exist in encoding file)
### Rawcounts
HD0/higher depending on cutoff
Contains ontargets, offtargets that have a target, cycle/location information
### No_ftm_calls
Features not called
### Ftm_calls
Features called, gene, counts

