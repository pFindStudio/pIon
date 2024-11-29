# pIon

## 1. Introduction

The ever-growing field of covalent drug discovery and chemoproteomics has fueled a need for bioconjugation methods with high selectivity in a native context. Given that the sheer number of functional groups present, achieving residue-specificity in biological systems remains a challenge. In order to identify the best residue-specific bioconjugation method, numerous small molecule models and/or purified proteins have been employed for rigorous method validation in terms of selectivity, efficiency and stability. Such in vitro experiments therefore became increasingly time consuming, yet they were unable to enumerate all functional groups present in complex, native biological systems. pIon is a computational tool that enables a cost-efficient pipeline for high-throughput evaluation of residue-specific bioconjugation chemistries. It starts with a rapid experimental phase, in which the proteome conjugated by a reactive probe is chemically coded for generating diagnostic report ions in tandem mass spectrometry analysis. The resulting MS data can be directly imported into pIon, which automatically calculates the accurate modification masses derived from a tested probe as well as the corresponding residue preferences. Thus, pIon has the potential to become a valuable option for routine evaluation of bioconjugation chemistries, thereby driving the field of bioconjugation chemistry to unprecedented dimensions and interfacing the worlds of biological and synthetic chemistry.

## 2. Download

You can download the software and demo dataset from the official **pIon website**:
[pIon Download Page](http://pfind.org/software/pIon/index.html)

The user guide is also available on the [user guide webpage](http://pfind.org/software/pIon/index.html).

## 3. Requirements

For python version:

- new version of pFind
- Python 3
- matplotlib
- pandas

## 4. Fast Usage

This section provides a quick overview of how to get started with **pIon**.

### 4.1. Set the Parameter File

The main configuration file for **pIon** is `pIon.cfg`. At a minimum, you need to set the paths to your FASTA file (protein sequence database) and your MS/MS data file(s). **pIon** supports the following MS/MS data formats: **RAW**, **MZML**, or **MGF**.

Example configuration file (`pIon.cfg`):

```
# pIon general parameter settings
# Path to the output directory where results will be saved
output_path=F:/pIon/result1

# Path to the protein sequence database (FASTA file)
fasta_path=F:/pIon/Protein_seq_database/Homo_sapiens_uniprot_canonical_20395_entries_20210516.fasta

# Format of the MS data (RAW, MZML, or MGF)
msmstype=MGF

# The number of MS/MS data files and their paths
msmsnum=1
msmspath1=H:/pIon/pIon1/demo_dataset/IPM_demo.mgf
# Additional MSMS files can be added here, e.g.
# msmspath2=H:/pIon/pIon1/demo_dataset/another_example.mgf
```

Make sure to adjust the paths according to your local setup.

### 4.2. Run the Software

Once your configuration file is set up, you can run **pIon** from the command line (CMD) as follows:

```
python pion.py
```

This will initiate the program using the settings specified in the `pIon.cfg` file.

For more detailed instructions and advanced usage, please refer to the complete **user guide** available on the [pIon website](http://pfind.org/software/pIon/index.html).

## 5. License

This software is provided for educational and research purposes. 

------

