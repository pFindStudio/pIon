'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2022-06-28 12:07:49
LastEditTime: 2024-09-28 14:37:55
FilePath: \pChem2\pchem.py
Description: Do not edit
'''
import os
import shutil
import time
import glob
from argparse import ArgumentParser
from blind_search import blind_search
from close_identify import new_close_search
from utils import (parameter_file_read, delete_file, parameter_file_read_i)
from ion_modification_filter import (blind_res_read, mgf_read_for_ion_pfind_filter,
                                      get_modification_from_result, close_ion_learning,
                                      heatmap_ion)

def run():
    current_path = os.getcwd()
    print('Welcome to use pChem! (version 2.1)')

    # Parse command-line arguments
    args = parse_arguments()
    parameter_dict = load_parameters(current_path)

    # Validate paths in parameter dictionary
    if not validate_paths(parameter_dict):
        return

    # Clean up existing output directories
    clean_output_paths(parameter_dict['output_path'])

    # Perform blind search
    start_time = time.time()
    blind_search(current_path)
    blind_time = time.time()

    # Determine whether to use close search based on isotope labeling
    parameter_dict['use_close_search'] = parameter_dict['isotope_labeling'] == 'True'

    # Perform close search if applicable
    if parameter_dict['use_close_search']:
        new_close_search(current_path)
        close_time = time.time()
        print_time_costs(start_time, blind_time, close_time)
    else:
        print_time_costs(start_time, blind_time)

    # Clean up temporary files
    delete_file(current_path, 'psite.txt')

    # Read parameters again for ion analysis
    parameter_dict = parameter_file_read_i(os.path.join(current_path, 'pChem.cfg'))
    if parameter_dict['ion_labeling'] == 'True' and parameter_dict['msmstype'] != "TIMS":
        process_ions(parameter_dict, current_path)

    # Delete unnecessary files
    for filename in ['modification-null.ini', 'modification-new.ini']:
        delete_file(current_path, filename)

def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--fasta_path", type=str, default='None', help='Path to the fasta file')
    parser.add_argument("--msms_path", type=str, default='None', help='Path to the msms file')
    parser.add_argument("--output_path", type=str, default='None', help='Path to the result file')
    return parser.parse_args()

def load_parameters(current_path):
    """Load parameters from the configuration file."""
    cfg_path = os.path.join(current_path, 'pChem.cfg')
    return parameter_file_read(cfg_path)

def validate_paths(parameter_dict):
    """Check the existence of specified paths."""
    if not os.path.exists(parameter_dict['output_path']):
        print('Output path does not exist!')
        return False
    if not os.path.exists(parameter_dict['fasta_path']):
        print('Fasta path does not exist!')
        return False
    for ms_path in parameter_dict['msms_path']:
        if not os.path.exists(ms_path.split('=')[1].strip()):
            print('MSMS path does not exist!')
            return False
    return True

def clean_output_paths(output_path):
    """Remove existing output directories to start fresh."""
    for folder in ['source/pParse', 'reporting_summary', 'source/close', 'source/blind']:
        path = os.path.join(output_path, folder)
        if os.path.exists(path):
            shutil.rmtree(path)

def print_time_costs(start_time, blind_time, close_time=None):
    """Print the time costs of the search processes."""
    print('[Time cost]')
    print('Blind search time (s):', round(blind_time - start_time, 1))
    if close_time:
        print('Restricted search time (s):', round(close_time - blind_time, 1))

def process_ions(parameter_dict, current_path):
    """Process ions based on the specified parameters."""
    filter_frequency = parameter_dict['filter_frequency']
    blind_res_path = os.path.join(parameter_dict['output_path'], "source/blind/pFind-Filtered.spectra")
    blind_res = blind_res_read(blind_res_path)
    pp_path = os.path.join(parameter_dict['output_path'], "source/pParse")
    mgf_path_list = glob.glob(os.path.join(pp_path, "*.mgf"))

    blind_res_raw_dict = organize_blind_results(blind_res, mgf_path_list)
    mass_spectra_dict = read_mass_spectra(mgf_path_list, blind_res_raw_dict)

    pchem_summary_path = os.path.join(parameter_dict['output_path'], "reporting_summary/pChem.summary")
    modifications = get_modification_from_result(pchem_summary_path)

    ion_analysis(parameter_dict, current_path, blind_res, mass_spectra_dict, modifications)

def organize_blind_results(blind_res, mgf_path_list):
    """Organize blind results by sample name."""
    blind_res_raw_dict = {}
    for mgf_path in mgf_path_list:
        t_name = os.path.basename(mgf_path).split('.')[0].rsplit('_', 1)[0]
        blind_res_raw_dict[t_name] = [line for line in blind_res if t_name in line]
    return blind_res_raw_dict

def read_mass_spectra(mgf_path_list, blind_res_raw_dict):
    """Read mass spectra from MGF files and associate with blind results."""
    mass_spectra_dict = {}
    for mgf_path in mgf_path_list:
        t_name = os.path.basename(mgf_path).split('.')[0].rsplit('_', 1)[0]
        mass_spectra_dict.update(mgf_read_for_ion_pfind_filter(mgf_path, blind_res_raw_dict[t_name]))
    return mass_spectra_dict

def ion_analysis(parameter_dict, current_path, blind_res, mass_spectra_dict, modifications):
    """Analyze ions and generate results."""
    pchem_output_path = os.path.join(parameter_dict['output_path'], "reporting_summary")
    ion_type = str(round(float(parameter_dict['ion_type']), 3))
    ion_filter_mode = 2
    ion_rank_threshold = 1
    ion_relative_mode = 1

    pchem_summary_path1 = close_ion_learning(pchem_output_path, current_path, ion_type, *modifications, 
                                             blind_res, mass_spectra_dict, parameter_dict['ion_filter_ratio'])
    psm_filter(pchem_summary_path1, parameter_dict['filter_frequency'])
    heat_map_draw = heatmap_ion(pchem_output_path, pchem_summary_path1)

def psm_filter(pchem_summary_path, filter_frequency):
    """Filter PSMs based on specified frequency criteria."""
    with open(pchem_summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    re_lines = [lines[0]]
    max_psm = max(int(line.split("\t")[5]) for line in lines[1:] if line.strip())
    
    for line in lines[1:]:
        if line.strip():
            t_psms = int(line.split("\t")[5])
            if t_psms > filter_frequency * max_psm / 100:
                re_lines.append(line)

    # Write filtered results back to the summary file
    with open(pchem_summary_path, 'w', encoding='utf-8') as f1:
        for index, line in enumerate(re_lines):
            if index == 0:
                f1.write(line)
            else:
                line_res = line.split("\t")
                line_res[0] = str(index)
                f1.write("\t".join(line_res))
        f1.write('\n')

if __name__ == "__main__":
    run()
    os.system("pause")

