'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2024-11-29 14:08:21
LastEditTime: 2024-11-29 15:29:05
FilePath: \code1\pIon\pion.py
Description: Do not edit
'''
from re import T
from blind_search import blind_search, psm_filter
from close_identify import new_close_search
from utils import parameter_file_read, pfind_path_find, delete_file, parameter_file_read_i
from ion_modification_filter import blind_res_read, get_ms2_ms1_scan_dict, mgf_read_for_ion_pfind_filter2, get_modification_from_result, close_ion_learning, heatmap_ion
import os
import shutil
from argparse import ArgumentParser
import time
import glob


def run():
    """
    Main function to execute the pIon pipeline. This function handles the setup, file checks,
    execution of blind search, close search (if applicable), and ion labeling. 
    """
    current_path = os.getcwd()
    print('Welcome to use pChem! (version 2.1)')

    parser = ArgumentParser()
    parser.add_argument("--fasta_path", type=str,
                        default='None', help='path to the fasta file')
    parser.add_argument("--msms_path", type=str,
                        default='None', help='path to the msms file')
    parser.add_argument("--output_path", type=str,
                        default='None', help='path to the result file')

    args = parser.parse_args()

    cfg_path = os.path.join(current_path, 'pIon.cfg')
    parameter_dict = parameter_file_read(cfg_path)

    if os.path.exists(parameter_dict['output_path']) == False:
        print('output path is not exist!')
        return
    if os.path.exists(parameter_dict['fasta_path']) == False:
        print('fasta path is not exist!')
        return
    for ms_path in parameter_dict['msms_path']:
        ms_path = ms_path.split('=')[1].strip()
        if os.path.exists(ms_path) == False:
            print('msms path is not exist!')
            return

    pparse_output_path = os.path.join(os.path.join(
        parameter_dict['output_path'], 'source'), 'pParse')
    if os.path.exists(pparse_output_path):
        shutil.rmtree(pparse_output_path)
    temp_result_to_delete_path = os.path.join(
        parameter_dict['output_path'], 'reporting_summary')
    if os.path.exists(temp_result_to_delete_path):
        shutil.rmtree(temp_result_to_delete_path)
    close_output_path = os.path.join(os.path.join(
        parameter_dict['output_path'], 'source'), 'close')
    if os.path.exists(close_output_path):
        shutil.rmtree(close_output_path)
    blind_output_path = os.path.join(os.path.join(
        parameter_dict['output_path'], 'source'), 'blind')
    if os.path.exists(blind_output_path):
        shutil.rmtree(blind_output_path)

    parameter_dict['use_close_search'] = 'False'
    start_time = time.time()
    blind_search(current_path)
    blind_time = time.time()
    if parameter_dict['isotope_labeling'] == 'True':
        parameter_dict['use_close_search'] = 'True'
    else:
        parameter_dict['use_close_search'] = 'False'
    
    # pyp：同位素模式，但不做限定式搜索，程序打包时需要删除下面这一行
    parameter_dict['use_close_search'] = 'False'
    
    if parameter_dict['use_close_search'] == 'True':
        new_close_search(current_path)
        close_time = time.time()
        print('[time cost]')
        print('blind search cost time (s): ',
              round(blind_time - start_time, 1))
        print('restricted search cost time (s): ',
              round(close_time - blind_time, 1))
    else:
        print('[time cost]')
        print('blind search cost time (s): ',
              round(blind_time - start_time, 1))

    delete_file(current_path, 'psite.txt')

    parameter_dict = parameter_file_read_i(cfg_path)
    if parameter_dict['ion_labeling'] == 'True':
        p_ion_run(parameter_dict, current_path)
    delete_file(current_path, 'modification-null.ini')
    delete_file(current_path, 'modification-new.ini')


def p_ion_run(parameter_dict, current_path):
    """
    Main function for pion to process mass spectrometry data, perform modification detection,
    and generate various reports and summaries.
    
    Returns:
    - pchem_summary_path1: Path to the updated pChem summary after processing.
    """
    filter_frequency = parameter_dict['filter_frequency']
    blind_res_path = os.path.join(
        parameter_dict['output_path'], "source/blind/pFind-Filtered.spectra")
    blind_res = blind_res_read(blind_res_path)
    pp_path = os.path.join(parameter_dict['output_path'], "source/pParse")
    mgf_path_list = glob.glob(os.path.join(pp_path, "*.mgf"))

    blind_res_raw_dict = {}
    mass_spectra_dict = {}
    for i in range(len(mgf_path_list)):
        mgf_path = mgf_path_list[i]
        t_name = mgf_path.split(
            "\\")[-1].split(".")[0][:mgf_path.split("\\")[-1].split(".")[0].rfind("_")]
        blind_res_raw_dict[t_name] = []
        for line in blind_res:
            if t_name in line:
                blind_res_raw_dict[t_name].append(line)
        mass_spectra_dict_tmp = mgf_read_for_ion_pfind_filter2(
            mgf_path, blind_res_raw_dict[t_name])
        mass_spectra_dict.update(mass_spectra_dict_tmp)

    pchem_summary_path = os.path.join(
        parameter_dict['output_path'], "reporting_summary/pChem.summary")
    modification_list, modification_dict, modification_PSM, modification_site = get_modification_from_result(
        pchem_summary_path)
    pchem_output_path = os.path.join(
        parameter_dict['output_path'], "reporting_summary")
    ion_type = str(round(float(parameter_dict['ion_type']), 3))
    ion_filter_mode = 2
    ion_rank_threshold = 1
    ion_relative_mode = 1
    pchem_summary_path1 = close_ion_learning(pchem_output_path, current_path, ion_type, modification_list, modification_dict,
                                             blind_res, mass_spectra_dict, modification_PSM, modification_site, pchem_summary_path, ion_relative_mode, ion_rank_threshold, ion_filter_mode, parameter_dict['ion_filter_ratio'])
    psm_filter(pchem_summary_path1, filter_frequency)
    heat_map_draw = heatmap_ion(pchem_output_path, pchem_summary_path1)


if __name__ == "__main__":
    run()
    os.system("pause")
