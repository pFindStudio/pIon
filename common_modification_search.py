'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2022-09-06 10:13:10
LastEditTime: 2022-11-30 01:24:58
FilePath: \pChem-main\common_modification_search.py
Description: Do not edit
'''
# 使用开放式寻找常见的修饰列表
import os
from utils import parameter_file_read, open_cfg_write, search_exe_path, spectra_result_read


# 将整个开放式搜索过程包装成一个函数
def open_search(current_path):

    # 参数文件的路径
    
    pchem_cfg_path = os.path.join(current_path, 'pChem.cfg')
    open_cfg_path = os.path.join(os.path.join(os.path.join(current_path, 'bin'), 'template'), 'open.cfg')
    
    # 读取参数
    parameter_dict = parameter_file_read(pchem_cfg_path)
    parameter_dict['pfind_install_path'] = 'G:/pChem_ion/pChem-main/pChem-main/bin/pFind'
    parameter_dict['msms_path'] = 'G:/pChem_ion/pChem-main/pChem-main/20221130result_IPM0619_p_1_PSM_0_200_500_test\source\pParse\HFX_YangJing_HJX_DMP500R1_20220619_R2_HCDFT.pf2'
    print('parameter_dict: ', parameter_dict) 
    
    if parameter_dict['open_flag'] == 'True':
        
        # 写入开放式pFind参数文件
        res_path = open_cfg_write(open_cfg_path, parameter_dict)
        
        # 调用pFind进行搜索
        bin_path, exe_path = search_exe_path(parameter_dict)
        cmd = exe_path + ' ' + open_cfg_path 
        os.chdir(bin_path)
        receive = os.system(cmd)
        print(receive)
        
        
        
        # 读取结果文件，给出常见修饰列表
        spectra_res_path = os.path.join(res_path, 'pFind.summary')
        spectra_result_read(spectra_res_path, current_path, parameter_dict['common_modification_number'])
    

if __name__ == "__main__":
    current_path = os.getcwd()
    open_search(current_path)
    