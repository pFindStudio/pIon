'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2023-02-01 16:16:24
LastEditTime: 2023-05-24 15:03:10
FilePath: \pChem-main\plabel_run.py
Description: Do not edit
'''
import os
from utils import parameter_file_read, parameter_modify 
import glob
import time
from utils import parameter_file_read_ion
import shutil 
import linecache
from configparser import ConfigParser
from utils import parameter_pick
from tqdm import tqdm
import configparser
from collections import Counter

# 获得用户人为设定的修饰类型
def get_common_modification(current_path):
    modification_file = os.path.join(current_path, 'modification-null.ini') 
    with open(modification_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    common_modifications = []
    for line in lines:
        if "name" in line:
            common_modifications.append(parameter_pick(line).split(" ")[0])
    return common_modifications

def get_unknown_modification(current_path):
    pchem_ion_cfg_path = os.path.join(current_path, 'pChem-ion.cfg') 
    # pchem_ion_cfg_path = os.path.join(current_path, 'pChem_label.cfg') 
    parameter_dict_ion = parameter_file_read_ion(pchem_ion_cfg_path)
    unknown_modification_file = os.path.join(parameter_dict_ion['output_path'], "reporting_summary/pChem_ion_filter_mode2.summary")
    # unknown_modification_file = os.path.join(parameter_dict_ion['output_path'], "reporting_summary/pChem.summary")
    with open(unknown_modification_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[1:]    
    unknown_modifications_dict = {}
    for line in lines:
        if "PFIND_DELTA" in line:
            unknown_modifications_dict[line.split("\t")[1]] = {}
            unknown_modifications_dict[line.split("\t")[1]]['acc_mass'] = line.split("\t")[2].split(" ")[0]
            t_mod_pos = line.split("\t")[3].split('|')[0]
            t_mod_pos_others = line.split("\t")[4]
            t_mod_pos_others_list = t_mod_pos_others.split(";  ")[:-1]
            t_mod_pos_others_list = [i.split('(')[0] for i in t_mod_pos_others_list]
            unknown_modifications_dict[line.split("\t")[1]]['pos_list'] = [t_mod_pos] + t_mod_pos_others_list
    return unknown_modifications_dict


# pLabel使用：读取template文件格式
def read_label_template(current_path, mgf_file_path):
    plabel_template_path = os.path.join(current_path, 'bin/pLabel/template/plabel_template.plabel') 
    # conf = ConfigParser()  # 需要实例化一个ConfigParser对象
    conf= configparser.RawConfigParser()
    conf.optionxform = lambda option: option
    conf.read(plabel_template_path)
    # 设置mgf file path
    conf.remove_section('FilePath')
    conf.add_section('FilePath')
    conf.set('FilePath', 'File_Path', mgf_file_path)
    
    # 设置修饰类型：用户人为设置的常见修饰类型以及PFIND_DELTA未知修饰类型
    conf.remove_section('Modification')
    conf.add_section('Modification')
    common_modifications = get_common_modification(current_path)
    for index, i in enumerate(common_modifications):
        conf.set('Modification', str(index+1), i)
    unknown_modifications_dict = get_unknown_modification(current_path)
    unknown_mod_num = 0
    for index, i in enumerate(unknown_modifications_dict.keys()):
        for mod_site in unknown_modifications_dict[i]['pos_list']:
            unknown_mod_num += 1
            if mod_site == "N-SIDE":
                t_mod_name = i + "[{}]".format("AnyN-term")
            elif mod_site == "C-SIDE":
                t_mod_name = i + "[{}]".format("AnyC-term")
            else:
                t_mod_name = i + "[{}]".format(mod_site)
            conf.set('Modification', str(unknown_mod_num+len(common_modifications)), t_mod_name)
           
    conf.write(open(plabel_template_path, "w"), space_around_delimiters=False)
    option = conf.options('Modification')
    return plabel_template_path, common_modifications, unknown_modifications_dict

def plabel_file_generate(current_path, plabel_template_path, file_path, common_modifications, unknown_modifications_dict): 
    with open(file_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
    modification_dict = {}
    
    for index, mod in enumerate(common_modifications):
        modification_dict[str(index+1)] = mod
    # for index, mod in enumerate(unknown_modifications_dict.keys()):
    #     modification_dict[str(index+1+len(common_modifications))] = mod
    unknown_mod_num = 0
    for index, mod in enumerate(unknown_modifications_dict.keys()):
        for mod_site in unknown_modifications_dict[mod]['pos_list']:
            unknown_mod_num += 1
            if mod_site == "N-SIDE":
                t_mod_name = mod + "[{}]".format("AnyN-term")
            elif mod_site == "C-SIDE":
                t_mod_name = mod + "[{}]".format("AnyC-term")
            else:
                t_mod_name = mod + "[{}]".format(mod_site)
            modification_dict[str(unknown_mod_num + len(common_modifications))] = t_mod_name
    mod_list_of_key = list(modification_dict.keys())
    mod_list_of_value = list(modification_dict.values())
    spectra_dict = {}
    for mod in unknown_modifications_dict.keys():
        spectra_dict[mod] = []
        for line in lines:
            line_split = line.split('\t')
            spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
            if mod + '.' in mod_list:
                pos_mod_list = mod_list.split(';')[:-1] 
                mod_res = ''
                save_flag = True
                for i in pos_mod_list:
                    pos, mod_type = i.split(",")
                    mod_res += (pos+',')
                    if mod_type in common_modifications:
                        p = mod_list_of_value.index(mod_type)
                        mod_res += mod_list_of_key[p]
                        mod_res += " "
                    else:
                        # 判断是否符合位点
                        pos_site = peptide_sequence[int(pos)-1]
                        if mod+"[{}]".format(pos_site) in mod_list_of_value:
                            p = mod_list_of_value.index(mod+"[{}]".format(pos_site))
                        if pos_site not in unknown_modifications_dict[mod]['pos_list']:
                            if (pos == '1') and ('N-SIDE' in unknown_modifications_dict[mod]['pos_list']):
                                save_flag = True
                                p = mod_list_of_value.index(mod+"[{}]".format("AnyN-term"))
                            elif  (pos == str(len(peptide_sequence))) and ('C-SIDE' in unknown_modifications_dict[mod]['pos_list']):
                                save_flag = True
                                p = mod_list_of_value.index(mod+"[{}]".format("AnyC-term"))
                            else:
                                save_flag = False
                            
                        # p = mod_list_of_value.index(mod+"[{}]".format(pos_site))
                        # if mod+"[{}]".format(pos_site) in mod_list_of_value:
                        if save_flag:
                            mod_res += mod_list_of_key[p]
                            mod_res += " "
                if save_flag:
                    spectra_dict[mod].append((spectrum_name, peptide_sequence, mod_res))
    # write_plabel_spectra(current_path, plabel_template_path, spectra_dict, unknown_modifications_dict) 
    # print("=====")    
    lines = lines[1:]
    spectra_dict["without_mod"] = []
    for line in lines:
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        if mod_list == "":
            spectra_dict["without_mod"].append((spectrum_name, peptide_sequence))
    return spectra_dict, modification_dict   
    
# 向plabel配置文件中写入谱图
def write_plabel_spectra(current_path, plabel_template_path, spectra_dict, unknown_modifications_dict, spec2score):
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg') 
    # cfg_path = os.path.join(current_path, 'pChem_label.cfg') 
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    out_put_path = os.path.join(parameter_dict_ion['output_path'], "source/pLabel")
    for mod in spectra_dict.keys():
        if mod != "without_mod":
            spectra_id = 0
            plabel_file = os.path.join(out_put_path, '{}.plabel'.format(mod))
            file2 = open(plabel_file,"w")
            file1 = open(plabel_template_path,"r")
            s = file1.read()
            w = file2.write(s)
            file2.close()
            # conf = ConfigParser()
            conf= configparser.RawConfigParser()
            conf.optionxform = lambda option: option
            conf.read(plabel_file)
            conf.remove_section('Total')
            conf.add_section('Total')
            conf.set('Total', 'total', str(len(spectra_dict[mod])))
            for index, i in enumerate(spectra_dict[mod]):
                if i[0] not in spec2score[mod].keys():
                    conf.add_section('Spectrum{}'.format(str(index+1)))
                    conf.set('Spectrum{}'.format(str(index+1)), 'name', i[0])
                    conf.set('Spectrum{}'.format(str(index+1)), 'pep1', '0 {}'.format(i[1]) + ' {} '.format(str(1)) + i[2])
                    continue
                conf.add_section('Spectrum{}'.format(str(index+1)))
                conf.set('Spectrum{}'.format(str(index+1)), 'name', i[0])
                conf.set('Spectrum{}'.format(str(index+1)), 'pep1', '0 {}'.format(i[1]) + ' {} '.format(str(spec2score[mod][i[0]])) + i[2])
            
        # conf.set('Total', 'total', str(len(spectra_dict[mod])))
            conf.write(open(plabel_file, "w"), space_around_delimiters=False) 
        else:
            spectra_id = 0
            plabel_file = os.path.join(out_put_path, '{}.plabel'.format(mod))
            file2 = open(plabel_file,"w")
            file1 = open(plabel_template_path,"r")
            s = file1.read()
            w = file2.write(s)
            file2.close()
            # conf = ConfigParser()
            conf= configparser.RawConfigParser()
            conf.optionxform = lambda option: option
            conf.read(plabel_file)
            conf.remove_section('Total')
            conf.add_section('Total')
            conf.set('Total', 'total', str(len(spectra_dict[mod])))
            for index, i in enumerate(spectra_dict[mod]):
                # if i[0] not in spec2score[mod].keys():
                #     conf.add_section('Spectrum{}'.format(str(index+1)))
                #     conf.set('Spectrum{}'.format(str(index+1)), 'name', i[0])
                #     conf.set('Spectrum{}'.format(str(index+1)), 'pep1', '0 {}'.format(i[1]))
                #     continue
                conf.add_section('Spectrum{}'.format(str(index+1)))
                conf.set('Spectrum{}'.format(str(index+1)), 'name', i[0])
                conf.set('Spectrum{}'.format(str(index+1)), 'pep1', '0 {}'.format(i[1]) + ' 1 ')
            conf.write(open(plabel_file, "w"), space_around_delimiters=False) 
        
def prepare_mgf_file(current_path):
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg') 
    # cfg_path = os.path.join(current_path, 'pChem_label.cfg') 
    # parameter_dict = parameter_file_read(cfg_path) 
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    pp_path = os.path.join(parameter_dict_ion['output_path'], "source/pParse")
    mgf_path_list = glob.glob(os.path.join(pp_path, "*.mgf"))
    pl_path = os.path.join(parameter_dict_ion['output_path'], "source/pLabel")
    if os.path.exists(pl_path):  
        shutil.rmtree(pl_path) 
    os.makedirs(pl_path) 
    data = []
    plabel_mgf = os.path.join(pl_path, 'pLabel_tmp.mgf') 
    file2 = open(plabel_mgf, "a+")
    for i in range(len(mgf_path_list)):
        mgf_path = mgf_path_list[i]
        # blind_res 按名称分类
        file1 = open(mgf_path,"r")
        # file2 = open(plabel_mgf,"a+")
        s = file1.read()
        w = file2.write(s)
        file1.close()
    file2.close()
    return plabel_mgf


def write_element_ini(current_path, unknown_modifications_dict):
    element_ini_path = os.path.join(current_path, 'bin/pLabel/bin/element.ini') 
    with open(element_ini_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    element_num = parameter_pick(lines[0])
    
    # 如果之前运行过这个函数，需要reset element.ini
    # reset:
    if element_num != '117':
        del_num = int(element_num) - 117
        lines = lines[:-del_num]
    element_num = '117'

    if lines[-1][-1] != '\n':
        lines[-1] += '\n'
    
    for index, mod in enumerate(unknown_modifications_dict.keys()):
        tmp = "E{}=".format(str(int(element_num)+index+1))
        tmp += (mod+'_element|')
        tmp += (unknown_modifications_dict[mod]['acc_mass'] + ',|1.0,|\n')
        lines.append(tmp)
    lines[-1] = lines[-1][:-1]
    line_0_tmp = lines[0].split("=")
    lines[0] = line_0_tmp[0] + "=" + str(int(element_num)+len(unknown_modifications_dict.keys())) + '\n'
    element_file = open(element_ini_path,"w")
    content = "".join(lines)
    w = element_file.write(content)
    
def write_modification_ini(current_path, unknown_modifications_dict):
    modification_ini_path = os.path.join(current_path, 'bin/pLabel/bin/modification.ini') 
    with open(modification_ini_path, 'a+', encoding='utf-8') as mod_file:
        mod_file.truncate(0)
        
    mod_null_ini_path = os.path.join(current_path, 'modification-null.ini')
    file1 = open(mod_null_ini_path, "r")
    file2 = open(modification_ini_path, "w+")
    s = file1.read()
    w = file2.write(s)
    file2.close()
    with open(modification_ini_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    modification_num = parameter_pick(lines[0])
    mod_id = int(modification_num)
    for mod in unknown_modifications_dict.keys():
        for mod_site in unknown_modifications_dict[mod]['pos_list']:
            if mod_site == 'N-SIDE':
                mod_id += 1
                t_line_1 = "name" + str(mod_id) + "="
                t_line_1 += mod
                t_line_1 += "[{}]".format("AnyN-term")
                t_line_1 += " "
                t_line_1 += "0\n"
                t_line_2 = mod
                t_line_2 += "[{}]=ABCDEFGHIJKLMNOPQRSTUVWXYZ PEP_N ".format("AnyN-term")
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " "
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " 0 "
                t_line_2 += (mod + "_element(1)\n")
            elif mod_site == 'C-SIDE':
                mod_id += 1
                t_line_1 = "name" + str(mod_id) + "="
                t_line_1 += mod
                t_line_1 += "[{}]".format("AnyC-term")
                t_line_1 += " "
                t_line_1 += "0\n"
                t_line_2 = mod
                t_line_2 += "[{}]=ABCDEFGHIJKLMNOPQRSTUVWXYZ PEP_C ".format("AnyC-term")
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " "
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " 0 "
                t_line_2 += (mod + "_element(1)\n")
            else:
                mod_id += 1
                t_line_1 = "name" + str(mod_id) + "="
                t_line_1 += mod
                t_line_1 += "[{}]".format(mod_site)
                t_line_1 += " "
                t_line_1 += "0\n"
                t_line_2 = mod
                t_line_2 += "[{}]={} NORMAL ".format(mod_site, mod_site)
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " "
                t_line_2 += unknown_modifications_dict[mod]['acc_mass']
                t_line_2 += " 0 "
                t_line_2 += (mod + "_element(1)\n")
            lines.append(t_line_1)
            lines.append(t_line_2)
    t_line_0 = lines[0].split("=")[0] + "=" + str(mod_id) + "\n"        
    lines[0] = t_line_0    
    content = "".join(lines)
    modification_file = open(modification_ini_path,"w")
    w = modification_file.write(content)    
    modification_file.close()

def write_plabel_ini(current_path):
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg') 
    # cfg_path = os.path.join(current_path, 'pChem_la.cfg') 
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    output_path = os.path.join(parameter_dict_ion['output_path'], "source/pLabel")
    plabel_ini_path = os.path.join(current_path, 'bin/pLabel/bin/pLabel.ini') 
    plabel_path_list = glob.glob(os.path.join(output_path, "*.plabel")) 
    if len(plabel_path_list) == 0:
        return
    plabel_file = plabel_path_list[0]
    plabel_mgf = os.path.join(parameter_dict_ion['output_path'], "source/pLabel/pLabel_tmp.mgf")
    conf= configparser.RawConfigParser()
    conf.optionxform = lambda option: option
    conf.read(plabel_ini_path)
    conf.set('File', 'DtaFile', plabel_mgf)
    conf.set('File', 'pLabelFile', plabel_file)
    
    with open(plabel_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    name_lines = []
    pep_list = []
    for index, line in enumerate(lines):
        if "name" in line:
            name_lines.append(line)
            pep_list.append(lines[index+1].split(" ")[1])
    name_pep_list = [(i, pep_list[index]) for index, i in enumerate(name_lines)]
    sorted_by_name = sorted(name_pep_list, key=lambda tup: tup[0])
    pep_ini_seq = sorted_by_name[0][1]
    conf.set('File', 'Sequence', pep_ini_seq)
    conf.set('File', 'PicFile', os.path.join(output_path, "Pictures"))
    conf.set('File', 'DataFile', os.path.join(output_path, "DataTemp"))
    conf.write(open(plabel_ini_path, "w"), space_around_delimiters=False) 
    
    
# mgf文件删减
def cut_mgf_file(plabel_mgf, spectra_dict):
    with open(plabel_mgf, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    content = "".join(lines)
    s = content.split("END IONS\n")
    print(spectra_dict.keys())
    s = s[:-1]
    reserve_spectra_name = []
    for mod in spectra_dict.keys():
        for i in spectra_dict[mod]:
            reserve_spectra_name.append(i[0])
    reserve_spectra_name.sort();

    mgf_reserve = []
    for name in tqdm(reserve_spectra_name):
        for index, i in enumerate(s):
            if name in i:
                mgf_reserve.append(i)
                s = s[index:]
                break
    print(len(mgf_reserve))
    

def spectra_name_list_generate(summary_file_path, mod): 
    spectra_name_list = [] 
    with open(summary_file_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 
    mod = 'PFIND_DELTA_' + mod 
    for line in lines: 
        if mod not in line: 
            continue 
        else:
            spectra_name_list.append(line.split('\t')[0]) 
    return spectra_name_list
    
# 读取psite的结果文件
def psite_result_read(file_path, blind_summary_file_path, mod): 
    with open(file_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 
    spec2score = {}
    #i = 0
    #total = 0
    spectra_name_list = spectra_name_list_generate(blind_summary_file_path, mod)
    for line in lines: 
        line = line.split('\t') 
        spectrum_name = line[0]
        if spectrum_name not in spectra_name_list:
            continue
        score = float(line[3])
        spec = line[0] 
        # 用于测试不同的psite阈值 
        #if score < 5.0:
        #    continue
        #print(score, spec2pos[spec]) 
        #if spec2pos[spec][0] == 'C':
        #    i += 1  
        #total += 1 
    #print(float(i/total)) 
        spec2score[spec] = score 
    
    return spec2score    
    

def psite_file_generate1(file_path, target_mod, current_path): 
    with open(file_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 
    
    idx = 1 
    new_lines = []
    spect2pos = {}
    for line in lines[1:]: 
        if target_mod not in line:
            continue 
        line = line.split() 
        spectrum, sequence, mod = line[0], line[5], line[10]  
        mod = mod.split(',')
        if len(mod) > 2:
            continue 
        pos = int(mod[0]) 
        if pos == 0: 
            sequence = 'm' + sequence 
            spect2pos[spectrum] = [sequence[0], 'N-SIDE']
        elif pos >= len(sequence):
            sequence = sequence + 'm' 
            spect2pos[spectrum] = [sequence[len(sequence)-1], 'C-SIDE']
        else:
            sequence = sequence[:pos] + 'm' + sequence[pos:] 
            spect2pos[spectrum] = [sequence[pos - 1]]

        new_lines.append('S' + str(idx) + '\t' + spectrum + '\n') 
        new_lines.append('P1\t' + sequence + '\t0\n') 
        new_lines.append('\n') 
        idx += 1 

    with open(os.path.join(current_path, 'psite.txt'), 'w', encoding='utf-8') as f: 
        for line in new_lines: 
            f.write(line)
            
            
def psite_cfg_write(psite_template_path, current_path, source_path): 
    with open(psite_template_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines() 

    pparse_path = os.path.join(source_path, 'pParse') 
    mgf_name = '.mgf'
    for name in os.listdir(pparse_path): 
        if '.mgf' in name: 
            mgf_name = name  
            break 
    
    mgf_path = os.path.join(pparse_path, mgf_name) 
    psite_input_path = os.path.join(current_path, 'psite.txt')  

    new_lines = [] 
    for line in lines: 
        if 'mgfPath' in line: 
            line = parameter_modify(line, mgf_path) 
        if 'resultPath' in line: 
            line = parameter_modify(line, psite_input_path)
        new_lines.append(line) 
    

    with open(psite_template_path, 'w', encoding='utf-8') as f: 
        for line in new_lines: 
            f.write(line)
            
            
# 统计新的位点频率 拷贝自函数mass_static
def position_static(mod, spec_name_list, blind_summary_file_path, side_position='True'): 
    mod_position_list = [] 
    if side_position == 'True': 
        side_flag = True 
    else: 
        side_flag = False 
    with open(blind_summary_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines() 
    spectra_num = len(lines)


    for i in range(1, spectra_num):
        if len(lines[i]) < 4:
            break
        spec_name = lines[i].split('\t')[0] 
        if spec_name not in spec_name_list: 
            continue 
        
        sequence = lines[i].split('\t')[5]
        mod_list = lines[i].split('\t')[10].split(';')[:-1]
        for mod_line in mod_list: 
            pos, mod_name = mod_line.split(',')
            if mod in mod_name: 
                pos = int(pos)
                if pos == 0 or pos == 1:
                    mod_position_list.append(sequence[0]) 
                    if side_flag == True:
                        mod_position_list.append('N-SIDE')
                elif pos >= len(sequence):
                    mod_position_list.append(sequence[-1]) 
                    if side_flag == True:
                        mod_position_list.append('C-SIDE')
                else:
                    mod_position_list.append(sequence[pos-1]) 
    
    return Counter(mod_position_list)

def local_list_combine(local_list, position_counter): 
    new_local_list = [] 
    cut_flag = False 
    cut_num = int(local_list[0][1]/3) 
    for pos, num in local_list: 
        if position_counter[pos] <= 1 and len(new_local_list) > 1: 
            break 
        if num < cut_num: 
            cut_flag = True
        if cut_flag == False: 
            new_local_list.append([pos, max(num, position_counter[pos])]) 
        else: 
            if position_counter[pos] == 0: 
                new_local_list.append([pos, num]) 
            else:
                new_local_list.append([pos, min(num, position_counter[pos])]) 
    return new_local_list

# def psite_run(parameter_dict, current_path, mod, pattern='blind', local_list=None): 
    # source_path = parameter_dict['output_path']
    # blind_summary_file_path = os.path.join(source_path, pattern) 
    # blind_summary_file_path = os.path.join(blind_summary_file_path, 'pFind-Filtered.spectra') 

    # bin_path = os.path.join(current_path, 'bin') 
    # psite_path = os.path.join(bin_path, 'pSite') 
    # psite_template_path = os.path.join(psite_path, 'template') 
    # psite_template_path = os.path.join(psite_template_path, 'param_pSite.txt') 
    
    # # 产生输入文件 只有第一次调用才会运行 
    # if parameter_dict['psite_run'] == 'True': 
    #     psite_file_generate(blind_summary_file_path, 'PFIND_DELTA_', current_path) 
        
    #     # 2. 生成psite参数文件 
    #     psite_cfg_write(psite_template_path, current_path, source_path)

    #     # 3. 运行pSite输出打分结果 
    #     # psite_exe_path = os.path.join(psite_path, 'pPredictAA.exe') 
    #     # 使用mingw64编译的psite不会报告UAC错误
    #     psite_exe_path = os.path.join(psite_path, 'a.exe') 
    #     cmd = psite_exe_path + ' ' + psite_template_path 
    #     os.chdir(psite_path)
    #     receive = os.system(cmd) 
    #     print(receive) 
    #     os.chdir(current_path) 

    #     parameter_dict['psite_run'] = 'False' 

    # # 4. 读取结果文件，卡值后返回新的位点分布 
    # psite_res_path = os.path.join(psite_path, 'res1.txt') 
    # spec2score_dict = psite_result_read(psite_res_path, blind_summary_file_path, mod) 

    # spec2score_list = []
    # for spec, score in spec2score_dict.items(): 
    #     spec2score_list.append([spec, score]) 
    
    # # 用于卡值，可以换成其他策略
    # cut_off_ratio = 10
    # spec2score_list.sort(key=lambda s:s[1]) 
    # cut_off_num = int(len(spec2score_list) * (1 - cut_off_ratio / 100.0)) 
    # spec_name_list = []
    # for i in range(cut_off_num): 
    #     spec_name_list.append(spec2score_list[i][0]) 
    
    # position_counter = position_static(mod, spec_name_list, blind_summary_file_path, parameter_dict['side_position'])
    
    # return local_list_combine(local_list, position_counter)


def psite_cfg_write(current_path, plabel_mgf, mod):
    bin_path = os.path.join(current_path, 'bin') 
    psite_path = os.path.join(bin_path, 'pSite') 
    psite_template_path = os.path.join(psite_path, 'template') 
    psite_template_path = os.path.join(psite_template_path, 'param_pSite.txt') 
    
    with open(psite_template_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    mgf_path = plabel_mgf 
    psite_input_path = os.path.join(current_path, 'psite_{}.txt'.format(mod))

    new_lines = []
    for line in lines:
        if 'mgfPath' in line:
            line = parameter_modify(line, mgf_path)
        if 'resultPath' in line:
            line = parameter_modify(line, psite_input_path)
        new_lines.append(line)

    with open(psite_template_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)
    return psite_template_path
            

def psite_file_generate(current_path, parameter_dict_ion, unknown_modifications_dict, spectra_dict):
    output_path = parameter_dict_ion['output_path']
    file_path = os.path.join(output_path, 'source/blind/pFind-Filtered.spectra')
    mod_psite_lines = {}
    
    for mod in unknown_modifications_dict.keys():
        mod_psite_lines[mod] = psite_file_generate_for_mod(file_path, mod, current_path, spectra_dict)
    
    new_lines = []
    for index, mod in enumerate(unknown_modifications_dict.keys()):
        new_lines = mod_psite_lines[mod]
        with open(os.path.join(current_path, 'psite_{}.txt'.format(mod)), 'w', encoding='utf-8') as f:
            for line in new_lines:
                f.write(line)    
            
def psite_file_generate_for_mod(file_path, target_mod, current_path, spectra_dict):
    mod_spectra_list = [i[0] for i in spectra_dict[target_mod]]
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    idx = 1
    new_lines = []
    spect2pos = {}
    for line in lines[1:]:
        if target_mod not in line:
            continue
        if target_mod + '.' in line:
            line = line.split()
            spectrum, sequence, mod = line[0], line[5], line[10]
            mod = mod.split(',')
            if len(mod) > 2:
                continue
            pos = int(mod[0])
            if pos == 0:
                sequence = 'm' + sequence
                spect2pos[spectrum] = [sequence[0], 'N-SIDE']
            elif pos >= len(sequence):
                sequence = sequence + 'm'
                spect2pos[spectrum] = [sequence[len(sequence)-1], 'C-SIDE']
            else:
                sequence = sequence[:pos] + 'm' + sequence[pos:]
                spect2pos[spectrum] = [sequence[pos - 1]]
            
            if spectrum in mod_spectra_list:
                new_lines.append('S' + str(idx) + '\t' + spectrum + '\n')
                new_lines.append('P1\t' + sequence + '\t0\n')
                new_lines.append('\n')
                idx += 1
                mod_spectra_list.remove(spectrum)
    return new_lines

def psite_run(current_path, parameter_dict_ion, spectra_dict, modification_dict, plabel_mgf):
    psite_path = os.path.join(current_path, 'bin/pSite')
    psite_exe_path = os.path.join(psite_path, 'a.exe')
    spec2score = {}
    if parameter_dict_ion['psite_run'] == "True":
        for mod in tqdm(modification_dict.keys()):
            print("start psite predicting for {}".format(mod))
            psite_template_path = psite_cfg_write(current_path, plabel_mgf, mod)
            cmd = psite_exe_path + ' ' + psite_template_path
            os.chdir(psite_path)
            receive = os.system(cmd)
            print(receive)
            os.chdir(current_path)
            
            psite_res_path = os.path.join(psite_path, 'res1.txt')
            with open(psite_res_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            spec2score[mod] = {}
            for line in lines:
                line = line.split('\t')
                spec_name = line[0]
                score = float(line[3])
                spec2score[mod][spec_name] = score
    else:
        for mod in tqdm(modification_dict.keys()):
            spec2score[mod] = {}
            for i in spectra_dict[mod]:
                spec_name = i[0]
                spec2score[mod][spec_name] = 1.0
    return spec2score

def plabel_run1(current_path):
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg') 
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    plabel_mgf = prepare_mgf_file(current_path)  
    plabel_template_path, common_modifications, unknown_modifications_dict = read_label_template(current_path, plabel_mgf)
    pfind_filtered_spectra_path = os.path.join(parameter_dict_ion['output_path'], "source/blind/pFind-Filtered.spectra")
    # plabel_file_generate(pfind_filtered_spectra_path)
    
    spectra_dict, modification_dict = plabel_file_generate(current_path, plabel_template_path, pfind_filtered_spectra_path, common_modifications, unknown_modifications_dict)
    psite_file_generate(current_path, parameter_dict_ion, unknown_modifications_dict, spectra_dict)
    
    # 这里的修饰位点打分情况个人觉得还有待改进，之后会优化这一部分
    spec2score = psite_run(current_path, parameter_dict_ion, spectra_dict, unknown_modifications_dict, plabel_mgf)
    
    write_plabel_spectra(current_path, plabel_template_path, spectra_dict, unknown_modifications_dict, spec2score) 
    write_element_ini(current_path, unknown_modifications_dict)
    write_modification_ini(current_path, unknown_modifications_dict)
    write_plabel_ini(current_path)
    return spectra_dict, modification_dict

      

if __name__ == "__main__":  
    current_path = os.getcwd() 
    cfg_path = os.path.join(current_path, 'pChem_label.cfg') 
    plabel_run1(current_path)
    # parameter_dict_ion = parameter_file_read_ion(cfg_path)
    # plabel_mgf = prepare_mgf_file(current_path)
    # plabel_template_path, common_modifications, unknown_modifications_dict = read_label_template(current_path, plabel_mgf)
    # pfind_filtered_spectra_path = os.path.join(parameter_dict_ion['output_path'], "source/blind/pFind-Filtered.spectra")
    # # plabel_file_generate(pfind_filtered_spectra_path)
    
    # spectra_dict, modification_dict = plabel_file_generate(current_path, plabel_template_path, pfind_filtered_spectra_path, common_modifications, unknown_modifications_dict)
    # psite_file_generate(current_path, parameter_dict_ion, unknown_modifications_dict, spectra_dict)
    
    # # 这里的修饰位点打分情况个人觉得还有待改进，之后会优化这一部分
    # spec2score = psite_run(current_path, parameter_dict_ion, spectra_dict, unknown_modifications_dict)
    
    # write_plabel_spectra(current_path, plabel_template_path, spectra_dict, unknown_modifications_dict, spec2score) 
    # write_element_ini(current_path, unknown_modifications_dict)
    # write_modification_ini(current_path, unknown_modifications_dict)
    # write_plabel_ini(current_path)
    
    # # psite_file_generate(current_path, parameter_dict_ion, unknown_modifications_dict, spectra_dict)
    # # spec2score = psite_run(current_path, parameter_dict_ion, spectra_dict, unknown_modifications_dict)
    
    
    # print(" ")
    os.system("pause")
    
    
   