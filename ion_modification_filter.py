'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2023-04-10 09:54:56
LastEditTime: 2024-09-28 14:39:15
FilePath: \pChem2\ion_modification_filter.py
Description: Do not edit
'''

from asyncio.windows_events import NULL
from utils import parameter_file_read, parameter_file_read_ion
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from parameter import element_dict, amino_acid_dict, common_dict_create
import pandas as pd
from tqdm import tqdm
import time
from scipy.stats import mannwhitneyu
import json
import glob
import warnings
from scipy.spatial.distance import cosine
from numba import jit
from numba.typed import List
import seaborn as sns

h2o_mass = element_dict["H"]*2 + element_dict["O"]*1
proton_mass = element_dict['Pm']
# 存放谱图的类
class MassSpectrum:
    def __init__(self, charge, pepmass, peak_list):
        self.charge = charge
        self.pepmass = pepmass
        self.peak_list = peak_list


class MassSpectrum_for_ion:
    def __init__(self, charge, pepmass, peak_list, spectrum_ms1_peak):
        self.charge = charge
        self.pepmass = pepmass
        self.peak_list = peak_list
        self.ms1_peak = spectrum_ms1_peak
        self.relative_peak_list = self.calculate_relative_peak_list()

    def calculate_relative_peak_list(self):
        """Calculate relative peak intensities based on the maximum peak."""
        if not self.peak_list:
            return []

        # Extract intensities from peak_list
        intensities = [float(item[1]) for item in self.peak_list]
        max_peak = max(intensities)

        # Create relative peak list
        return [(item[0], intensity / max_peak) for item, intensity in zip(self.peak_list, intensities)]


@jit
def cal_divide(a, b):
    return a/b


class MassSpectrum_for_ion1:
    def __init__(self, charge, pepmass, peak_list, rt):
        self.charge = charge
        self.pepmass = pepmass
        self.peak_list = peak_list
        self.rt = rt
        self.relative_peak_list = self.get_relative_peak_list()

    # def forward(self):
    #     self.get_relative_peak_list()

    def get_relative_peak_list(self):
        position_list = []
        peak_list = []
        relative_peak_list = []
        for item in self.peak_list:
            peak_list.append(float(item[1]))
        max_peak = max(peak_list)
        for item in self.peak_list:
            relative_peak_list.append(
                (item[0], cal_divide(float(item[1]), max_peak)))
            # relative_peak_list.append((item[0], float(item[1])/max_peak))
        # relative_peak_list = np.array(relative_peak_list)
        return relative_peak_list


# 从mgf文件中读取谱图，但这个方法很慢
def mgf_read(mgf_path):
    mass_spectra_dict = {}
    with open(mgf_path, 'r') as f:
        lines = f.readlines()
    # print('reading mgf data.')
    i = 0
    while i < len(lines):
        if 'BEGIN' in lines[i]:
            i += 1
            spectrum_name = lines[i].split('=')[1].strip()
            i += 1
            spectrum_charge = int(lines[i].split('=')[1][0])
            i += 2
            spectrum_pepmass = float(lines[i].split('=')[1])
            spectrum_peak_list = []
            while i < len(lines):
                i += 1
                if 'END' in lines[i]:
                    break
                spectrum_peak_list.append(lines[i].split())
            # print(spectrum_peak_list)
        spectrum = MassSpectrum(
            spectrum_charge, spectrum_pepmass, spectrum_peak_list)
        mass_spectra_dict[spectrum_name] = spectrum
        i += 1
        # break
    print('The number of spectra: ', len(mass_spectra_dict.keys()))
    return mass_spectra_dict

# 从mgf文件中读取谱图，但这个方法很慢


def mgf_read_for_ion(mgf_path):
    mass_spectra_dict = {}
    with open(mgf_path, 'r') as f:
        lines = f.readlines()
    # print('reading mgf data.')
    i = 0
    while i < len(lines):
        if 'BEGIN' in lines[i]:
            i += 1
            spectrum_name = lines[i].split('=')[1].strip()
            i += 1
            spectrum_charge = int(lines[i].split('=')[1][0])
            i += 2
            spectrum_pepmass = float(lines[i].split('=')[1])
            spectrum_peak_list = []
            while i < len(lines):
                i += 1
                if 'END' in lines[i]:
                    break
                spectrum_peak_list.append(lines[i].split())
            # print(spectrum_peak_list)
        spectrum = MassSpectrum_for_ion(
            spectrum_charge, spectrum_pepmass, spectrum_peak_list)
        mass_spectra_dict[spectrum_name] = spectrum
        i += 1
    print('The number of spectra: ', len(mass_spectra_dict.keys()))
    return mass_spectra_dict


def mgf_read_for_ion_pfind_filter(mgf_path, blind_res):
    mass_spectra_dict = {}
    with open(mgf_path, 'r') as f:
        lines = f.readlines()
    start_index_list = []
    # end_index_list = []

    blind_res_title_list = []
    for item in blind_res:
        # 底层是哈希表
        blind_res_title_list.append(item.split("\t")[0])

    print('reading mgf data.')

    mgf_content = "".join(lines)
    mgf_content = mgf_content.split("BEGIN IONS\n")
    mgf_content = mgf_content[1:]

    reserve_mgf_content = []
    sl = set(blind_res_title_list)
    dct = dict(zip(blind_res_title_list, blind_res_title_list))
    for i in tqdm(mgf_content):
        line = i.split("\n")
        line = [l for l in line if l != ""]
        t_spec_name = line[0].split("=")[1]
        if t_spec_name not in sl:
            continue
        else:
            # reserve_mgf_content.append(i)
            t_peak_list = []
            for index, l in enumerate(line):
                if l == "":
                    continue
                if "CHARGE" in l:
                    t_charge = int(l.split("=")[1][:-1])
                elif "TITLE" in l:
                    continue
                elif "PEPMASS" in l:
                    t_pepmass = float(l.split("=")[1])
                elif "RTINSECONDS" in l:
                    t_rt = float(l.split("=")[1])
                elif l[0].isdigit():
                    if re.match(r'^[+-]?\d+(\.\d+)?$', l.split(" ")[0]):
                        t_peak_list = line[index:-1]
                        break
            spectrum_peak_list = [j.split() for j in t_peak_list]
            # for j in t_peak_list:
            #     spectrum_peak_list.append(j.split())
            spectrum = MassSpectrum_for_ion1(
                t_charge, t_pepmass, spectrum_peak_list, t_rt)
            mass_spectra_dict[t_spec_name] = spectrum
    print('The number of spectra: ', len(mass_spectra_dict.keys()))
    return mass_spectra_dict


# 读取盲搜的结果
def blind_res_read(blind_res_path):
    # print(blind_res_path)
    with open(blind_res_path, 'r') as f:
        lines = f.readlines()
    return lines[1:]


# 筛选出含有指定修饰的PSM
def PSM_filter(blind_res, modification):
    filtered_res = []
    for line in blind_res:
        if modification in line:
            filtered_res.append(line)
    return filtered_res


def PSM_filter1(blind_res, modification, modification_site):
    filtered_res = []
    for line in blind_res:
        if modification+'.' in line:
            filtered_res.append(line)
    return filtered_res


# 给定修饰的位置和质量，修改质量数组
def mass_vector_modify(pos, mass, mass_vector):
    if pos == 0:
        mass_vector[0] += mass
    elif pos == len(mass_vector)+1:
        mass_vector[len(mass_vector)] += mass
    else:
        mass_vector[pos-1] += mass
    return mass_vector


# 生成质量数组，每个位置对应质量
def mass_vector_generation(peptide_sequence, mod_list, modification, accurate_mass, common_modification_dict):
    mass_vector = []

    # 生成原始的质量数组
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])
    # print(mass_vector)

    pos_mod_list = mod_list.split(';')[:-1]
    for pos_mod in pos_mod_list:
        pos, mod_name = pos_mod.split(',')
        if modification in mod_name:
            mod_pos = int(pos)
            mass_vector = mass_vector_modify(
                mod_pos, accurate_mass, mass_vector)
        else:
            mass_vector = mass_vector_modify(
                int(pos), common_modification_dict[mod_name], mass_vector)
    return mass_vector, mod_pos


# 位置校准
def position_correct(mod_pos, sequence_len):
    if mod_pos == 0:
        return 0
    elif mod_pos == sequence_len + 1:
        return sequence_len - 1
    else:
        return mod_pos - 1


# 生成b,y离子谱峰的数组
def b_y_vector_generation(mass_vector, mod_pos):
    mod_peak_list = []
    sequence_len = len(mass_vector)
    mass_sum_vector = [mass_vector[0]]

    for i in range(1, sequence_len):
        mass_sum_vector.append(mass_vector[i] + mass_sum_vector[i-1])
    # print(mass_vector)
    # print(mass_sum_vector)

    mod_pos = position_correct(mod_pos, sequence_len)

    # 生成含有修饰的b离子谱峰
    mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[mod_pos:]]
    # mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[mod_pos:]]
    # 生成含有修饰的y离子谱峰
    mod_peak_list += [mass_sum_vector[sequence_len-1] - mass +
                      h2o_mass + proton_mass for mass in mass_sum_vector[:mod_pos]]

    return mod_peak_list


# 统计质量偏差
def ion_diff_sum(mod_peak_list, peak_list):

    # ion_diff_counter = Counter()
    coarse_ion_diff_list = []
    ion_diff_list = []
    weight_ion_diff_list = []
    for mod_peak in mod_peak_list:
        # 质量偏差保留2位小数
        ion_diff_fine = [round(mod_peak - float(peak[0]), 6)
                         for peak in peak_list]
        ion_diff_list += ion_diff_fine
        weight_ion_diff_fine = [
            [round(mod_peak - float(peak[0]), 6), float(peak[1])] for peak in peak_list]
        weight_ion_diff_list += weight_ion_diff_fine
        ion_diff = [round(mass, 2) for mass in ion_diff_fine]
        # ion_diff_counter.update(ion_diff)
        coarse_ion_diff_list += ion_diff
    return coarse_ion_diff_list, ion_diff_list, weight_ion_diff_list


# 统计中性丢失的数目
def ion_type_compute(filtered_res, modification, accurate_mass, common_modification_dict, mass_spectra_dict):
    # total_ion_diff_counter = Counter()
    total_ion_diff_counter_list = []
    total_ion_diff_list = []
    total_weight_ion_diff_list = []
    #segment = int(len(filtered_res) / 10)
    #i = 0
    for line in tqdm(filtered_res):
        # if i % segment == 0:
        #    print('finished ', i / segment * 10,  'percetage')
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        # print(spectrum_name, peptide_sequence, mod_list)
        # 理论谱图
        mass_vector, mod_pos = mass_vector_generation(
            peptide_sequence, mod_list, modification, accurate_mass, common_modification_dict)
        # print(mass_vector, mod_pos)
        # 生成b/y离子有关的谱峰
        mod_peak_list = b_y_vector_generation(mass_vector, mod_pos)

        # print(mod_peak_list)
        if spectrum_name in mass_spectra_dict.keys():
            peak_list = mass_spectra_dict[spectrum_name].peak_list
        else:
            continue
        # ion_diff_counter, ion_diff_fine, weight_ion_diff = ion_diff_sum(mod_peak_list, peak_list)
        # 统计理论谱峰和实际谱峰的差值
        coarse_ion_diff, ion_diff_fine, weight_ion_diff = ion_diff_sum(
            mod_peak_list, peak_list)
        # total_ion_diff_counter.update(ion_diff_counter)
        total_ion_diff_counter_list += coarse_ion_diff
        total_ion_diff_list += ion_diff_fine
        total_weight_ion_diff_list += weight_ion_diff
        # i += 1
        # break
    total_ion_diff_counter = Counter(total_ion_diff_counter_list)
    # print(total_ion_diff_counter.most_common()[:10])
    return total_ion_diff_counter, total_ion_diff_list, total_weight_ion_diff_list


# 绘制频率曲线图
def freq_line_plot(total_ion_diff_counter):
    list_len = total_ion_diff_counter.most_common()[0][1]
    freq_list = [0] * (list_len + 1)
    x = [i for i in range(0, list_len+1)]
    for _, v in total_ion_diff_counter.items():
        freq_list[v] += 1
        # freq_list.append(v)
    # 画直方图太笼统
    #data = np.array(freq_list)
    # plt.hist(data,bins=20)
    plt.plot(x[1:], freq_list[1:])
    plt.xlabel('occur times')
    plt.ylabel('frequency')
    plt.show()


# 绘制论文里面的图
def freq_point_plot(total_ion_diff_counter, modification):
    xy_pair = []
    x = []
    y = []
    i = 0
    # for k, v in total_ion_diff_counter.items():
    #    xy_pair.append([k, v])

    # 选择top-n的画折线图更加好看
    for k, v in total_ion_diff_counter.most_common(10000):
        xy_pair.append([k, v])
    xy_pair = sorted(xy_pair, key=lambda k: k[0])

    for xy in xy_pair:
        x.append(xy[0])
        y.append(xy[1])

    with open(modification+'.txt', 'w', encoding='utf-8') as f:
        for xy in xy_pair:
            line = str(xy[0]) + '\t' + str(xy[1]) + '\n'
            f.write(line)

    plt.plot(x, y, 'r-', alpha=0.6)
    plt.xlabel('Offset')
    plt.ylabel('Counts')
    plt.xlim((200, 2000))
    # plt.xlim((200, 800))
    plt.show()

# 精准质量计算


def accurate_ion_mass_computation(coarse_mass, total_ion_diff_list):
    mass_sum = 0.0
    mass_num = 0
    for ion_diff in total_ion_diff_list:
        if(abs(ion_diff - coarse_mass) < 0.05):
            mass_sum += ion_diff
            mass_num += 1
    return mass_sum / mass_num


# 谱峰加权的精准质量计算
def weight_accurate_ion_mass_computation(coarse_mass, total_weight_ion_diff_list):
    mass_sum = 0.0
    mass_num = 0.0
    for ion_diff, peak in total_weight_ion_diff_list:
        if(abs(ion_diff - coarse_mass) < 0.01):
            mass_sum += ion_diff * peak
            mass_num += peak
    return mass_sum / mass_num


# 对离群点进行检测和分析
def freq_analysis(total_ion_diff_counter):
    counter_list = []
    for k, v in total_ion_diff_counter.items():
        counter_list.append([k, v])
    counter_list = sorted(counter_list, key=lambda x: x[1], reverse=True)
    print(counter_list[:10])
    print(len(counter_list))
    counter_list = counter_cluster(counter_list)
    counter_list = sorted(counter_list, key=lambda x: x[1], reverse=True)
    print(counter_list[:10])
    value_counter_list = [p[1] for p in counter_list]
    arr_mean = np.mean(value_counter_list)
    # 方差 np.var
    arr_var = np.std(value_counter_list, ddof=1)
    print(arr_mean, arr_var)


# 对近似点进行聚类
def counter_cluster(counter_list):
    new_counter_list = []
    for pair in counter_list:
        if pair[1] > 1:
            flag = False
            for cur_pair in new_counter_list:
                if abs(pair[0]-cur_pair[0]) < 0.02:
                    cur_pair[1] += pair[1]
                    flag = True
                    break
            if flag == False:
                new_counter_list.append(pair)
        # else:
        #    new_counter_list.append(pair)
    return new_counter_list


# 特征离子确定 (使用了全集，不正确)
def old_feature_peak_determine(mass_spectra_dict):
    position_list = []
    for key in mass_spectra_dict:
        cur_position_list = [round(float(p[0]), 2)
                             for p in mass_spectra_dict[key].peak_list]
        position_list += cur_position_list

    position_counter = Counter(position_list)
    print(position_counter.most_common()[:10])


def plot_mass_distribution(peak_dict):

    for key in peak_dict.keys():
        result = dict(Counter(peak_dict[key]))
        # print(result)
        result_sort = sorted(result.items(), key=lambda x: x[0])
        # print("\n")
        # print(result_sort)
        # print(len(result_sort))

        x1 = []
        y1 = []
        for x_y in result_sort:
            print(x_y)
            x1.append(x_y[0])
            y1.append(x_y[1])
        width = 1
        indexes = np.arange(len(y1))
        # plt.bar(indexes, y1, width)

        plt.plot(x1, y1, 'c*-')
        # plt.xticks(indexes + width * 3, y1)
        plt.xlabel('intensity')
        plt.ylabel('count')
        # plt.title(mod + "_" + str(len(mod_char_ion_dict[mod])))
        plt.show()
        # plt.savefig('./ion_intensity_img/{}_{}.png'.format(mod, str(len(mod_char_ion_dict[mod]))), bbox_inches='tight')
        plt.close()

# 返回指定谱图系列filtered_res的特征离子


def feature_peak_determine(mass_spectra_dict, filtered_res):
    # 保留6位小数的
    fine_position_list = []
    coarse_position_list = []
    # 保存谱峰的精确质量
    peak_dict = {}
    for line in filtered_res:
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        if spectrum_name in mass_spectra_dict.keys():
            peak_list = mass_spectra_dict[spectrum_name].peak_list
        else:
            continue
        fine_position_list += [float(p[0]) for p in peak_list]
        # 是否应当按照±10ppm的精度来看待特征离子
        coarse_position_list += [round(float(p[0]), 2) for p in peak_list]
    position_counter = Counter(coarse_position_list)
    filtered_position_list = [p[0]
                              for p in position_counter.most_common()[:300]]
    filtered_position_counter_list = [
        (p[0], p[1]) for p in position_counter.most_common()[:300]]
    for position in filtered_position_list:
        peak_dict[position] = []
    for position in fine_position_list:
        peak = round(position, 2)
        if peak in peak_dict.keys():
            peak_dict[peak].append(position)
    # plot_mass_distribution(peak_dict)
    for key in peak_dict.keys():
        peak_dict[key] = np.mean(peak_dict[key])
    # print(peak_dict)

    return filtered_position_list, peak_dict


def b_y_vector_generation_for_ion(mass_vector, mod_pos):
    mod_peak_list = []
    sequence_len = len(mass_vector)
    mass_sum_vector = [mass_vector[0]]

    for i in range(1, sequence_len):
        mass_sum_vector.append(mass_vector[i] + mass_sum_vector[i-1])
    # print(mass_vector)
    # print(mass_sum_vector)

    mod_pos = position_correct(mod_pos, sequence_len)

    # 生成含有修饰的b离子谱峰
    mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[mod_pos:]]
    # 生成含有修饰的y离子谱峰
    mod_peak_list += [mass_sum_vector[sequence_len-1] - mass +
                      h2o_mass + proton_mass for mass in mass_sum_vector[:mod_pos]]

    return mod_peak_list


def mass_vector_generation_for_no_mod_ion(peptide_sequence):
    mass_vector = []
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])

    return mass_vector


def mass_vector_generation_for_mod_ion(peptide_sequence, mod_list, modification, common_modification_dict, modification_dict):
    mass_vector = []

    # 生成原始的质量数组
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])
    # print(mass_vector)
    exist_flag = False
    pos_mod_list = mod_list.split(';')[:-1]
    for pos_mod in pos_mod_list:
        pos, mod_name = pos_mod.split(',')
        if modification+"." in mod_name:
            mod_pos = int(pos)
            mass_vector = mass_vector_modify(
                mod_pos, modification_dict[modification], mass_vector)
            exist_flag = True
        else:
            mass_vector = mass_vector_modify(
                int(pos), common_modification_dict[mod_name], mass_vector)
    return mass_vector, mod_pos


def mass_vector_generation_for_without_mod_ion(peptide_sequence, mod_list, modification, common_modification_dict, modification_dict):
    mass_vector = []

    # 生成原始的质量数组
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])
    # print(mass_vector)
    exist_flag = False
    pos_mod_list = mod_list.split(';')[:-1]
    for pos_mod in pos_mod_list:
        pos, mod_name = pos_mod.split(',')
        if modification+"." in mod_name:
            mod_pos = int(pos)
            mass_vector = mass_vector_modify(
                mod_pos, modification_dict[modification], mass_vector)
            exist_flag = True
        else:
            mass_vector = mass_vector_modify(
                int(pos), common_modification_dict[mod_name], mass_vector)
    return mass_vector


@jit
def cal_b_mass(mass, charge, proton_mass):
    return (mass+proton_mass*charge)/charge


@jit
def cal_y_mass(mass_sum, mass, charge, proton_mass, h2o_mass):
    return (mass_sum - mass + h2o_mass + charge*proton_mass)/charge


def b_y_vector_generation_for_no_mod_ion(mass_vector, charge):
    mod_peak_list = []
    sequence_len = len(mass_vector)
    mass_sum_vector = [mass_vector[0]]

    for i in range(1, sequence_len):
        mass_sum_vector.append(mass_vector[i] + mass_sum_vector[i-1])
    # print(mass_vector)
    # print(mass_sum_vector)

    # # 生成b离子谱峰
    # mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[:]]
    # # 生成y离子谱峰
    # mod_peak_list += [mass_sum_vector[sequence_len-1] - mass +
    #                   h2o_mass + proton_mass for mass in mass_sum_vector[:-1]]
    # if charge > 1:
    #     for i in range(2, charge+1):
    #         mod_peak_list += [(mass + i*proton_mass) /
    #                           i for mass in mass_sum_vector[:]]
    #         mod_peak_list += [(mass_sum_vector[sequence_len-1] - mass +
    #                            h2o_mass + i*proton_mass)/i for mass in mass_sum_vector[:-1]]

    for i in range(1, charge+1):
        mod_peak_list += [cal_b_mass(mass, i, proton_mass)
                          for mass in mass_sum_vector[:]]
        mod_peak_list += [cal_y_mass(mass_sum_vector[sequence_len-1],
                                     mass, i, proton_mass, h2o_mass) for mass in mass_sum_vector[:]]
    return mod_peak_list


def b_y_vector_generation_for_mod_ion(mass_vector, charge, mod_pos):
    mod_peak_list = []
    sequence_len = len(mass_vector)
    mass_sum_vector = [mass_vector[0]]

    for i in range(1, sequence_len):
        mass_sum_vector.append(mass_vector[i] + mass_sum_vector[i-1])

    # # 生成b离子谱峰
    # mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[:]]

    # # 生成y离子谱峰
    # mod_peak_list += [mass_sum_vector[sequence_len-1] - mass +
    #                   h2o_mass + proton_mass for mass in mass_sum_vector[:-1]]

    neighbor_peak_b = []
    neighbor_peak_y = []
    mod_peak_list_b = []
    mod_peak_list_y = []
    mod_peak_list_b_c = {}
    mod_peak_list_y_c = {}
    for i in range(1, charge+1):
        mod_peak_list_b_c[i] = []
        mod_peak_list_y_c[i] = []

    for i in range(1, charge+1):
        tmp = [cal_b_mass(mass, i, proton_mass)
               for mass in mass_sum_vector[:]]
        mod_peak_list_b += tmp
        mod_peak_list_b_c[i] += tmp

    for i in range(1, charge+1):
        tmp = [cal_y_mass(mass_sum_vector[sequence_len-1],
                          mass, i, proton_mass, h2o_mass) for mass in mass_sum_vector[:]]
        mod_peak_list_y += tmp[:-1]
        mod_peak_list_y_c[i] += tmp[:-1]

    mod_peak_list += mod_peak_list_b
    mod_peak_list += mod_peak_list_y
    return mod_peak_list, mod_peak_list_b_c, mod_peak_list_y_c


@jit(nopython=True)
def match_peak(i, delete_peak):
    for j in delete_peak:
        if calc_ppm(j, i) <= 20:
            return True


def delete_peak(peak_list, delete_peak_list):
    filter_peak_list = []
    a = List([1.0])
    for i in delete_peak_list:
        a.append(i)
    a = a[1:]
    for i in peak_list:
        if match_peak(float(i[0]), a):
            filter_peak_list.append(i)

    for i in filter_peak_list:
        if i in peak_list:
            peak_list.remove(i)
    return peak_list


def delete_peak_for_mod(peak_list, delete_peak_list, mass_vector, mod_pos, mod_peak_list_b_c, mod_peak_list_y_c):
    filter_peak_list = []

    a = List([1.0])
    for i in delete_peak_list:
        a.append(i)
    a = a[1:]
    for i in peak_list:
        if match_peak(float(i[0]), a):
            filter_peak_list.append(i)

    reserve_flag = False
    if mod_pos == 1:
        # y_n
        y_n_l = [mod_peak_list_b_c[i][-1] for i in mod_peak_list_b_c.keys()]
        # y_(n-1)
        y_n1_l = [mod_peak_list_y_c[i][0] for i in mod_peak_list_y_c.keys()]
        b1_l = [mod_peak_list_b_c[i][0] for i in mod_peak_list_b_c.keys()]
        for i in b1_l:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    reserve_flag = True
                    break
                if reserve_flag:
                    break
            if reserve_flag:
                break
        if reserve_flag == False:
            y_n_flag = False
            y_n1_flag = False
            for i in y_n_l:
                for j in filter_peak_list:
                    if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                        y_n_flag = True
                        break
                    if y_n_flag:
                        break
                if y_n_flag:
                    break
            for i in y_n1_l:
                for j in filter_peak_list:
                    if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                        y_n1_flag = True
                        break
                    if y_n1_flag:
                        break
                if y_n1_flag:
                    break
            if y_n_flag and y_n1_flag:
                reserve_flag = True
    elif mod_pos == len(mass_vector):
        y_1_l = [mod_peak_list_y_c[i][-1] for i in mod_peak_list_y_c.keys()]
        b_n1_l = [mod_peak_list_b_c[i][-2] for i in mod_peak_list_b_c.keys()]
        b_n_l = [mod_peak_list_b_c[i][-1] for i in mod_peak_list_b_c.keys()]
        for i in y_1_l:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    reserve_flag = True
                    break
                if reserve_flag:
                    break
            if reserve_flag:
                break
        if reserve_flag == False:
            b_n_flag = False
            b_n1_flag = False
            for i in b_n_l:
                for j in filter_peak_list:
                    if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                        b_n_flag = True
                        break
                    if b_n_flag:
                        break
                if b_n_flag:
                    break
            for i in b_n1_l:
                for j in filter_peak_list:
                    if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                        b_n1_flag = True
                        break
                    if b_n1_flag:
                        break
                if b_n1_flag:
                    break
            if b_n_flag and b_n1_flag:
                reserve_flag = True
    else:
        b_pos_l = [mod_peak_list_b_c[i][mod_pos-2]
                   for i in mod_peak_list_b_c.keys()]
        b_pos_r = [mod_peak_list_b_c[i][mod_pos-1]
                   for i in mod_peak_list_b_c.keys()]
        y_pos_l = [mod_peak_list_y_c[i][-(len(mass_vector)-mod_pos+1)]
                   for i in mod_peak_list_y_c.keys()]
        y_pos_r = [mod_peak_list_y_c[i][-(len(mass_vector)-mod_pos)]
                   for i in mod_peak_list_y_c.keys()]
        b_posl_flag = False
        b_posr_flag = False
        y_posl_flag = False
        y_posr_flag = False
        for i in b_pos_l:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    b_posl_flag = True
                    break
                if b_posl_flag:
                    break
            if b_posl_flag:
                break

        for i in b_pos_r:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    b_posr_flag = True
                    break
                if b_posr_flag:
                    break
            if b_posr_flag:
                break

        for i in y_pos_l:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    y_posl_flag = True
                    break
                if y_posl_flag:
                    break
            if y_posl_flag:
                break

        for i in y_pos_r:
            for j in filter_peak_list:
                if abs(i-float(j[0]))/(i+0.000001)*1000000 <= 20:
                    y_posr_flag = True
                    break
                if y_posr_flag:
                    break
            if y_posr_flag:
                break

        if y_posl_flag and y_posr_flag:
            reserve_flag = True
        elif b_posl_flag and b_posr_flag:
            reserve_flag = True
        elif b_posr_flag and y_posl_flag:
            reserve_flag = True

    for i in filter_peak_list:
        if i in peak_list:
            peak_list.remove(i)
    return peak_list, reserve_flag


@jit
def calc_ppm(a, b):
    return abs((a-b)/a)*1000000

# 返回指定谱图系列filtered_res的特征离子
def feature_peak_determine_for_ion(mass_spectra_dict, filtered_res, common_modification_dict, mod_flag, modification_dict, modification, close_ion, ion_relative_mode, modification_site, pchem_output_path, ion_filter_mode):
    # 保留6位小数的
    fine_position_list = []
    coarse_position_list = []
    fine_position_peak_list = []
    coarse_position_peak_list = []
    fine_position_relative_peak_list = []
    coarse_position_relative_peak_list = []
    # 保存谱峰的精确质量
    peak_dict = {}
    fine_peak_dict = {}
    fine_relative_peak_dict = {}
    filtered_res_tmp = []
    if mod_flag == True:
        for line in filtered_res:
            if line == "\n":
                break
            t_line = line.split("\t")
            t_seq = t_line[5]
            t_mod_list = t_line[10]
            t_mod_list = t_mod_list.split(";")
            t_mod_list = t_mod_list[:-1]
            for i in t_mod_list:
                if modification+'.' in i:
                    t_mod = i.split(",")
                    t_mod_site = t_seq[int(t_mod[0])-1]

                    if t_mod_site not in modification_site[modification]:
                        if (t_mod[0] == '1') and ('N-SIDE' in modification_site[modification]):
                            filtered_res_tmp.append(line)
                        elif (t_mod[0] == str(len(t_seq))) and ('C-SIDE' in modification_site[modification]):
                            filtered_res_tmp.append(line)
                        else:
                            continue
                    else:
                        filtered_res_tmp.append(line)

        filtered_res = filtered_res_tmp
    ion_common_list_all = []

    reserve_final_dict = {}

    have_close_ion_scan_dict = {}

    center_bins = []
    pr_dict = {}
    if int(close_ion[-1]) % 2 == 0:
        center_bins = [str(round(num, 3))
                       for num in np.arange(100, 5000, 0.002)]

    else:
        center_bins = [str(round(num, 3))
                       for num in np.arange(100.001, 5000.001, 0.002)]

    for line in tqdm(filtered_res):
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]

        if spectrum_name not in mass_spectra_dict.keys():
            continue
        else:
            # 对于不含修饰的谱图，需要去掉b、y离子
            if mod_flag == False:
                peak_list = mass_spectra_dict[spectrum_name].peak_list
                max_peak = max([float(i[1]) for i in peak_list])
                mass_vector = mass_vector_generation_for_no_mod_ion(
                    peptide_sequence)
                # print(mass_vector, mod_pos)
                # 生成b/y离子有关的谱峰
                delete_peak_list = b_y_vector_generation_for_no_mod_ion(
                    mass_vector, mass_spectra_dict[spectrum_name].charge)
                peak_list = delete_peak(peak_list, delete_peak_list)
                peak_list = [i for i in peak_list if float(i[0]) < 4800]
                ion_common_list = []

                relative_peak_list = []
                if ion_relative_mode == 1:
                    for item in peak_list:
                        relative_peak_list.append(
                            (item[0], cal_divide(float(item[1]), max_peak)))


            else:
                peak_list = mass_spectra_dict[spectrum_name].peak_list
                max_peak = max([float(i[1]) for i in peak_list])
                mass_vector, mod_pos = mass_vector_generation_for_mod_ion(
                    peptide_sequence, mod_list, modification, common_modification_dict, modification_dict)
                delete_peak_list, mod_peak_list_b_c, mod_peak_list_y_c = b_y_vector_generation_for_mod_ion(
                    mass_vector, mass_spectra_dict[spectrum_name].charge, mod_pos)
                peak_list, reserve_flag = delete_peak_for_mod(
                    peak_list, delete_peak_list, mass_vector, mod_pos, mod_peak_list_b_c, mod_peak_list_y_c)
                peak_list = [i for i in peak_list if float(i[0]) < 4800]
                if reserve_flag:
                    reserve_final_dict[spectrum_name] = True
                else:
                    reserve_final_dict[spectrum_name] = False

                ion_common_list = []

                relative_peak_list = []

                if ion_relative_mode == 1:
                    for item in peak_list:

                        relative_peak_list.append(
                            (item[0], cal_divide(float(item[1]), max_peak)))


        close_ion_count = 0
        close_ion_index = []
        close_ion_value = []
        for idx, pl in enumerate(peak_list):
            if abs(float(close_ion) - float(pl[0])) <= 0.002:
                # if round(float(close_ion), 2) == round(float(pl[0]), 2):
                close_ion_count += 1
                close_ion_index.append(idx)
                close_ion_value.append(float(pl[0]))
            if round(float(pl[0]), 2) > round(float(close_ion), 2) + 2:
                break
        if close_ion_count > 1:
            target_close_ion_index = close_ion_index[close_ion_value.index(
                min(close_ion_value, key=lambda x: abs(x-float(close_ion))))]
            for ci in reversed(close_ion_index):
                if ci != target_close_ion_index:
                    del(peak_list[ci])
                    del(relative_peak_list[ci])
            # print("")

        fine_position_list += [float(p[0]) for p in peak_list]
        fine_position_peak_list += [(float(p[0]), float(p[1]))
                                    for p in peak_list]
        fine_position_relative_peak_list += [(float(p[0]), float(p[1]))
                                             for p in relative_peak_list]
        # 是否应当按照±0.001的精度来看待特征离子
        for p, pr in zip(peak_list, relative_peak_list):
            p_t = float(p[0])
            index_t = int((p_t - float(center_bins[0])) // 0.002)  # 向下取整
            index_t1 = index_t+1
            # p_label = find_closest_value(center_bins, 0.002, p_t)
            p_label = center_bins[index_t]
            p_label1 = center_bins[index_t1]
            if abs(float(p_label) - p_t) <= abs(float(p_label1) - p_t):
                pr_dict[float(p[0])] = p_label
                coarse_position_list.append(center_bins[index_t])
                coarse_position_peak_list.append(
                    (p_label, float(p[1])))
                coarse_position_relative_peak_list.append(
                    (p_label, float(pr[1])))
            else:
                pr_dict[float(p[0])] = p_label1
                coarse_position_list.append(center_bins[index_t1])
                coarse_position_peak_list.append(
                    (p_label1, float(p[1])))
                coarse_position_relative_peak_list.append(
                    (p_label1, float(pr[1])))


    position_counter = Counter(coarse_position_list)
    filtered_position_list = [p[0]
                              for p in position_counter.most_common()[:300]]
    filtered_position_counter_list = [
        (p[0], p[1]) for p in position_counter.most_common()[:300]]
    for position in filtered_position_list:
        peak_dict[position] = []
        fine_peak_dict[position] = []
        fine_relative_peak_dict[position] = []
    for position, position1 in zip(fine_position_peak_list, fine_position_relative_peak_list):
        peak = pr_dict[position[0]]
        if peak in peak_dict.keys():
            peak_dict[peak].append(position[0])
            fine_peak_dict[peak].append((position[0], position[1]))
            fine_relative_peak_dict[peak].append((position1[0], position1[1]))

    peak_dict1 = {}
    for key in peak_dict.keys():
        peak_dict1[key] = np.mean(peak_dict[key])

    common_ion_list = []
    ion_relative_peak_distribution = {}
    ion_common_dict = {}
    for ion_key in ion_common_dict.keys():
        common_ion_list.append(ion_common_dict[ion_key])
    for ion in common_ion_list:

        index_t = int((ion - float(center_bins[0])) // 0.002)  # 向下取整
        index_t1 = index_t+1
        p_label = center_bins[index_t]
        p_label1 = center_bins[index_t1]
        if abs(float(p_label) - ion) <= abs(float(p_label1) - ion):
            ion_label = p_label
        else:
            ion_label = p_label1
        if ion_label in fine_relative_peak_dict.keys():
            ion_relative_peak_distribution[round(
                ion, 6)] = fine_relative_peak_dict[ion_label]
        else:
            ion_relative_peak_distribution[round(ion, 6)] = []

    close_ion_index = int((float(close_ion) - float(center_bins[0])) // 0.002)
    close_ion_index1 = close_ion_index + 1
    close_ion_label = center_bins[close_ion_index]
    close_ion_label1 = center_bins[close_ion_index1]
    if abs(float(close_ion_label) - float(close_ion)) <= abs(float(close_ion_label1) - float(close_ion)):
        close_ion_refine = close_ion_label
    else:
        close_ion_refine = close_ion_label1
    if close_ion_refine in fine_relative_peak_dict.keys():
        ion_relative_peak_distribution[float(
            close_ion)] = fine_relative_peak_dict[close_ion_refine]
    else:
        ion_relative_peak_distribution[close_ion] = []

    if mod_flag:
        ion_area_dict = {}
        n_bins_dict = {}
        if int(ion_filter_mode) == 2:
            distribution_save_path = os.path.join(
                pchem_output_path, "distribution")
            if not os.path.exists(distribution_save_path):
                os.makedirs(distribution_save_path)
            n_bins_dict = draw_distribution_for_ions(
                filtered_res, ion_relative_peak_distribution, distribution_save_path, "{}_ion_peak.png".format(modification))
            # draw_cumulative_distribution_for_ions(
            #     n_bins_dict, filtered_res, ion_relative_peak_distribution, distribution_save_path, "{}_ion_cum_peak.png".format(modification))
            ion_area_dict = calc_cumulative_distribution_area(n_bins_dict)

        return filtered_position_list, filtered_position_counter_list, peak_dict1, fine_peak_dict, fine_relative_peak_dict, filtered_res, reserve_final_dict, ion_area_dict, n_bins_dict, have_close_ion_scan_dict
    else:
        ion_area_dict = {}
        n_bins_dict = {}
        if int(ion_filter_mode) == 2:
            distribution_save_path = os.path.join(
                pchem_output_path, "distribution")
            if not os.path.exists(distribution_save_path):
                os.makedirs(distribution_save_path)
            n_bins_dict = draw_distribution_for_ions(
                filtered_res, ion_relative_peak_distribution, distribution_save_path, "without_mod_ion_peak.png")
            # draw_cumulative_distribution_for_ions(
            #     n_bins_dict, filtered_res, ion_relative_peak_distribution, distribution_save_path, "without_mod_ion_cum_peak.png")
            ion_area_dict = calc_cumulative_distribution_area(n_bins_dict)
        return filtered_position_list, filtered_position_counter_list, peak_dict1, fine_peak_dict, fine_relative_peak_dict, filtered_res, ion_area_dict, n_bins_dict, have_close_ion_scan_dict


def calc_cumulative_distribution_area(n_bins_dict):
    ion_area_dict = {}
    for ion in n_bins_dict.keys():
        y = n_bins_dict[ion]['y']
        ion_area_dict[ion] = sum(y)
    return ion_area_dict


def draw_distribution_for_ions(filtered_res, ion_relative_peak_distribution, distribution_save_path, img_name):
    plt.figure(dpi=300, figsize=(26, 8))
    all_num = len(filtered_res)
    color_list = ["#63b2ee", "#76da91", "#f8cb7f", "#9192ab", "#9370DB"]
    n_bins_dict = {}
    plt.subplot(1, 2, 1)
    for index, ion in enumerate(ion_relative_peak_distribution.keys()):
        ion_mass_list = []
        ion_relative_peak_list = [i[1]
                                  for i in ion_relative_peak_distribution[ion]]
        ion_relative_peak_list.sort()

        if all_num - len(ion_relative_peak_list) > 0:
            tmp_peak = [0.0 for i in range(
                all_num - len(ion_relative_peak_list))] + [i for i in ion_relative_peak_list]
        else:
            tmp_peak = [i for i in ion_relative_peak_list]

        # n, bins, patches = plt.hist(tmp_peak, bins=100, label=str(ion), alpha=0.3, align='mid',
        #                            color=color_list[index], edgecolor='#999999', rwidth=0.9, weights=[1./len(tmp_peak)]*len(tmp_peak))
        t_bins = np.linspace(0, 1, 100)
        if index != len(ion_relative_peak_distribution.keys()) - 1:
            n, bins, patches = plt.hist(tmp_peak, bins=t_bins, label=str(ion), align='mid',
                                        edgecolor=color_list[index], weights=[1./len(tmp_peak)]*len(tmp_peak), histtype='step')
        else:
            n, bins, patches = plt.hist(tmp_peak, bins=t_bins, label=str(ion), align='mid',
                                        edgecolor="r", weights=[1./len(tmp_peak)]*len(tmp_peak), histtype='step')
        center = (bins[:-1] + bins[1:]) / 2  # 计算每个bin的中心位置
        t_n = [sum(n[:index+1]) for index in range(len(n))]
        n_bins_dict[ion] = {}
        n_bins_dict[ion]['y'] = t_n
        n_bins_dict[ion]['x'] = center
        n_bins_dict[ion]['distribute'] = n
        n_bins_dict[ion]['peaks'] = tmp_peak

    plt.legend()
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)
    plt.title("Distribution")

    plt.subplot(1, 2, 2)
    for index, ion in enumerate(ion_relative_peak_distribution.keys()):
        y = n_bins_dict[ion]['y']
        x = n_bins_dict[ion]['x']
        if index != len(ion_relative_peak_distribution.keys()) - 1:
            plt.plot(x, y, label=str(ion),
                     color=color_list[index], marker='+')
        else:
            plt.plot(x, y, label=str(ion),
                     color='r', marker='+')
    plt.legend()
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)
    plt.title("Cum-Distribution")
    plt.savefig(os.path.join(distribution_save_path, img_name))
    plt.close()
    return n_bins_dict


def draw_cumulative_distribution_for_ions(n_bins_dict, filtered_res, ion_relative_peak_distribution, distribution_save_path, img_name):
    plt.figure(dpi=300, figsize=(16, 8))
    all_num = len(filtered_res)
    color_list = ["#63b2ee", "#76da91", "#f8cb7f", "#9192ab", "r"]

    for index, ion in enumerate(ion_relative_peak_distribution.keys()):
        y = n_bins_dict[ion]['y']
        x = n_bins_dict[ion]['x']
        plt.plot(x, y, label=str(ion),
                 color=color_list[index], marker='+')
    plt.legend()
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)
    plt.savefig(os.path.join(distribution_save_path, img_name))
    plt.close()


def ppm_calculate(a, b, mass_diff_diff):
    return abs(abs(b-a)-mass_diff_diff)/(mass_diff_diff+0.000001)*1000000


def feature_pair_find(position_list, peak_dict, mass_diff):
    # print(position_list)
    coarse_mass_diff = round(mass_diff, 2)
    pair_list = []
    for light_mass in position_list[0]:
        for heavy_mass in position_list[1]:
            if round(heavy_mass - light_mass, 2) == coarse_mass_diff:
                ppm_error = ppm_calculate(
                    peak_dict[0][light_mass], peak_dict[1][heavy_mass], mass_diff)
                pair_list.append([light_mass, heavy_mass, int(ppm_error)])
    print(pair_list)
    return pair_list


# 读取mgf文件中所有的谱图
def read_mgf(parameter_dict):
    mass_spectra_dict = {}
    for msms_path in parameter_dict['msms_path']:
        mgf_path = msms_path.split('=')[1].split('.')[0] + '.mgf'
        cur_mass_spectra_dict = mgf_read(mgf_path)
        mass_spectra_dict.update(cur_mass_spectra_dict)
    return mass_spectra_dict


def ion_type_determine(current_path, modification_list, modification_dict, mass_spectra_dict, blind_res, close_ion, ion_relative_mode, modification_site, ion_filter_mode, pchem_output_path):
    pchem_cfg_path = os.path.join(current_path, 'pChem.cfg')
    parameter_dict = parameter_file_read(pchem_cfg_path)
    print(parameter_dict)

    # 读取常见修饰的列表
    common_modification_dict = common_dict_create(current_path)
    # print(common_modification_dict)
    position_list = []
    peak_dict = []

    # 修饰->中性丢失
    mod2ion = {}
    exist_ion_flag = True
    ion_list = []
    ion_dict = {}
    mod_have_close_ion_scan_dict = {}
    # 筛选有效的PSM
    for modification in modification_list:
        # pfind-filtered line
        mod = modification.split('_')[2]
        int_mod = int(mod)
        mod2ion[mod] = []
        t_ion_list = []
        filtered_res = PSM_filter1(blind_res, modification, modification_site)
        # 只保留psm>=5的这些修饰
        if len(filtered_res) >= 5 and int(ion_filter_mode) == 2:
            print('calculate the ion of ', modification)
            # 确定报告每个未知修饰的离子，modification in modification_list
            # t_position_list：前300个候选特征离子；
            # t_peak_dict：前300个特征离子每个特征离子的精确质量
            ion_dict[modification] = {}
            t_position_list, t_position_counter_list, t_peak_dict, t_fine_peak_dict, t_fine_relative_peak_dict, filtered_res, reserve_final_dict, ion_area_dict, n_bins_dict, t_mod_have_close_ion_scan_dict = feature_peak_determine_for_ion(
                mass_spectra_dict, filtered_res, common_modification_dict, True, modification_dict, modification, close_ion, ion_relative_mode, modification_site, pchem_output_path, ion_filter_mode)
            ion_dict[modification]['t_position_list'] = t_position_list
            ion_dict[modification]['t_position_counter_list'] = t_position_counter_list
            ion_dict[modification]['t_peak_dict'] = t_peak_dict
            ion_dict[modification]['t_fine_peak_dict'] = t_fine_peak_dict
            ion_dict[modification]['t_fine_relative_peak_dict'] = t_fine_relative_peak_dict
            ion_dict[modification]['mod_psm'] = len(filtered_res)
            ion_dict[modification]['reserve_dict'] = reserve_final_dict
            ion_dict[modification]['ion_area_dict'] = ion_area_dict
            ion_dict[modification]['n_bins_dict'] = n_bins_dict
            # mod_have_close_ion_scan_dict = dict(
            #     mod_have_close_ion_scan_dict, **t_mod_have_close_ion_scan_dict)
        elif int(ion_filter_mode) == 1:
            print('calculate the ion of ', modification)
            ion_dict[modification] = {}
            t_position_list, t_position_counter_list, t_peak_dict, t_fine_peak_dict, t_fine_relative_peak_dict, filtered_res, reserve_final_dict, ion_area_dict, n_bins_dict, t_mod_have_close_ion_scan_dict = feature_peak_determine_for_ion(
                mass_spectra_dict, filtered_res, common_modification_dict, True, modification_dict, modification, close_ion, ion_relative_mode, modification_site, pchem_output_path, ion_filter_mode)
            ion_dict[modification]['t_position_list'] = t_position_list
            ion_dict[modification]['t_position_counter_list'] = t_position_counter_list
            ion_dict[modification]['t_peak_dict'] = t_peak_dict
            ion_dict[modification]['t_fine_peak_dict'] = t_fine_peak_dict
            ion_dict[modification]['t_fine_relative_peak_dict'] = t_fine_relative_peak_dict
            ion_dict[modification]['mod_psm'] = len(filtered_res)
            ion_dict[modification]['reserve_dict'] = reserve_final_dict
            ion_dict[modification]['ion_area_dict'] = ion_area_dict
            ion_dict[modification]['n_bins_dict'] = n_bins_dict
            # mod_have_close_ion_scan_dict = dict(
            #     mod_have_close_ion_scan_dict, **t_mod_have_close_ion_scan_dict)
        else:
            continue
    return ion_dict, mod_have_close_ion_scan_dict


# 计算信噪比，然后返回是否符合阈值
def signal_noise_filter(ion_mass, total_ion_diff_counter, mass_range=50, threshold=2.0):
    noise_num = 0
    noise_intensity = 0.0
    signal_intensity = 0.0
    min_mass = max(0, ion_mass - mass_range)
    max_mass = ion_mass + mass_range
    name_list = [k for k, _ in total_ion_diff_counter.most_common()[:20]]

    for k, v in total_ion_diff_counter.items():
        if abs(k - ion_mass) < 1.0:
            signal_intensity += v
            continue
        if k > min_mass and k < max_mass and k not in name_list:
            noise_num += 1
            noise_intensity += v

    noise_intensity = float(noise_intensity / (max_mass - min_mass))
    if noise_intensity > 1.0:
        signal2noise = float(signal_intensity / noise_intensity)
    else:
        signal2noise = float(signal_intensity)
    # print(ion_mass, signal2noise)
    return signal2noise >= threshold



# 筛选出没有修饰的PSM
def without_mod_PSM_filter(current_path, blind_res, mass_spectra_dict, modification_dict, close_ion, ion_relative_mode, pchem_output_path, ion_filter_mode, modification_site):
    without_mod_res = []
    start_time = time.time()
    for line in blind_res:
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        if mod_list == "":
            if spectrum_name in mass_spectra_dict.keys():
                without_mod_res.append(line)
    common_modification_dict = common_dict_create(current_path)
    without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict, filtered_res, ion_area_dict, n_bins_dict, without_mod_have_close_ion_scan_dict = feature_peak_determine_for_ion(
        mass_spectra_dict, without_mod_res, common_modification_dict, False, modification_dict, NULL, close_ion, ion_relative_mode, modification_site, pchem_output_path, ion_filter_mode)
    end_time = time.time()
    print('filter without-mod PSM cost time (s): ',
          round(end_time - start_time, 1))
    return without_mod_res, without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict, ion_area_dict, n_bins_dict, without_mod_have_close_ion_scan_dict


def get_modification_from_result(pchem_summary_path):
    with open(pchem_summary_path, 'r') as f:
        lines = f.readlines()
    result = lines[1:]
    modification_list = []
    modification_dict = {}
    modification_PSM = {}
    modification_site = {}
    for i in result:
        if i == "\n":
            break
        t_l_result = i.split('\t')
        modification_name = t_l_result[1]
        accurate_mass = t_l_result[2].split(' ')[0]
        modification_psm = t_l_result[-1][:-1]
        modification_site_list = []
        modification_site_list.append(t_l_result[3].split('|')[0])
        if t_l_result[4] != "":
            others = t_l_result[4].split(";")[:-1]
            for i in others:
                modification_site_list.append(i.split("(")[0].split(" ")[-1])
        modification_list.append(modification_name)
        modification_dict[modification_name] = float(accurate_mass)
        modification_PSM[modification_name] = int(modification_psm)
        modification_site[modification_name] = modification_site_list
    return modification_list, modification_dict, modification_PSM, modification_site


def deal_without_mod_psm(pchem_output_path, current_path, ion_type, modification_list, modification_dict, blind_res, mass_spectra_dict, modification_PSM, modification_site, pchem_summary_path, ion_relative_mode, ion_rank_threshold, ion_filter_mode):
    without_mod_ion_result_path = os.path.join(
        pchem_output_path, 'pChem_without_mod_ion_result.summary')

    without_mod_res, without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict, ion_area_dict, n_bins_dict, without_mod_have_close_ion_scan_dict = without_mod_PSM_filter(
        current_path, blind_res, mass_spectra_dict, modification_dict, ion_type, ion_relative_mode, pchem_output_path, ion_filter_mode, modification_site)
    without_mod_relative_peak = {}
    without_mod_relative_peak_result = {}
    for key in without_mod_fine_relative_peak_dict.keys():
        without_mod_relative_peak[str(key)] = []
        without_mod_relative_peak_result[str(key)] = {}
        without_mod_relative_peak_result[str(key)]['mean'] = 0
        without_mod_relative_peak_result[str(key)]['median'] = 0
    for key in without_mod_fine_relative_peak_dict.keys():
        for item in without_mod_fine_relative_peak_dict[key]:
            without_mod_relative_peak[str(key)].append(round(item[1], 6))
        if ion_relative_mode != 4 and ion_relative_mode != 5:
            without_mod_relative_peak_result[str(
                key)]['mean'] += np.sum(without_mod_relative_peak[str(key)])/len(without_mod_res)
        # without_mod_relative_peak_result[str(key)]['mean'] += np.mean(without_mod_relative_peak[str(key)])
        elif ion_relative_mode == 4:
            without_mod_relative_peak_result[str(key)]['mean'] += np.median([0]*(len(without_mod_res)-len(
                without_mod_relative_peak[str(key)]))+without_mod_relative_peak[str(key)])
        else:
            without_mod_relative_peak_result[str(
                key)]['mean'] += np.sum(without_mod_relative_peak[str(key)])/len(without_mod_res)
    without_mod_tuple = []
    for index, i in enumerate(without_mod_filtered_position_counter_list):
        without_mod_tuple.append(
            (i[0], i[1], without_mod_relative_peak_result[str(i[0])]['mean']))
        # without_mod_tuple.append((i[0], i[1], without_mod_relative_peak_result[str(i[0])]['median']))
    # without_mod_tuple.sort(key=lambda word:word[2], reverse=True)
    without_mod_tuple.sort(key=lambda word: word[1], reverse=True)
    without_mod_tuple.sort(key=lambda word: word[2], reverse=True)
    with open(without_mod_ion_result_path, 'w', encoding='utf-8') as f:
        f.write("Top 300 characteristic ions in the unmodified spectra: (PSM: {})\n".format(
            len(without_mod_res)))
        f.write("Rank" + '\t' + "ion type" + '\t' + 'ion count' +
                '\t' + 'ion accuracy' + '\t' + 'ion relative peak' + '\n')
        for index, i in enumerate(without_mod_tuple):
            f.write(str(index+1) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + str(
                round(without_mod_peak_dict[i[0]], 6)) + '\t' + str(round(i[2], 6)) + '\n')
        f.write('\n')
    return ion_area_dict, n_bins_dict, without_mod_tuple, without_mod_have_close_ion_scan_dict



def close_ion_learning(pchem_output_path, current_path, ion_type, modification_list, modification_dict, blind_res, mass_spectra_dict, modification_PSM, modification_site, pchem_summary_path, ion_relative_mode, ion_rank_threshold, ion_filter_mode, ion_filter_ratio):
    without_mod_ion_area_dict, without_mod_n_bins_dict, without_mod_tuple, without_mod_have_close_ion_scan_dict = deal_without_mod_psm(pchem_output_path, current_path, ion_type, modification_list, modification_dict, blind_res,
                                                                                                                                       mass_spectra_dict, modification_PSM, modification_site, pchem_summary_path, ion_relative_mode, ion_rank_threshold, ion_filter_mode)

    mod_ion_result_path = os.path.join(
        pchem_output_path, 'pChem_mod_ion_result.summary')
    ion_dict, mod_have_close_ion_scan_dict = ion_type_determine(current_path, modification_list, modification_dict,
                                                                 mass_spectra_dict, blind_res, ion_type, ion_relative_mode, modification_site, ion_filter_mode, pchem_output_path)

    # if int(ion_filter_mode) == 1:
    #     pchem_summary_path1, mod_filter = modification_filter_mode1(
    #         mod_ion_result_path, ion_dict, modification_list, ion_filter_mode, modification_PSM, ion_type)
    # elif int(ion_filter_mode) == 2:
    #     pchem_summary_path1, mod_filter = modification_filter_mode2(
    #         without_mod_n_bins_dict, without_mod_ion_area_dict, mod_ion_result_path, ion_dict, modification_list, ion_filter_mode, without_mod_tuple, ion_type, modification_PSM, pchem_output_path, pchem_summary_path, ion_filter_ratio)
    pchem_summary_path1, mod_filter = modification_filter_mode(
            without_mod_n_bins_dict, without_mod_ion_area_dict, mod_ion_result_path, ion_dict, modification_list, ion_filter_mode, without_mod_tuple, ion_type, modification_PSM, pchem_output_path, pchem_summary_path, ion_filter_ratio)
    
    # return pchem_summary_path1, ion_dict, mod_filter
    return pchem_summary_path1


def modification_filter_mode(without_mod_n_bins_dict, without_mod_ion_area_dict, mod_ion_result_path, ion_dict, modification_list, ion_filter_mode, without_mod_tuple, close_ion, modification_PSM, pchem_output_path, pchem_summary_path, ion_filter_ratio):
    if round(float(close_ion), 3) in without_mod_ion_area_dict.keys():
        without_mod_close_ion_rank = sorted([without_mod_ion_area_dict[i] for i in without_mod_ion_area_dict.keys()]).index(
            without_mod_ion_area_dict[round(float(close_ion), 3)])
    else:
        without_mod_close_ion_rank = -1

    mod_filter = write_mod_ion_file(
        mod_ion_result_path, modification_list, ion_dict, 20, modification_PSM, close_ion)

    result = []
    for mod in mod_filter:
        # score = (1- 2/np.pi *np.arccos((1-cosine(without_mod_n_bins_dict[126.128]['distribute'], ion_dict[mod]['n_bins_dict']
        #    [126.128]['distribute'])))) * np.log10(len(ion_dict[mod]['n_bins_dict'][126.128]['peaks']))
        if float(close_ion) in ion_dict[mod]['n_bins_dict'].keys():
            if not float(close_ion) in without_mod_n_bins_dict.keys():
                w_tmp = np.zeros_like(
                    ion_dict[mod]['n_bins_dict'][float(close_ion)]['distribute'])
                w_tmp[0] = 1.0
                score = cosine(w_tmp, ion_dict[mod]['n_bins_dict']
                               [float(close_ion)]['distribute']) * np.log10(len(ion_dict[mod]['n_bins_dict'][float(close_ion)]['peaks']))
            else:
                score = cosine(without_mod_n_bins_dict[float(close_ion)]['distribute'], ion_dict[mod]['n_bins_dict']
                               [float(close_ion)]['distribute']) * np.log10(len(ion_dict[mod]['n_bins_dict'][float(close_ion)]['peaks']))

            result.append((mod, score))

    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    # print(sorted_result)
    # draw_summary_pdf(sorted_result)
    modification_score_path = os.path.join(
        pchem_output_path, 'pChem_modification_score.txt')

    mod_filter = []
    if len(sorted_result) > 0:
        max_score = sorted_result[0][1]
        threshold = ion_filter_ratio
        score_threshold = max_score * threshold
        print(sorted_result)
        with open(modification_score_path, 'w', encoding='utf-8') as f_m:
            for sr in sorted_result:
                f_m.write(str(sr[0]) + "\t" + str(sr[1]) + "\n")
        mod_filter = []
        mod_num = 0
        for i in sorted_result:
            if i[1] > score_threshold:
                mod_filter.append(i[0])
                mod_num += 1
            else:
                break
    print(mod_filter)
    if without_mod_close_ion_rank == 0:
        # mod_filter = []
        warnings.warn(
            "The concentration of eigenions in the unmodified spectra is very high, it will be difficult to distinguish probe modifications from other modifications!")
    mod_final_filter, pchem_summary_path1 = filter_write_mod(
        mod_filter, pchem_summary_path, ion_dict, ion_filter_mode, pchem_output_path)
    return pchem_summary_path1, mod_filter


def write_mod_ion_file(mod_ion_result_path, modification_list, ion_dict, ion_rank_threshold, modification_PSM, ion_type):
    mod_filter = []
    with open(mod_ion_result_path, 'w', encoding='utf-8') as f1:
        for i in tqdm(ion_dict.keys()):
            f1.write("Modification info for " + i + ":(PSM = {}) ".format(
                modification_PSM[i]) + "(filtered PSM = {})".format(ion_dict[i]['mod_psm']) + '(neighbor filtered PSM = {})'.format(len([x for x in ion_dict[i]['reserve_dict'].keys() if ion_dict[i]['reserve_dict'][x] == True])) + "\n")
            # f1.write("Modification info for " + i  + ":(filtered PSM = {})".format(ion_dict[i]['mod_psm'])+ "\n")
            f1.write("Top 300 characteristic ions in the " +
                     i + " spectra:" + "\n")
            f1.write("Rank" + '\t' + "ion type" + '\t' + 'ion count' +
                     '\t' + 'ion accuracy' + '\t' + 'ion relative peak' + '\n')
            mod_relative_peak = {}
            mod_relative_peak_result = {}
            for key in ion_dict[i]['t_fine_relative_peak_dict'].keys():
                mod_relative_peak[str(key)] = []
                mod_relative_peak_result[str(key)] = {}
                mod_relative_peak_result[str(key)]['mean'] = 0
                mod_relative_peak_result[str(key)]['median'] = 0
            for key in ion_dict[i]['t_fine_relative_peak_dict'].keys():
                for item in ion_dict[i]['t_fine_relative_peak_dict'][key]:
                    mod_relative_peak[str(key)].append(round(item[1], 6))

                mod_relative_peak_result[str(
                    key)]['mean'] += np.sum(mod_relative_peak[str(key)])/ion_dict[i]['mod_psm']

            mod_tuple = []
            for index, j in enumerate(ion_dict[i]['t_position_counter_list']):
                mod_tuple.append((j[0], j[1], mod_relative_peak_result[str(
                    j[0])]['mean'], ion_dict[i]['t_peak_dict'][j[0]]))
            # mod_tuple.sort(key=lambda word:word[2], reverse=True)
            mod_tuple.sort(key=lambda word: word[1], reverse=True)
            mod_tuple.sort(key=lambda word: word[2], reverse=True)
            for index, j in enumerate(mod_tuple):
                f1.write(str(index+1) + '\t' + str(j[0]) + '\t' + str(j[1]) + '\t' + str(
                    round(ion_dict[i]['t_peak_dict'][j[0]], 6)) + '\t' + str(round(j[2], 6)) + '\n')
            f1.write("\n")
            ion_common_dict = {}
            ion_common_acc_list = [ion_common_dict[i]
                                   for i in ion_common_dict.keys()]
            save_flag = False

            for k in range(0, 300):
                if k >= len(mod_tuple):
                    break
                if abs((mod_tuple[k][3] - float(ion_type)))/float(ion_type)*1000000 <= 20:
                    if k < ion_rank_threshold:
                        save_flag = True
                        break
            if save_flag:
                mod_filter.append(i)
    return mod_filter

def filter_write_mod(mod_filter, pchem_summary_path, ion_dict, ion_filter_mode, pchem_output_path):
    with open(pchem_summary_path, 'r') as f2:
        lines = f2.readlines()
    line_0 = lines[0]
    # line_0 = line_0[:-1] + '\t' + 'Remark PSM\n'
    result = lines[1:]
    line_re = []
    # line_re.append(line_0)
    for mod_rank in mod_filter:
        for line in result:
            if mod_rank == line.split("\t")[1]:
                line_re.append(line)
                break

    line_re.sort(key=lambda line: int(
        line.split("\t")[5]), reverse=True)
    line_re.insert(0, line_0)
    pchem_summary_path1 = os.path.join(
        pchem_output_path, 'pChem_ion_filter.summary')
    mod_final_filter = []
    with open(pchem_summary_path1, 'w', encoding='utf-8') as f3:
        id = 0
        for index, i in enumerate(line_re):
            if index == 0:
                t = i
                f3.write(t)
            else:
                mod_name = i.split("\t")[1]
                # if len([x for x in ion_dict[mod_name]['reserve_dict'].keys()
                #         if ion_dict[mod_name]['reserve_dict'][x] == True]) > 0:
                id += 1
                line_res = i.split("\t")
                line_res[0] = str(id)
                t = "\t".join(line_res)
                # t = t[:-1] + "\t"
                # t += str(len([x for x in ion_dict[mod_name]['reserve_dict'].keys()
                #               if ion_dict[mod_name]['reserve_dict'][x] == True])) + "\n"
                f3.write(t)
                mod_final_filter.append(mod_name)

        f3.write('\n')


    return mod_final_filter, pchem_summary_path1

def heatmap_ion(pchem_output_path, pchem_summary_path2):
    with open(pchem_summary_path2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    re_lines = lines[1:]
    y_stick = ["N-SIDE", "C-SIDE", "A", "C", "D", "E", "F", "G", "H",
               "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    y_stick_dict = {}
    for index, i in enumerate(y_stick):
        y_stick_dict[i] = index
    x_stick = []
    mod_map = {}
    for line in re_lines:
        if line != "\n":
            mod_name = line.split("\t")[1][12:]
            pos_list = []
            pos_list.append((line.split("\t")[3].split(
                "|")[0], float(line.split("\t")[3].split("|")[1])))
            other_mods = line.split("\t")[4]
            other_mods_list = other_mods.split(";  ")[:-1]
            if len(other_mods_list) != 0:
                for t_mod in other_mods_list:
                    pos_list.append((t_mod.split("(")[0], float(
                        t_mod.split("(")[1].split(",")[0])))
            mod_map[mod_name] = [0.0]*len(y_stick)
            for t in pos_list:
                mod_map[mod_name][y_stick_dict[t[0]]] = t[1]
            x_stick.append(mod_name)
    pd_mod_map = pd.DataFrame(mod_map, index=y_stick, columns=x_stick)
    ax = sns.heatmap(pd_mod_map, vmin=0.0, vmax=1.0, cmap='YlGnBu')
    plt.ylabel('amino acid selectivity')
    plt.xlabel('modifications')
    png_path = os.path.join(pchem_output_path, 'heat_map.pdf')
    # plt.show()
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    return True

