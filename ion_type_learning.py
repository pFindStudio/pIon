# 中性丢失鉴定

from asyncio.windows_events import NULL
from utils import parameter_file_read
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from parameter import element_dict, amino_acid_dict, ion_common_dict, common_dict_create
from tqdm import tqdm
import time
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
import json
import glob
from bisect import bisect_left

h2o_mass = element_dict["H"]*2 + element_dict["O"]*1
proton_mass = element_dict['Pm']
# ion_common = ion_common_dict

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
            relative_peak_list.append((item[0], float(item[1])/max_peak))
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
    # print('reading mgf data.')
    blind_res_title_list = []
    for item in blind_res:
        # 底层是哈希表
        blind_res_title_list.append(item.split("\t")[0])
    # tempLst = list(range(len(blind_res_title_list)))
    dct = dict(zip(blind_res_title_list, blind_res_title_list))
    i = 0
    while i < len(lines):
        if 'BEGIN' in lines[i]:
            i += 1
            spectrum_name = lines[i].split('=')[1].strip()
            flag = False
            if spectrum_name in dct:
                flag = True
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
            if flag:
                spectrum = MassSpectrum_for_ion(
                    spectrum_charge, spectrum_pepmass, spectrum_peak_list)
                mass_spectra_dict[spectrum_name] = spectrum
        i += 1
    print('The number of spectra: ', len(mass_spectra_dict.keys()))
    return mass_spectra_dict


def mgf_read_for_ion_pfind_filter1(mgf_path, blind_res, blind_res_ms1_dict):
    mass_spectra_dict = {}
    with open(mgf_path, 'r') as f:
        lines = f.readlines()
    start_index_list = []
    # end_index_list = []
    for index, line in enumerate(lines):
        if "BEGIN IONS" in line:
            start_index_list.append(
                (index, lines[index+1].split("=")[-1][:-1]))
    # for index, line in enumerate(lines):
    #     if "END IONS" in line:
    #         end_index_list.append(index)
    # print(index_list)

    # print('reading mgf data.')
    blind_res_title_list = []
    for item in blind_res:
        # 底层是哈希表
        blind_res_title_list.append(item.split("\t")[0])
    dct = dict(zip(blind_res_title_list, blind_res_title_list))

    result_index = []
    # result_end_index = []
    # end_index = end_index_l[start_index_l.index(start_index)]
    for i in start_index_list:
        if i[1] in dct:
            result_index.append(i[0])
            # result_end_index.append(end_index_list[start_index_list.index(i)])
    # print(result_index)

    for index, i in enumerate(result_index):
        spectrum_name = lines[i+1].split('=')[1].strip()
        spectrum_charge = int(lines[i+2].split('=')[1][0])
        spectrum_pepmass = float(lines[i+4].split('=')[1])
        spectrum_peak_list = []
        spectrum_ms1_peak = float(
            blind_res_ms1_dict[spectrum_name].split(" ")[-1][:-1])
        j = i+5
        while j < len(lines):
            if 'END' in lines[j]:
                break
            else:
                spectrum_peak_list.append(lines[j].split())
            j += 1
        # ms1 = ms2_ms1_scan_dict[int(spectrum_name.split(".")[2])]
        spectrum = MassSpectrum_for_ion(
            spectrum_charge, spectrum_pepmass, spectrum_peak_list, spectrum_ms1_peak)
        mass_spectra_dict[spectrum_name] = spectrum
    print('The number of spectra: ', len(mass_spectra_dict.keys()))
    return mass_spectra_dict

# def mgf_read1(mgf_path):
#     tmp_block = []
#     mass_spectra_dict = {}
#     for line in open(mgf_path):
#         if 'BEGIN' in line:
#             tmp_block = []
#             continue
#         elif 'END' not in line:
#             tmp_block.append(line)
#             continue
#         else:
#             spectrum_name = tmp_block[0].split('=')[1].strip()
#             spectrum_charge = int(tmp_block[1].split('=')[1][0])
#             spectrum_pepmass = float(tmp_block[3].split('=')[1])
#             spectrum_peak_list = []
#             i = 4
#             while i < len(tmp_block):
#                 spectrum_peak_list.append(tmp_block[i].split())
#                 i += 1
#             spectrum = MassSpectrum(spectrum_charge, spectrum_pepmass, spectrum_peak_list)
#             mass_spectra_dict[spectrum_name] = spectrum
#     print('The number of spectra: ', len(mass_spectra_dict.keys()))
#     return mass_spectra_dict


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


def PSM_filter1(blind_res, modification):
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
    '''
    # 生成原始的质量数组
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])
    # print(mass_vector)

    pos_mod_list = mod_list.split(';')[:-1] 
    for pos_mod in pos_mod_list: 
        pos, mod_name = pos_mod.split(',')
        if modification in mod_name:
            mod_pos = int(pos)
            mass_vector = mass_vector_modify(mod_pos, accurate_mass, mass_vector)
        else:
            mass_vector = mass_vector_modify(int(pos), common_modification_dict[mod_name], mass_vector)
    '''
    for amino_acid in peptide_sequence:
        mass_vector.append(amino_acid_dict[amino_acid])

    return mass_vector


def mass_vector_generation_for_mod_ion(peptide_sequence, mod_list, modification, common_modification_dict, modification_dict):
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
                mod_pos, modification_dict[modification], mass_vector)
        else:
            mass_vector = mass_vector_modify(
                int(pos), common_modification_dict[mod_name], mass_vector)
    return mass_vector

# 生成b,y离子谱峰的数组


def b_y_vector_generation_for_no_mod_ion(mass_vector, charge):
    mod_peak_list = []
    sequence_len = len(mass_vector)
    mass_sum_vector = [mass_vector[0]]

    for i in range(1, sequence_len):
        mass_sum_vector.append(mass_vector[i] + mass_sum_vector[i-1])
    # print(mass_vector)
    # print(mass_sum_vector)

    # 生成b离子谱峰
    mod_peak_list = [mass + proton_mass for mass in mass_sum_vector[:]]
    # 生成y离子谱峰
    mod_peak_list += [mass_sum_vector[sequence_len-1] - mass +
                      h2o_mass + proton_mass for mass in mass_sum_vector[:-1]]
    if charge > 1:
        for i in range(2, charge+1):
            mod_peak_list += [(mass + i*proton_mass) /
                              i for mass in mass_sum_vector[:]]
            mod_peak_list += [(mass_sum_vector[sequence_len-1] - mass +
                               h2o_mass + i*proton_mass)/i for mass in mass_sum_vector[:-1]]

    return mod_peak_list


def delete_peak(peak_list, delete_peak_list):
    filter_peak_list = []
    for i in peak_list:
        for j in delete_peak_list:
            if abs(float(i[0])-j)/(j+0.000001)*1000000 <= 20:
                # if abs(float(i[0]) - j) <= 0.02:
                filter_peak_list.append(i)

    for i in filter_peak_list:
        if i in peak_list:
            peak_list.remove(i)
    return peak_list


# 返回指定谱图系列filtered_res的特征离子, pyp
def feature_peak_determine_for_ion(mass_spectra_dict, filtered_res, common_modification_dict, mod_flag, modification_dict, modification, close_ion, ion_relative_mode):
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
    ion_common_list_all = []
    # ion_common_dict.update({str(close_ion): float(close_ion)})
    for line in tqdm(filtered_res):
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        if spectrum_name in mass_spectra_dict.keys():
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
                ion_common_list = []
                for pk in peak_list:
                    for ic in ion_common_dict.keys():
                        if len(ion_common_list) < len(ion_common_dict.keys()):
                            if abs(ion_common_dict[ic] - float(pk[0]))/ion_common_dict[ic]*1000000 <= 20:
                                ion_common_list.append(pk)
                        else:
                            break
                ion_common_list_all.append(ion_common_list)
                ion_common_sum = np.sum([float(i[1]) for i in ion_common_list])
                relative_peak_list = []
                if ion_relative_mode == 1:
                    for item in peak_list:
                        relative_peak_list.append(
                            (item[0], float(item[1])/max_peak))

                elif ion_relative_mode == 2:
                    for item in peak_list:
                        if ion_common_sum + float(item[1]) != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum + max_peak)))
                        else:
                            relative_peak_list.append((item[0], 0.0))
                elif ion_relative_mode == 3:
                    for item in peak_list:
                        if ion_common_sum + float(item[1]) != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum + float(item[1]))))
                        else:
                            relative_peak_list.append((item[0], 0.0))

                elif ion_relative_mode == 4:
                    for item in peak_list:
                        if ion_common_sum != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum)))
                        else:
                            continue

                elif ion_relative_mode == 5:
                    for item in peak_list:
                        if ion_common_sum != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum)))
                        else:
                            continue
                elif ion_relative_mode == 6:
                    for item in peak_list:
                        relative_peak_list.append(
                            (item[0], float(item[1])/mass_spectra_dict[spectrum_name].ms1_peak))

                # elif ion_relative_mode == 4:
                #     # 选取范围0-300的peaks总和当背景
                #     t_peak = []
                #     for p in peak_list:
                #         if float(p[0]) <= 300:
                #             t_peak.append(p)
                #         if float(p[0]) > 300:
                #             break

            else:
                peak_list = mass_spectra_dict[spectrum_name].peak_list
                max_peak = max([float(i[1]) for i in peak_list])
                mass_vector = mass_vector_generation_for_mod_ion(
                    peptide_sequence, mod_list, modification, common_modification_dict, modification_dict)
                delete_peak_list = b_y_vector_generation_for_no_mod_ion(
                    mass_vector, mass_spectra_dict[spectrum_name].charge)
                peak_list = delete_peak(peak_list, delete_peak_list)
                ion_common_list = []
                for pk in peak_list:
                    for ic in ion_common_dict.keys():
                        if len(ion_common_list) < len(ion_common_dict.keys()):
                            if abs(ion_common_dict[ic] - float(pk[0]))/ion_common_dict[ic]*1000000 <= 20:
                                ion_common_list.append(pk)
                        else:
                            break
                ion_common_list_all.append(ion_common_list)
                ion_common_sum = np.sum([float(i[1]) for i in ion_common_list])
                relative_peak_list = []
                # for item in peak_list:
                #     if ion_common_sum + float(item[1]) != 0:
                #         relative_peak_list.append((item[0], float(item[1])/(ion_common_sum + max_peak)))
                #         # relative_peak_list.append((item[0], float(item[1])/(ion_common_sum + float(item[1]))))
                #     else:
                #         relative_peak_list.append((item[0], 0.0))
                # max_peak = max([float(i[1]) for i in peak_list])
                if ion_relative_mode == 1:
                    for item in peak_list:
                        relative_peak_list.append(
                            (item[0], float(item[1])/max_peak))

                elif ion_relative_mode == 2:
                    for item in peak_list:
                        if ion_common_sum + float(item[1]) != 0:
                            # relative_peak_list.append((item[0], float(item[1])/(ion_common_sum + float(item[1]))))
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum + max_peak)))
                        else:
                            relative_peak_list.append((item[0], 0.0))
                elif ion_relative_mode == 3:
                    for item in peak_list:
                        if ion_common_sum + float(item[1]) != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum + float(item[1]))))
                            # relative_peak_list.append((item[0], float(item[1])/(ion_common_sum + max_peak)))
                        else:
                            relative_peak_list.append((item[0], 0.0))
                elif ion_relative_mode == 4:
                    for item in peak_list:
                        if ion_common_sum != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum)))
                        else:
                            continue
                elif ion_relative_mode == 5:
                    for item in peak_list:
                        if ion_common_sum != 0:
                            relative_peak_list.append(
                                (item[0], float(item[1])/(ion_common_sum)))
                        else:
                            continue
                elif ion_relative_mode == 6:
                    for item in peak_list:
                        relative_peak_list.append(
                            (item[0], float(item[1])/mass_spectra_dict[spectrum_name].ms1_peak))

        else:
            continue
        fine_position_list += [float(p[0]) for p in peak_list]
        fine_position_peak_list += [(float(p[0]), float(p[1]))
                                    for p in peak_list]
        fine_position_relative_peak_list += [(float(p[0]), float(p[1]))
                                             for p in relative_peak_list]
        # 是否应当按照±10的精度来看待特征离子
        coarse_position_list += [round(float(p[0]), 2) for p in peak_list]
        coarse_position_peak_list += [(round(float(p[0]), 2), float(p[1]))
                                      for p in peak_list]
        coarse_position_relative_peak_list += [
            (round(float(p[0]), 2), float(p[1])) for p in relative_peak_list]
    position_counter = Counter(coarse_position_list)
    filtered_position_list = [p[0]
                              for p in position_counter.most_common()[:300]]
    filtered_position_counter_list = [
        (p[0], p[1]) for p in position_counter.most_common()[:300]]
    for position in filtered_position_list:
        peak_dict[position] = []
        fine_peak_dict[position] = []
        fine_relative_peak_dict[position] = []
    for position in fine_position_peak_list:
        peak = round(position[0], 2)
        if peak in peak_dict.keys():
            peak_dict[peak].append(position[0])
            fine_peak_dict[peak].append((position[0], position[1]))
    for position in fine_position_relative_peak_list:
        peak = round(position[0], 2)
        if peak in peak_dict.keys():
            fine_relative_peak_dict[peak].append((position[0], position[1]))
    peak_dict1 = {}
    for key in peak_dict.keys():
        peak_dict1[key] = np.mean(peak_dict[key])

    # if round(float(close_ion), 2) in fine_relative_peak_dict.keys():
    #     tmp = [round(i[1], 3) for i in fine_relative_peak_dict[126.13]]
    #     t_c = Counter(tmp)
    #     t_d = dict(t_c)
    #     result_sort1 = sorted(t_d.items(), key = lambda x:x[0])
    #     x1 = []
    #     y1 = []
    #     for x_y in result_sort1:
    #         print(x_y)
    #         x1.append(x_y[0])
    #         y1.append(x_y[1])
    #     width = 1
    #     indexes = np.arange(len(y1))
    #     # plt.bar(indexes, y1, width)

    #     plt.plot(x1, y1, 'c*-')
    #     # plt.xticks(indexes + width * 3, y1)
    #     plt.xlabel('intensity')
    #     plt.ylabel('count')
    #     # plt.title(mod + "_" + str(len(mod_char_ion_dict[mod])))
    #     # plt.show()
    #     if os.path.exists('./ion_intensity_img') == False:
    #         os.makedirs('./ion_intensity_img')
    #     if mod_flag == True:
    #         plt.savefig('./ion_intensity_img/{}_{}.png'.format(modification, close_ion, bbox_inches='tight'))
    #     else:
    #         plt.savefig('./ion_intensity_img/{}_{}.png'.format("without_mod", close_ion, bbox_inches='tight'))
    #     plt.close()

    # if round(float(close_ion), 2) in peak_dict.keys():
    #     tmp = [i for i in peak_dict[126.13]]
    #     t_c = Counter(tmp)
    #     t_d = dict(t_c)
    #     result_sort1 = sorted(t_d.items(), key = lambda x:x[0])
    #     x1 = []
    #     y1 = []
    #     for x_y in result_sort1:
    #         print(x_y)
    #         x1.append(x_y[0])
    #         y1.append(x_y[1])
    #     width = 1
    #     indexes = np.arange(len(y1))
    #     # plt.bar(indexes, y1, width)

    #     plt.plot(x1, y1, marker="o", markersize=2, color="b")
    #     plt.xlim((round(float(close_ion), 6)-0.004, round(float(close_ion), 6)+0.004))
    #     # plt.plot(x1, y1)
    #     # plt.xticks(indexes + width * 3, y1)
    #     plt.xlabel('mass')
    #     plt.ylabel('count')
    #     if mod_flag == True:
    #         info = str(modification) + " | " + "mean: " + str(round(np.mean(tmp), 6)) + "  " + "std: " + str(round(np.std(tmp), 6)) + "  " + "sum: " + str(len(tmp))
    #         plt.title(info)
    #     else:
    #         info = "without_mod | " + "mean: " + str(round(np.mean(tmp), 6)) + "  " + "std: " + str(round(np.std(tmp), 6)) + "  " + "sum: " + str(len(tmp))
    #         plt.title(info)

    #     if os.path.exists('./ion_mass_img') == False:
    #         os.makedirs('./ion_mass_img')
    #     if mod_flag == True:
    #         plt.savefig('./ion_mass_img/{}_{}.png'.format(modification, close_ion, bbox_inches='tight'))
    #         file_tmp = open('./ion_mass_img/{}.txt'.format(modification),'w')
    #         file_tmp.write(str(x1) + "\n")
    #         file_tmp.write(str(y1) + "\n")
    #         file_tmp.close()
    #         file_ion_mass= open('ion_mass.summary','a+')
    #         file_ion_mass.write(info + "\n")
    #         file_ion_mass.close()
    #     else:
    #         plt.savefig('./ion_mass_img/{}_{}.png'.format("without_mod", close_ion, bbox_inches='tight'))
    #         file_tmp = open('./ion_mass_img/{}.txt'.format("without_mod"),'w')
    #         file_tmp.write(str(x1) + "\n")
    #         file_tmp.write(str(y1) + "\n")
    #         file_tmp.close()
    #         file_ion_mass= open('ion_mass.summary','a+')
    #         file_ion_mass.write(info + "\n")
    #         file_ion_mass.close()
    #     plt.close()
    return filtered_position_list, filtered_position_counter_list, peak_dict1, fine_peak_dict, fine_relative_peak_dict


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


# 离子类型学习
def ion_type_determine(current_path, modification_list, modification_dict, parameter_dict):

    pchem_cfg_path = os.path.join(current_path, 'pChem.cfg')
    # parameter_dict = parameter_file_read(pchem_cfg_path)
    print(parameter_dict)

    # 质谱数据读取
    mass_spectra_dict = {}
    for msms_path in parameter_dict['msms_path']:
        # pp_path = os.path.join(parameter_dict['output_path'], "source/pParse")
        # mgf_path = glob.glob(os.path.join(pp_path, "*.mgf"))
        # mgf_path = mgf_path[0]
        mgf_path = msms_path.split('=')[1].split('.')[0] + '.mgf'
        cur_mass_spectra_dict = mgf_read(mgf_path)
        mass_spectra_dict.update(cur_mass_spectra_dict)

    # mass_spectra_dict = read_mgf(parameter_dict)
    # print('The number of spectra: ', len(mass_spectra_dict.keys()))
    # feature_peak_determine(mass_spectra_dict)

    # 读取盲搜得到的结果
    blind_path = os.path.join(parameter_dict['output_path'], 'blind')
    blind_res_path = os.path.join(blind_path, 'pFind-Filtered.spectra')
    blind_res = blind_res_read(blind_res_path)

    # 读取常见修饰的列表
    common_modification_dict = common_dict_create(current_path)
    # print(common_modification_dict)
    position_list = []
    peak_dict = []

    # 修饰->中性丢失
    mod2ion = {}
    exist_ion_flag = True
    ion_list = []

    # 筛选有效的PSM
    for modification in modification_list:
        # pfind-filtered line
        print('calculate the ion of ', modification)
        mod = modification.split('_')[2]
        int_mod = int(mod)
        mod2ion[mod] = []
        t_ion_list = []
        filtered_res = PSM_filter(blind_res, modification)

        # 确定报告每个未知修饰的离子，modification in modification_list
        # t_position_list：前300个候选特征离子；
        # t_peak_dict：每个特征离子的精确质量
        t_position_list, t_peak_dict = feature_peak_determine(
            mass_spectra_dict, filtered_res)
        peak_dict.append(t_peak_dict)
        position_list.append(t_position_list)

        # total_ion_diff_list: 所有质量差组成的列表
        total_ion_diff_counter, total_ion_diff_list, total_weight_ion_diff_list = ion_type_compute(
            filtered_res, modification, modification_dict[modification], common_modification_dict, mass_spectra_dict)
        # 画频率图
        # freq_point_plot(total_ion_diff_counter, modification)
        # print(total_ion_diff_counter)
        # freq_line_plot(total_ion_diff_counter)
        # freq_analysis(total_ion_diff_counter)

        # 判断是否存在中性丢失
        if int(total_ion_diff_counter.most_common()[0][0]) == 0:
            exist_ion_flag = False

        repeat_list = []
        for ion_mass, _ in total_ion_diff_counter.most_common()[:10]:

            # 是否引入信噪比过滤
            if signal_noise_filter(ion_mass, total_ion_diff_counter) == False:
                continue

            accurate_ion_mass = accurate_ion_mass_computation(
                ion_mass, total_ion_diff_list)
            # weight_accurate_ion_mass = weight_accurate_ion_mass_computation(ion_mass, total_weight_ion_diff_list)
            # print('average: ', accurate_ion_mass)
            # print('weight: ', weight_accurate_ion_mass)
            int_ion_mass = int(accurate_ion_mass)
            if int_ion_mass not in repeat_list and int_mod != int_ion_mass:
                mod2ion[mod].append(accurate_ion_mass)
                t_ion_list.append(accurate_ion_mass)
                repeat_list.append(int_ion_mass)
        ion_list.append(t_ion_list)
        # print(filtered_res)
    # 特征离子发现
    # print('Feature ion results')
    # pair_list = feature_pair_find(position_list, peak_dict, parameter_dict['mass_of_diff_diff'])
    return mod2ion, ion_list, exist_ion_flag


def ion_type_determine1(current_path, modification_list, modification_dict, mass_spectra_dict, blind_res, close_ion, ion_relative_mode):
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
    # 筛选有效的PSM
    for modification in modification_list:
        # pfind-filtered line
        print('calculate the ion of ', modification)
        mod = modification.split('_')[2]
        int_mod = int(mod)
        mod2ion[mod] = []
        t_ion_list = []
        filtered_res = PSM_filter1(blind_res, modification)

        # 确定报告每个未知修饰的离子，modification in modification_list
        # t_position_list：前300个候选特征离子；
        # t_peak_dict：前300个特征离子每个特征离子的精确质量
        ion_dict[modification] = {}
        t_position_list, t_position_counter_list, t_peak_dict, t_fine_peak_dict, t_fine_relative_peak_dict = feature_peak_determine_for_ion(
            mass_spectra_dict, filtered_res, common_modification_dict, True, modification_dict, modification, close_ion, ion_relative_mode)
        ion_dict[modification]['t_position_list'] = t_position_list
        ion_dict[modification]['t_position_counter_list'] = t_position_counter_list
        ion_dict[modification]['t_peak_dict'] = t_peak_dict
        ion_dict[modification]['t_fine_peak_dict'] = t_fine_peak_dict
        ion_dict[modification]['t_fine_relative_peak_dict'] = t_fine_relative_peak_dict
        '''
        peak_dict.append(t_peak_dict)
        position_list.append(t_position_list) 

        
        # total_ion_diff_list: 所有质量差组成的列表 
        total_ion_diff_counter, total_ion_diff_list, total_weight_ion_diff_list = ion_type_compute(filtered_res, modification, modification_dict[modification], common_modification_dict, mass_spectra_dict) 
        # 画频率图 
        # freq_point_plot(total_ion_diff_counter, modification)
        # print(total_ion_diff_counter)
        # freq_line_plot(total_ion_diff_counter) 
        # freq_analysis(total_ion_diff_counter)

        # 判断是否存在中性丢失 
        if int(total_ion_diff_counter.most_common()[0][0]) == 0:
            exist_ion_flag = False 

        repeat_list = []
        for ion_mass, _ in total_ion_diff_counter.most_common()[:10]: 

            # 是否引入信噪比过滤
            if signal_noise_filter(ion_mass, total_ion_diff_counter) == False: 
                continue 
            
            accurate_ion_mass = accurate_ion_mass_computation(ion_mass, total_ion_diff_list) 
            # weight_accurate_ion_mass = weight_accurate_ion_mass_computation(ion_mass, total_weight_ion_diff_list)
            # print('average: ', accurate_ion_mass)
            # print('weight: ', weight_accurate_ion_mass) 
            int_ion_mass = int(accurate_ion_mass) 
            if int_ion_mass not in repeat_list and int_mod != int_ion_mass: 
                mod2ion[mod].append(accurate_ion_mass) 
                t_ion_list.append(accurate_ion_mass)
                repeat_list.append(int_ion_mass)
        ion_list.append(t_ion_list)

    return mod2ion, ion_list, exist_ion_flag  
    '''

    return ion_dict


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


# 利用轻重标记筛选中性丢失
def ion_filter(ion_list, mass_diff):
    refine_ion_list = [[], []]
    for light_mass in ion_list[0]:
        for heavy_mass in ion_list[1]:
            diff = int(heavy_mass - light_mass)
            if diff == mass_diff:
                refine_ion_list[0].append(round(light_mass, 6))
                refine_ion_list[1].append(round(heavy_mass, 6))
            if len(refine_ion_list[0]) >= 3:
                return refine_ion_list
    return refine_ion_list

# 筛选出没有修饰的PSM


def without_mod_PSM_filter(current_path, blind_res, mass_spectra_dict, modification_dict, close_ion, ion_relative_mode):
    without_mod_res = []
    start_time = time.time()
    for line in blind_res:
        line_split = line.split('\t')
        spectrum_name, peptide_sequence, mod_list = line_split[0], line_split[5], line_split[10]
        if mod_list == "":
            if spectrum_name in mass_spectra_dict.keys():
                without_mod_res.append(line)
    common_modification_dict = common_dict_create(current_path)
    without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict = feature_peak_determine_for_ion(
        mass_spectra_dict, without_mod_res, common_modification_dict, False, modification_dict, NULL, close_ion, ion_relative_mode)
    end_time = time.time()
    print('filter without-mod PSM cost time (s): ',
          round(end_time - start_time, 1))
    return without_mod_res, without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict


# 比较没有修饰的PSM和有修饰的PSM之间特征离子的差别
def feature_ion_compare(current_path, modification_list, modification_dict, blind_res, mass_spectra_dict, close_ion, ion_relative_mode):
    without_mod_res, without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict = without_mod_PSM_filter(
        current_path, blind_res, mass_spectra_dict, modification_dict, close_ion)
    ion_dict = ion_type_determine1(current_path, modification_list,
                                   modification_dict, mass_spectra_dict, blind_res, ion_relative_mode)

    # 差别1：比较无修饰和全体有修饰的谱图之间，前300候选离子（两位小数）对应的mass count的百分比来确定p-value
    without_mod_acc_peak_list = []
    for key in without_mod_peak_dict.keys():
        without_mod_acc_peak_list.append(without_mod_peak_dict[key])

    all_ion_acc_peak_list = []
    all_ion_acc_peak_dict = {}
    for mod in modification_list:
        for key in ion_dict[mod]['t_fine_relative_peak_dict'].keys():
            all_ion_acc_peak_dict[key] = []
    for mod in modification_list:
        for key in ion_dict[mod]['t_fine_relative_peak_dict'].keys():
            t_ion_acc_peak_list = []
            for i in ion_dict[mod]['t_fine_relative_peak_dict'][key]:
                t_ion_acc_peak_list.append(i[0])
            all_ion_acc_peak_dict[key] += t_ion_acc_peak_list
    sorted(all_ion_acc_peak_dict.items(),
           key=lambda item: len(item[1]), reverse=True)
    for i in all_ion_acc_peak_dict.keys():
        all_ion_acc_peak_list.append(np.mean(all_ion_acc_peak_dict[i]))

    all_ion_acc_peak_counter = {}
    for i in all_ion_acc_peak_dict.keys():
        c = dict(Counter(all_ion_acc_peak_dict[i]))
        all_ion_acc_peak_counter[i] = c

    # draw1(all_ion_acc_peak_counter)

    all_ion_peak_counter = {}
    for mod in modification_list:
        for item in ion_dict[mod]['t_position_counter_list']:
            all_ion_peak_counter[item[0]] = 0
    for mod in modification_list:
        for item in ion_dict[mod]['t_position_counter_list']:
            all_ion_peak_counter[item[0]] += item[1]

    without_mod_peak_counter = {}
    for item in without_mod_filtered_position_counter_list:
        without_mod_peak_counter[item[0]] = item[1]

    # draw2(all_ion_peak_counter)
    # draw3(without_mod_filtered_position_counter_list)

    # data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    # data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169,]

    # data_mod = [key for key in all_ion_peak_counter.keys() for i in range(all_ion_peak_counter[key])]
    # data_without_mod = [val[0] for val in without_mod_filtered_position_counter_list for i in range(val[1])]

    # encode，将有修饰和无修饰的encode成一种向量，向量下标为sim_mass,向量里的值为计数百分比

    merge_encode_list = list(set(without_mod_position_list).union(
        set([key for key in all_ion_peak_counter.keys()])))
    without_mod_encode_count = []
    mod_encode_count = []
    for key in merge_encode_list:
        without_mod_encode_count.append(0)
        mod_encode_count.append(0)
    for i in range(len(merge_encode_list)):
        if merge_encode_list[i] in all_ion_peak_counter.keys():
            mod_encode_count[i] = all_ion_peak_counter[merge_encode_list[i]]
        else:
            mod_encode_count[i] = 0
        if merge_encode_list[i] in without_mod_peak_counter.keys():
            without_mod_encode_count[i] = without_mod_peak_counter[merge_encode_list[i]]
        else:
            without_mod_encode_count[i] = 0

    data_mod1 = [value/max(mod_encode_count) for value in mod_encode_count]
    data_without_mod1 = [value/max(without_mod_encode_count)
                         for value in without_mod_encode_count]

    data_mod = [value/sum(mod_encode_count) for value in mod_encode_count]
    data_without_mod = [value/sum(without_mod_encode_count)
                        for value in without_mod_encode_count]
    # 差别1：比较无修饰和全体有修饰的谱图之间，前300候选离子（两位小数）对应的mass count的百分比来确定p-value
    stat1, p_value_for_compare_count1 = mannwhitneyu(
        data_mod1, data_without_mod1)
    stat, p_value_for_compare_count = mannwhitneyu(data_mod, data_without_mod)

    # 差别2：比较无修饰和全体有修饰的谱图之间，前300候选离子（两位小数）对应的所有谱峰差别来确定p-value
    all_ion_relative_peak = {}
    all_ion_relative_peak_result = {}
    for mod in modification_list:
        for key in ion_dict[mod]['t_fine_relative_peak_dict'].keys():
            all_ion_relative_peak[str(key)] = []
            all_ion_relative_peak_result[str(key)] = {}
            all_ion_relative_peak_result[str(key)]['mean'] = 0
            all_ion_relative_peak_result[str(key)]['median'] = 0
    for mod in modification_list:
        for key in ion_dict[mod]['t_fine_relative_peak_dict'].keys():
            for item in ion_dict[mod]['t_fine_relative_peak_dict'][key]:
                all_ion_relative_peak[str(key)].append(round(item[1], 2))
            all_ion_relative_peak_result[str(
                key)]['mean'] += np.mean(all_ion_relative_peak[str(key)])
            all_ion_relative_peak_result[str(
                key)]['median'] += np.median(all_ion_relative_peak[str(key)])

    without_mod_relative_peak = {}
    without_mod_relative_peak_result = {}
    for key in without_mod_fine_relative_peak_dict.keys():
        without_mod_relative_peak[str(key)] = []
        without_mod_relative_peak_result[str(key)] = {}
        without_mod_relative_peak_result[str(key)]['mean'] = 0
        without_mod_relative_peak_result[str(key)]['median'] = 0
    for key in without_mod_fine_relative_peak_dict.keys():
        for item in without_mod_fine_relative_peak_dict[key]:
            without_mod_relative_peak[str(key)].append(round(item[1], 2))
        without_mod_relative_peak_result[str(
            key)]['mean'] += np.mean(without_mod_relative_peak[str(key)])
        without_mod_relative_peak_result[str(
            key)]['median'] += np.median(without_mod_relative_peak[str(key)])

    without_mod_encode_relative_peak = []
    mod_encode_relative_peak = []
    for key in merge_encode_list:
        without_mod_encode_relative_peak.append(0)
        mod_encode_relative_peak.append(0)
    for i in range(len(merge_encode_list)):
        if str(merge_encode_list[i]) in all_ion_relative_peak_result.keys():
            mod_encode_relative_peak[i] = all_ion_relative_peak_result[str(
                merge_encode_list[i])]['mean']
        else:
            mod_encode_relative_peak[i] = 0
        if str(merge_encode_list[i]) in without_mod_relative_peak_result.keys():
            without_mod_encode_relative_peak[i] = without_mod_relative_peak_result[str(
                merge_encode_list[i])]['mean']
        else:
            without_mod_encode_relative_peak[i] = 0

    stat3, p_value_for_compare_relative_peak = mannwhitneyu(
        mod_encode_relative_peak, without_mod_encode_relative_peak)

    mod_encode_relative_peak1 = [((i - np.min(mod_encode_relative_peak)) / (np.max(
        mod_encode_relative_peak) - np.min(mod_encode_relative_peak))) for i in mod_encode_relative_peak]
    without_mod_encode_relative_peak1 = [((i - np.min(without_mod_encode_relative_peak)) / (np.max(
        without_mod_encode_relative_peak) - np.min(without_mod_encode_relative_peak))) for i in without_mod_encode_relative_peak]

    stat4, p_value_for_compare_relative_peak1 = mannwhitneyu(
        mod_encode_relative_peak1, without_mod_encode_relative_peak1)

    delta_relative_peak1 = [(mod_encode_relative_peak1[i] - without_mod_encode_relative_peak1[i])
                            for i in range(len(merge_encode_list))]
    delta_relative_peak = [(mod_encode_relative_peak[i] - without_mod_encode_relative_peak[i])
                           for i in range(len(merge_encode_list))]
    delta_percent_relative_peak = [0]*len(merge_encode_list)
    for i in range(len(merge_encode_list)):
        if without_mod_encode_relative_peak1[i] > 0:
            # delta_percent_relative_peak[i] = float((mod_encode_relative_peak1[i] - without_mod_encode_relative_peak1[i])/(without_mod_encode_relative_peak1[i]))
            delta_percent_relative_peak[i] = float(
                (mod_encode_relative_peak[i] - without_mod_encode_relative_peak[i])/(without_mod_encode_relative_peak[i]))
    t_max_delta_percent = max(delta_percent_relative_peak)+1
    for i in range(len(merge_encode_list)):
        if without_mod_encode_relative_peak1[i] == 0:
            delta_percent_relative_peak[i] = t_max_delta_percent

    delta_relative_peak_dict = {}
    delta_percent_relative_peak_dict = {}
    for i in range(len(merge_encode_list)):
        delta_relative_peak_dict[merge_encode_list[i]] = delta_relative_peak[i]
        delta_percent_relative_peak_dict[merge_encode_list[i]
                                         ] = delta_percent_relative_peak[i]

    plot_class_list = []
    for i in range(len(merge_encode_list)):
        if without_mod_encode_relative_peak1[i] == 0:
            plot_class_list.append(0)
        else:
            plot_class_list.append(1)

    # #定义散点图的标题,标题的大小和颜色等
    # plt.title('delta ion relative peak',fontsize=20,color='blue')
    # #定义坐标轴
    # plt.xlabel('ion m/z', fontsize=15,color='black')
    # plt.ylabel('ion delta m/z', fontsize=15,color='black')

    # index=list(range(0,len(merge_encode_list)))
    # #对于x刻度的设置
    # plt.xticks(index, rotation=30, fontsize=5)#横坐标刻度，也可以进行设置文字大小,如：fontsize=15，对文字进行旋转，如：rotation=30

    #     #绘制散点图
    # plt.scatter(index, delta_relative_peak,color='r', s=20)
    # plt.plot(index, np.array([0]*len(merge_encode_list)), c='b', ls='--')
    # for x,y in zip(index,delta_relative_peak):
    #     plt.text(x,y,'%f' %merge_encode_list[x] ,fontdict={'fontsize':10})
    # plt.show()
    draw4(merge_encode_list, delta_relative_peak, plot_class_list)
    draw4(merge_encode_list, delta_percent_relative_peak, plot_class_list)
    draw5(merge_encode_list, delta_relative_peak, plot_class_list)
    draw5(merge_encode_list, delta_percent_relative_peak, plot_class_list)
    return p_value_for_compare_count, p_value_for_compare_count1, p_value_for_compare_relative_peak, p_value_for_compare_relative_peak1, delta_relative_peak_dict

    # 差别3：从300个候选特征离子分别和无修饰再进行筛选，比较每个离子的p-value，选择p最小的，先排序看看


def draw5(merge_encode_list, delta_relative_peak, plot_class_list):
    # 定义散点图的标题,标题的大小和颜色等
    plt.title('delta ion relative peak', fontsize=20, color='blue')
    # 定义坐标轴
    plt.xlabel('ion m/z', fontsize=15, color='black')
    plt.ylabel('ion delta m/z', fontsize=15, color='black')

    index = list(range(0, len(merge_encode_list)))
    # 对于x刻度的设置
    # 横坐标刻度，也可以进行设置文字大小,如：fontsize=15，对文字进行旋转，如：rotation=30
    plt.xticks(index, rotation=30, fontsize=5)

    # 绘制散点图
    col = []
    colors = ['#FF0000', '#FFA500']

    index1 = [i for i in range(len(plot_class_list))
              if plot_class_list[i] == 0]
    index2 = [i for i in range(len(plot_class_list))
              if plot_class_list[i] == 1]
    y1 = [delta_relative_peak[i] for i in index1]
    y2 = [delta_relative_peak[i] for i in index2]
    # plt.scatter(index1, delta_relative_peak,color='r', s=20)
    plt.scatter(index1, y1, color='r', s=20)
    plt.scatter(index2, y2, color='g', s=20)
    plt.plot(index, np.array([0]*len(merge_encode_list)), c='b', ls='--')
    # for x,y in zip(index,delta_relative_peak):
    #     plt.text(x,y,'%.2f' %merge_encode_list[x] ,fontdict={'fontsize':10})
    plt.show()
    plt.close()


def draw4(merge_encode_list, delta_relative_peak, plot_class_list):
    # 定义散点图的标题,标题的大小和颜色等
    plt.title('delta ion relative peak', fontsize=20, color='blue')
    # 定义坐标轴
    plt.xlabel('ion m/z', fontsize=15, color='black')
    plt.ylabel('ion delta m/z', fontsize=15, color='black')

    index = list(range(0, len(merge_encode_list)))
    # 对于x刻度的设置
    # 横坐标刻度，也可以进行设置文字大小,如：fontsize=15，对文字进行旋转，如：rotation=30
    plt.xticks(index, rotation=30, fontsize=5)

    # 绘制散点图
    col = []
    colors = ['#FF0000', '#FFA500']

    index1 = [i for i in range(len(plot_class_list))
              if plot_class_list[i] == 0]
    index2 = [i for i in range(len(plot_class_list))
              if plot_class_list[i] == 1]
    y1 = [delta_relative_peak[i] for i in index1]
    y2 = [delta_relative_peak[i] for i in index2]
    # plt.scatter(index1, delta_relative_peak,color='r', s=20)
    plt.scatter(index1, y1, color='r', s=20)
    plt.scatter(index2, y2, color='g', s=20)
    plt.plot(index, np.array([0]*len(merge_encode_list)), c='b', ls='--')
    for x, y in zip(index, delta_relative_peak):
        plt.text(x, y, '%.2f' %
                 merge_encode_list[x], fontdict={'fontsize': 10})
    plt.show()
    plt.close()


def draw1(all_ion_acc_peak_counter):
    # fig = plt.figure()
    # ax1 = plt.axes(projection='3d')
    # ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
    x = []
    y = []
    z = []
    for i in all_ion_acc_peak_counter.keys():
        for item in all_ion_acc_peak_counter[i]:
            x.append(i)
            y.append(item)
            z.append(all_ion_acc_peak_counter[i][item])
    # ax1.scatter3D(x,y,z, cmap='Blues')  #绘制散点图
    # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
    plt.figure('Draw')
    plt.scatter(y, z)  # scatter绘制散点图
    plt.draw()  # 显示绘图
    plt.pause(10)  # 显示10秒
    plt.savefig("easyplot.jpg")  # 保存图象
    plt.show()
    plt.close()


def draw2(all_ion_peak_counter):
    x = []
    y = []
    for i in all_ion_peak_counter.keys():
        x.append(i)
        y.append(all_ion_peak_counter[i])
    plt.figure('Draw')
    plt.scatter(x, y)  # scatter绘制散点图
    plt.draw()  # 显示绘图
    plt.pause(10)  # 显示10秒
    plt.savefig("easyplot1.jpg")  # 保存图象
    plt.show()
    plt.close()


def draw3(without_mod_filtered_position_counter_list):
    x = []
    y = []
    for i in without_mod_filtered_position_counter_list:
        x.append(i[0])
        y.append(i[1])
        plt.figure('Draw')
    plt.scatter(x, y)  # scatter绘制散点图
    plt.draw()  # 显示绘图
    plt.pause(10)  # 显示10秒
    plt.savefig("easyplot2.jpg")  # 保存图象
    plt.show()
    plt.close()


def get_modification_from_result(pchem_summary_path):
    with open(pchem_summary_path, 'r') as f:
        lines = f.readlines()
    result = lines[1:]
    modification_list = []
    modification_dict = {}
    modification_PSM = {}
    for i in result:
        t_l_result = i.split('\t')
        modification_name = t_l_result[1]
        accurate_mass = t_l_result[2].split(' ')[0]
        modification_psm = t_l_result[-1][:-1]
        modification_list.append(modification_name)
        modification_dict[modification_name] = float(accurate_mass)
        modification_PSM[modification_name] = int(modification_psm)
    return modification_list, modification_dict, modification_PSM


def draw2pdf():
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    # 第一行第一列的图形
    x1 = [1, 2, 3, 4, 5]
    y1 = [5, 5.5, 7, 9, 12]
    ax[0, 0].plot(x1, y1, '-ro', ms=5)  # 红色实线连点

    # 第一行第二列的图形
    x2 = [1, 2, 3, 4, 5]
    y2 = [3, 4, 5, 7, 10]
    ax[0, 1].plot(x2, y2, '-.bo', ms=5)  # 蓝色虚线连点

    # 第二行第一列的图形
    x2 = [1, 2, 3, 4, 5]
    y2 = [3, 4, 5, 7, 10]
    ax[1, 0].plot(x2, y2, '-.bo', ms=5)  # 蓝色虚线连点

    # 第二行第二列的图形
    x2 = [1, 2, 3, 4, 5]
    y2 = [3, 4, 5, 7, 10]
    ax[1, 1].plot(x2, y2, '-.bo', ms=5)  # 蓝色虚线连点
    plt.savefig("Test.pdf", dpi=300)


def close_ion_learning(pchem_output_path, current_path, ion_type, modification_list, modification_dict, blind_res, mass_spectra_dict, modification_PSM, pchem_summary_path, ion_relative_mode):
    common_modification_dict = common_dict_create(current_path)
    without_mod_ion_result_path = os.path.join(
        pchem_output_path, 'pChem_without_mod_ion_result_mode{}.summary'.format(ion_relative_mode))

    without_mod_res, without_mod_position_list, without_mod_filtered_position_counter_list, without_mod_peak_dict, without_mod_fine_peak_dict, without_mod_fine_relative_peak_dict = without_mod_PSM_filter(
        current_path, blind_res, mass_spectra_dict, modification_dict, ion_type, ion_relative_mode)
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
            # without_mod_relative_peak_result[str(key)]['mean'] += np.mean([0]*(len(without_mod_res)-len(without_mod_relative_peak[str(key)]))+without_mod_relative_peak[str(key)])
        # without_mod_relative_peak_result[str(key)]['mean'] += np.median(without_mod_relative_peak[str(key)])
    # without_mod_json_file_path = os.path.join(current_path, 'pChem_without_mod.info')
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

    close_ion_result = {}

    mod_ion_result_path = os.path.join(
        pchem_output_path, 'pChem_mod_ion_result_mode{}.summary'.format(ion_relative_mode))
    ion_dict = ion_type_determine1(current_path, modification_list, modification_dict,
                                   mass_spectra_dict, blind_res, ion_type, ion_relative_mode)
    mod_filter = []
    with open(mod_ion_result_path, 'w', encoding='utf-8') as f1:
        for i in tqdm(modification_list):
            f1.write("Modification info for " + i +
                     ":(PSM = {}) ".format(modification_PSM[i]) + "\n")
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
                if ion_relative_mode != 4 and ion_relative_mode != 5:
                    mod_relative_peak_result[str(
                        key)]['mean'] += np.sum(mod_relative_peak[str(key)])/modification_PSM[i]
                # mod_relative_peak_result[str(key)]['mean'] += np.mean(mod_relative_peak[str(key)])
                elif ion_relative_mode == 4:
                    mod_relative_peak_result[str(key)]['mean'] += np.median([0]*(
                        modification_PSM[i]-len(mod_relative_peak[str(key)]))+mod_relative_peak[str(key)])
                else:
                    mod_relative_peak_result[str(
                        key)]['mean'] += np.sum(mod_relative_peak[str(key)])/modification_PSM[i]
                    # mod_relative_peak_result[str(key)]['mean'] += np.mean([0]*(modification_PSM[i]-len(mod_relative_peak[str(key)]))+mod_relative_peak[str(key)])
                # mod_relative_peak_result[str(key)]['mean'] += np.median(mod_relative_peak[str(key)])
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

            ion_common_acc_list = [ion_common_dict[i]
                                   for i in ion_common_dict.keys()]
            save_flag = False
            for k in range(0, len(ion_common_dict.keys())+1):
                if abs((mod_tuple[k][3] - float(ion_type)))/float(ion_type)*1000000 <= 20:
                    if k == 0:
                        save_flag = True
                        break
                    if k > 0:
                        flag_count = 0
                        for id in range(0, k):
                            for ic in ion_common_acc_list:
                                if abs(mod_tuple[id][3]-ic)/ic*1000000 <= 20:
                                    flag_count += 1
                        if flag_count == k:
                            save_flag = True
                            break
            if save_flag:
                mod_filter.append(i)
    print(mod_filter)
    with open(pchem_summary_path, 'r') as f2:
        lines = f2.readlines()
    line_0 = lines[0]
    result = lines[1:]
    line_re = []
    line_re.append(line_0)
    for mod in mod_filter:
        for line in result:
            if mod in line:
                line_re.append(line)
                break

    pchem_summary_path1 = os.path.join(
        pchem_output_path, 'pChem_ion_filter_mode{}.summary'.format(ion_relative_mode))
    with open(pchem_summary_path1, 'w', encoding='utf-8') as f3:
        for index, i in enumerate(line_re):
            if index == 0:
                t = i
            else:
                line_res = i.split("\t")
                line_res[0] = str(index)
                t = "\t".join(line_res)
            f3.write(t)
        f3.write('\n')
    return pchem_summary_path1


def parameter_pick1(line):
    eq_idx = line.find('=')
    parameter_content = line[eq_idx+1:].strip()
    return parameter_content


def parameter_file_read_ion(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    parameter_dict = {}
    for i in range(len(lines)):
        # if 'pchem_cfg_path' in lines[i]:
        #     parameter_dict['pchem_cfg_path'] = parameter_pick1(lines[i])
        if 'output_path' in lines[i]:
            parameter_dict['output_path'] = parameter_pick1(lines[i])
        if 'ion_type' in lines[i]:
            parameter_dict['ion_type'] = parameter_pick1(lines[i])
        if 'p_value_threshold' in lines[i]:
            parameter_dict['p_value_threshold'] = parameter_pick1(lines[i])
        if 'ion_relative_mode' in lines[i]:
            parameter_dict['ion_relative_mode'] = parameter_pick1(lines[i])
        if 'ion_rank_threshold' in lines[i]:
            parameter_dict['ion_rank_threshold'] = parameter_pick1(lines[i])
    return parameter_dict


def modify_p_summary(pchem_summary_path1, p_value_threshold):
    if p_value_threshold == 1:
        return
    else:
        with open(pchem_summary_path1, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        os.remove(pchem_summary_path1)
        lines_0 = lines[0]
        result = []
        result.append(lines_0)
        for line in lines[1:]:
            if line == '\n':
                break
            line_res = line.split("\t")
            pos0_p = float(line_res[3].split("|")[-1])
            if pos0_p > p_value_threshold:
                continue
            else:
                pos_list = line_res[4].split(";")[:-1]
                pos_list = [i[2:] if i[0] == " " else i[:] for i in pos_list]
                result_list = []
                line_res_4 = ""
                for index, i in enumerate(pos_list):
                    l = i.split(", ")
                    pos_p = l[-1][:-1]
                    if float(pos_p) <= p_value_threshold:
                        result_list.append(index)
                for i in result_list:
                    line_res_4 += pos_list[i]
                    line_res_4 += ";  "
                line_res[4] = line_res_4
                result.append("\t".join(line_res))
        with open(pchem_summary_path1, 'w', encoding='utf-8') as f3:
            for index, i in enumerate(result):
                if index == 0:
                    t = i
                else:
                    line_res = i.split("\t")
                    line_res[0] = str(index)
                    t = "\t".join(line_res)
                f3.write(t)
            f3.write('\n')
        # with open(pchem_summary_path1, 'w', encoding='utf-8') as f3:
        #     f3.write(lines_0)
        #     for i in result:
        #         f3.write(i)


def take_closest(myList, myNumber):
    """
     Assumes myList is sorted. Returns closest value to myNumber.
     If two numbers are equally close, return the smallest number.
     If number is outside of min or max return False
    """
    # if (myNumber > myList[-1] or myNumber < myList[0]):
    #     return False
    res_list = []
    for i in myList:
        res_list.append(float(i.split(" ")[0]))
    pos = bisect_left(res_list, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = res_list[pos - 1]
    after = res_list[pos]
    if after - myNumber < myNumber - before:
        return myList[pos]
    else:
        return myList[pos - 1]


def get_ms2_ms1_scan_dict(ms2_path, ms1_path, blind_res):
    with open(ms2_path, 'r') as f:
        lines = f.readlines()
    index_list = []
    for index, line in enumerate(lines):
        if "S	" in line:
            index_list.append(index)
    ms2_ms1_scan_dict = {}
    # ms2_mz_dict = {}
    for i in index_list:
        ms2_scan = int(lines[i].split("\t")[1])
        ms2_ms1_scan_dict[ms2_scan] = int(lines[i+6].split("\t")[-1][:-1])

    blind_res_dict = {}
    blind_res_list = []
    for line in blind_res:
        t_line = line.split("\t")
        # print(t_line)
        t_ms2_name = t_line[0]
        t_ms2_scan = t_line[1]
        t_exp_MH = t_line[2]
        t_ms2_charge = t_line[3]
        t_mz = round(((int(t_ms2_charge)-1) *
                     element_dict['Pm']+float(t_exp_MH))/int(t_ms2_charge), 6)
        blind_res_dict[t_ms2_name] = {}
        blind_res_dict[t_ms2_name]['ms2_scan'] = int(t_ms2_scan)
        blind_res_dict[t_ms2_name]['mz'] = t_mz
        blind_res_dict[t_ms2_name]['ms1_scan'] = ms2_ms1_scan_dict[int(
            t_ms2_scan)]
        blind_res_list.append(
            (t_ms2_name, int(t_ms2_scan), ms2_ms1_scan_dict[int(t_ms2_scan)], t_mz))

    blind_res_list = sorted(blind_res_list, key=lambda k: k[2])

    with open(ms1_path, 'r') as f:
        lines1 = f.readlines()
    start_index_list_ms1_dict = {}
    for index, line in enumerate(lines1):
        if "S	" in line:
            t = line.split("\t")
            start_index_list_ms1_dict[int(t[1])] = index

    end_index_list_ms1_dict = {}
    # start_index_l = list(start_index_list_ms1_dict.keys())
    start_index_l = [i[1] for i in start_index_list_ms1_dict.items()]
    end_index_l = start_index_l[1:]
    end_index_l = [(i-1) for i in end_index_l]
    end_index_l.append(len(lines1)-1)
    blind_res_ms1_dict = {}
    start_time = time.time()
    ppm_list = []
    for item in tqdm(blind_res_list):
        t_ms1_scan = item[2]
        peak_index = start_index_list_ms1_dict[t_ms1_scan]
        start_index = peak_index
        end_index = end_index_l[start_index_l.index(start_index)]
        t_peak_list = lines1[start_index+5:end_index+1]
        blind_res_ms1_dict[item[0]] = take_closest(t_peak_list, item[3])
        t_ppm = abs(item[3]-float(blind_res_ms1_dict[item[0]].split(" ")[0])) / \
            float(blind_res_ms1_dict[item[0]].split(" ")[0])*1000000
        # if t_ppm <= 20:
        #     ppm_list.append((item[0], t_ppm))
        #     continue
        # else:
        # 可能仪器出现了问题

        # ppm_list.append((item[0], t_ppm))

    end_time = time.time()
    print("Get ms1 cost time (s): ", round(end_time - start_time, 1))
    return ms2_ms1_scan_dict, blind_res_dict, blind_res_ms1_dict


if __name__ == "__main__":
    # # draw2pdf()
    current_path = os.getcwd()
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg')
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    blind_res_path = os.path.join(
        parameter_dict_ion['output_path'], "source/blind/pFind-Filtered.spectra")
    pp_path = os.path.join(parameter_dict_ion['output_path'], "source/pParse")
    mgf_path = glob.glob(os.path.join(pp_path, "*.mgf"))
    mgf_path = mgf_path[0]
    ms2_path = glob.glob(os.path.join(pp_path, "*.ms2"))
    ms2_path = ms2_path[0]
    ms1_path = glob.glob(os.path.join(pp_path, "*.ms1"))
    ms1_path = ms1_path[0]
    # ms2_ms1_scan_dict = get_ms2_ms1_scan_dict(ms2_path, ms1_path)
    start_time = time.time()
    blind_res = blind_res_read(blind_res_path)
    end_time = time.time()
    print("blind filter file read time (s): ", round(end_time - start_time, 1))
    ms2_ms1_scan_dict, blind_res_dict, blind_res_ms1_dict = get_ms2_ms1_scan_dict(
        ms2_path, ms1_path, blind_res)
    start_time = time.time()
    mass_spectra_dict = mgf_read_for_ion_pfind_filter1(
        mgf_path, blind_res, blind_res_ms1_dict)

    end_time = time.time()
    print("reading mgf file cost time (s): ", round(end_time - start_time, 1))
    pchem_summary_path = os.path.join(
        parameter_dict_ion['output_path'], "reporting_summary/pChem.summary")

    modification_list, modification_dict, modification_PSM = get_modification_from_result(
        pchem_summary_path)
    p_value_threshold = float(parameter_dict_ion['p_value_threshold'])
    pchem_output_path = os.path.join(
        parameter_dict_ion['output_path'], "reporting_summary")

    ion_type = parameter_dict_ion['ion_type']
    ion_relative_mode = int(parameter_dict_ion['ion_relative_mode'])
    pchem_summary_path1 = close_ion_learning(pchem_output_path, current_path, ion_type, modification_list,
                                             modification_dict, blind_res, mass_spectra_dict, modification_PSM, pchem_summary_path, ion_relative_mode)

    modify_p_summary(pchem_summary_path1, p_value_threshold)

    # feature_ion_compare(current_path, modification_list, modification_dict, blind_res, mass_spectra_dict)
