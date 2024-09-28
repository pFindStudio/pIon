import os
from utils import parameter_file_read, parameter_file_read_ion
from parameter import common_dict_create
from mass_diff_correction import ppm_system_shift_compute, origin_mass_list_generate, small_delta_filter, mass_static
import numpy as np
from collections import Counter
from parameter import amino_acid_dict, proton_mass, h2o_mass
from confidence_set import prior_distribution_compute, total_trail_compute
from tqdm import tqdm
import math
from scipy.stats import binom_test


# 读取盲搜结果并生成未知质量数修饰
def generate_mass_diff_list(pfind_summary_path, current_path):
    with open(pfind_summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    mass_diff_lines = []
    i = 0
    while i < len(lines):
        if 'Modifications:' in lines[i]:
            i += 1
            mass_diff_lines.append(lines[i])
            i += 1
            while i < len(lines):
                if '------' in lines[i]:
                    break
                if 'PFIND' in lines[i]:
                    mass_diff_lines.append(lines[i])
                i += 1
        i += 1
    return mass_diff_lines

# 通过多次迭代优化


def iterative_compute_mass(mass_diff, original_unknown_list):
    target_mass = float(mass_diff.split('_')[2])
    data = []
    for mass in original_unknown_list:
        data.append(mass)
    spectrum_num = len(data)
    if len(data) == 0:
        return 0.0, 1.0, 0
    mu0 = np.mean(data)
    std0 = np.std(data)
    times = 1

    while True:
        if len(data) <= 100:
            break
        a = []
        for mass in original_unknown_list:
            if mass >= mu0 - 0.01 and mass <= mu0 + 0.01:
                a.append(mass)
        spectrum_num = max(spectrum_num, len(a))
        mu, sigma = np.mean(a), np.std(a)
        data = a
        if mu == mu0 or times == 3:
            break
        mu0 = mu
        std0 = sigma
        times += 1
    return mu0, std0, spectrum_num


# 生成所有未知修饰的质量列表
def generate_origin_mass(lines, common_dict, factor_shift):
    origin_mass_list = []
    origin_mass_dict = {}
    for line in lines:
        if 'PFIND_DELTA' not in line:
            continue
        line = line.split('\t')
        sequence = line[5]
        # 5,PFIND_DELTA_95.04;30,Oxidation[M];LADQCTGLQGFLVFHSFGGGTGSGFTSLLMER
        mod_list = line[10].split(';')[:-1]
        for mod in mod_list:
            if 'PFIND_DELTA' in mod:
                mod_name = mod.split(',')[1]
                mod_pos = int(mod.split(',')[0])
                mod_site = sequence[mod_pos-1]

            else:
                continue
        parent_mass = float(line[2]) * factor_shift
        # sequence = line[5]
        amino_mass = 0.0
        for a in sequence:
            if a in amino_acid_dict.keys():
                amino_mass += amino_acid_dict[a]
        mod_mass = parent_mass - amino_mass - proton_mass - h2o_mass
        if len(mod_list) > 1:
            for mod in mod_list:
                mod = mod.split(',')[1]
                if 'PFIND_DELTA' in mod:
                    continue
                mod_mass -= common_dict[mod]
        origin_mass_list.append(float('%.6f' % (mod_mass)))

    return origin_mass_list


def mass_correct1(current_path, blind_path, mass_diff_list, parameter_dict_ion, system_correct='mean'):
    # 读取常见修饰列表
    common_dict = common_dict_create(current_path)
    # 读取盲搜鉴定结果文件
    with open(blind_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[1:]
    system_shift = ppm_system_shift_compute(lines, system_correct)
    factor_shift = 1.0 / (1.0 + system_shift / 1000000.0)
    # 计算未知修饰的精确质量
    original_unknown_list = generate_origin_mass(
        lines, common_dict, factor_shift)
    mass_dict = {}
    mod_number_dict = {}

    for mass_diff in mass_diff_list:
        sim_mod_name = mass_diff
        accurate_mass_diff, std, spectrum_num = iterative_compute_mass(
            mass_diff, original_unknown_list)
        mass_dict[sim_mod_name] = (accurate_mass_diff, spectrum_num)

    reserve_mass_dict = {}
    for key in mass_dict.keys():
        if mass_dict[key][0] >= parameter_dict_ion['explain_modification_min'] and mass_dict[key][0] <= parameter_dict_ion['explain_modification_max']:
            if mass_dict[key][0] != 0.0:
                reserve_mass_dict[key] = mass_dict[key]

    mass_list = []
    for key in reserve_mass_dict.keys():
        mass_list.append(reserve_mass_dict[key])
    sorted_mass_list = sorted(mass_list, key=lambda x: x[0])

    # 第一种聚类方式
    # new_mass_list = []
    # cluster_start = sorted_mass_list[0]
    # cluster_sum = sorted_mass_list[0][0] * sorted_mass_list[0][1]
    # cluster_count = sorted_mass_list[0][1]

    # for i in range(1, len(sorted_mass_list)):
    #     # 如果当前值与聚类起始值的差距小于等于0.01，则加入当前聚类
    #     if sorted_mass_list[i][0] - cluster_start[0] < 0.1:
    #         # if str(sorted_mass_list[i][0]).split(".")[0] == str(cluster_start[0]).split(".")[0]:
    #         cluster_sum += sorted_mass_list[i][0] * sorted_mass_list[i][1]
    #         cluster_count += sorted_mass_list[i][1]
    #     # 否则，将当前聚类的均值加入新list，并更新聚类的起始值和计数
    #     else:
    #         new_mass_list.append((cluster_sum / cluster_count, cluster_count))
    #         cluster_start = sorted_mass_list[i]
    #         cluster_sum = sorted_mass_list[i][0] * sorted_mass_list[i][1]
    #         cluster_count = sorted_mass_list[i][1]

    # # 处理最后一个聚类
    # new_mass_list.append((cluster_sum / cluster_count, cluster_count))

    # 第二种聚类方式
    clusters = {}
    new_mass_list = []
    # 遍历原始list
    for mass_num in sorted_mass_list:
        # 计算聚类的标签
        label = int(mass_num[0] // 0.1)

        # 将浮点数加入对应标签的聚类中
        if label in clusters:
            clusters[label].append(mass_num)
        else:
            clusters[label] = [mass_num]

    # 计算聚类的均值并放入新list中
    # new_lst = [c[0]*c[1] / len(cluster) for cluster in clusters.values() for c in cluster]
    for i in clusters.keys():
        c = clusters[i]
        c_sum = 0
        c_num = 0
        for j in c:
            c_sum += j[0] * j[1]
            c_num += j[1]
        new_mass_list.append(c_sum/c_num)

    return mass_dict, mod_number_dict


def mass_correct2(current_path, blind_path, mass_diff_list, parameter_dict_ion, system_correct='mean'):
    # 读取常见修饰列表
    common_dict = common_dict_create(current_path)
    # 读取盲搜鉴定结果文件
    with open(blind_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = lines[1:]
    system_shift = ppm_system_shift_compute(lines, system_correct)
    factor_shift = 1.0 / (1.0 + system_shift / 1000000.0)
    # factor_shift = 1.0
    # 计算未知修饰的精确质量
    original_unknown_list = generate_origin_mass(
        lines, common_dict, factor_shift)
    mass_dict = {}
    mod_number_dict = {}

    for line in tqdm(lines):
        if "PFIND_DELTA_" not in line:
            continue
        else:
            line = line.split('\t')
            sequence = line[5]
            # 5,PFIND_DELTA_95.04;30,Oxidation[M];LADQCTGLQGFLVFHSFGGGTGSGFTSLLMER
            mod_list = line[10].split(';')[:-1]
            parent_mass = float(line[2]) * factor_shift
            amino_mass = 0.0
            for a in sequence:
                if a in amino_acid_dict.keys():
                    amino_mass += amino_acid_dict[a]
            mod_mass = parent_mass - amino_mass - proton_mass - h2o_mass
            if len(mod_list) > 1:
                for mod in mod_list:
                    mod = mod.split(',')[1]
                    if 'PFIND_DELTA' in mod:
                        mod_name = mod
                        continue
                    mod_mass -= common_dict[mod]
            else:
                mod_name = mod_list[0].split(',')[1]

            if mod_name in mass_dict.keys():
                mass_dict[mod_name].append(mod_mass)
            else:
                mass_dict[mod_name] = [mod_mass]

    mass_dict_merge = {}

    for key in mass_dict.keys():
        if key in mass_diff_list:
            key_1 = key[:-1]
            if key_1 not in mass_dict_merge.keys():
                mass_dict_merge[key_1] = mass_dict[key]
            else:
                mass_dict_merge[key_1] += mass_dict[key]

    new_mass_dict = {}
    new_mass_acc_dict = {}
    for key in mass_dict_merge.keys():
        # if key in mass_diff_list:
        new_mass_dict[key] = mass_dict_merge[key]
    for key in new_mass_dict.keys():
        mu0, std0, spectrum_num = iterative_compute_mass(
            key, new_mass_dict[key])
        new_mass_acc_dict[key] = mu0

    mod_position_dict = {}
    spectra_num = len(lines)
    for i in range(1, spectra_num):
        if len(lines[i]) < 4:
            break
        sequence = lines[i].split('\t')[5]
        mod_list = lines[i].split('\t')[10].split(';')[:-1]
        for mod in mod_list:
            pos, mod_name = mod.split(',')
            if mod_name in mass_diff_list:
                pos = int(pos)
                mod_name_sim = mod_name[:-1]
                if mod_name_sim not in mod_position_dict.keys():
                    mod_position_dict[mod_name_sim] = []
                # mod_number_dict[mod_name_sim] += 1
                if pos == 0 or pos == 1:
                    mod_position_dict[mod_name_sim].append(sequence[0])
                    mod_position_dict[mod_name_sim].append('N-SIDE')
                elif pos >= len(sequence):
                    mod_position_dict[mod_name_sim].append(sequence[-1])
                    mod_position_dict[mod_name_sim].append('C-SIDE')
                else:
                    mod_position_dict[mod_name_sim].append(sequence[pos-1])

    mod_static_dict = {}
    for mod_name in mod_position_dict.keys():
        counter = Counter(mod_position_dict[mod_name])
        mod_static_dict[mod_name] = counter

    p_value_dict = {}
    for mod in mod_position_dict.keys():
        mod_prior_distribution = prior_distribution_compute(blind_path, mod)
        mod_total_trail = total_trail_compute1(mod_position_dict[mod])
        mod_p_value_dict = {}
        p_value_dict[mod] = {}
        # counter_mod = mod_static_dict[mod]
        for i, j in dict(mod_static_dict[mod]).items():
            if i in mod_prior_distribution.keys():
                mod_p_value_dict[i] = format(binom_test(
                    j, mod_total_trail, p=mod_prior_distribution[i], alternative='greater'), '.4f')
        p_value_dict[mod] = mod_p_value_dict
        # print("")
    reserve_p_value_dict = {}
    for mod in p_value_dict.keys():
        p_dict = p_value_dict[mod]
        reserve_p_value_dict[mod] = []
        for i in p_dict.keys():
            if float(p_dict[i]) <= 0.0001:
                reserve_p_value_dict[mod].append(i)
        if len(reserve_p_value_dict[mod]) == 0:
            del reserve_p_value_dict[mod]

    reserve_pfind_delta_dict = {}
    for mod in reserve_p_value_dict.keys():
        reserve_pfind_delta_dict[mod] = {}
        reserve_pfind_delta_dict[mod]['accurate_mass'] = new_mass_acc_dict[mod]
        reserve_pfind_delta_dict[mod]['site_list'] = []
        for site in reserve_p_value_dict[mod]:
            reserve_pfind_delta_dict[mod]['site_list'].append(
                [site, mod_static_dict[mod][site], p_value_dict[mod][site]])

    return new_mass_acc_dict, new_mass_dict


def total_trail_compute1(position_list):
    total_num = 0
    for pair in position_list:
        # if 'N-CIDE' == pair[0]:
        #    continue
        total_num += 1
    return total_num


# 统计修饰发生的位点分布
def mass_static1(blind_path, mass_diff_list):
    mod_position_dict = {}
    mod_number_dict = {}

    side_flag = True
    for mass in mass_diff_list:
        mod_position_dict[mass] = []
        mod_number_dict[mass] = 0

    with open(blind_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    spectra_num = len(lines)
    for i in range(1, spectra_num):
        if len(lines[i]) < 4:
            break
        sequence = lines[i].split('\t')[5]
        mod_list = lines[i].split('\t')[10].split(';')[:-1]
        for mod in mod_list:
            pos, mod_name = mod.split(',')
            if mod_name in mass_diff_list:
                pos = int(pos)
                mod_number_dict[mod_name] += 1
                if pos == 0 or pos == 1:
                    mod_position_dict[mod_name].append(sequence[0])
                    if side_flag == True:
                        mod_position_dict[mod_name].append('N-SIDE')
                elif pos >= len(sequence):
                    mod_position_dict[mod_name].append(sequence[-1])
                    if side_flag == True:
                        mod_position_dict[mod_name].append('C-SIDE')
                else:
                    mod_position_dict[mod_name].append(sequence[pos-1])

    mod_static_dict = {}
    for mod_name in mass_diff_list:
        counter = Counter(mod_position_dict[mod_name])
        mod_static_dict[mod_name] = counter
    return mod_static_dict, mod_number_dict


def get_pfind_delta_modifications(parameter_dict_ion):
    output_path = parameter_dict_ion['output_path']
    source_path = os.path.join(output_path, 'source')
    blind_path = os.path.join(source_path, 'blind')
    pfind_summary_path = os.path.join(blind_path, 'pFind.summary')
    blind_path = os.path.join(blind_path, 'pFind-Filtered.spectra')
    mass_diff_list = generate_mass_diff_list(pfind_summary_path, current_path)
    lines = mass_diff_list[1:]
    mass_diff_list = []
    mod2pep = {}
    for line in lines:
        # print(line)
        if len(line) < 2:
            break
        mod, pep = line.split('\t')[0], line.split('\t')[1].split()[0]
        mass_diff_list.append(mod)
        mod2pep[mod] = pep
    # mass_diff_list = mass_diff_list[1:]\
    # 发现20-200质量的探针衍生修饰，因为点击化学效率不可能100%
    name2mass, mass_diff_list = small_delta_filter1(mass_diff_list, 20, 200)
    mod_static_dict, mod_number_dict = mass_static1(
        blind_path, mass_diff_list)  # 统计位点分布
    mass_dict, mod_number_dict = mass_correct2(
        current_path, blind_path, mass_diff_list, parameter_dict_ion, system_correct='mean')
    return mass_diff_list


def small_delta_filter1(mass_difference_list, min_mass, max_mass):
    name2mass = {}
    new_mass_diff_list = []
    for mass_diff in mass_difference_list:
        mass = mass_diff.split('_')[-1]
        if mass[0] == '-':
            continue
        mass = float(mass)
        if mass <= min_mass or mass >= max_mass:
            continue
        name2mass[mass_diff] = mass
        new_mass_diff_list.append(mass_diff)
    return name2mass, new_mass_diff_list


if __name__ == "__main__":
    current_path = os.getcwd()
    cfg_path = os.path.join(current_path, 'pChem-ion.cfg')
    parameter_dict_ion = parameter_file_read_ion(cfg_path)
    mass_diff_lines = get_pfind_delta_modifications(parameter_dict_ion)
