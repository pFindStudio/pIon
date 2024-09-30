'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2022-09-06 10:13:10
LastEditTime: 2023-07-07 14:58:49
FilePath: \pChem-main\parameter.py
Description: Do not edit
'''
import os


# 这里和pFind中的值有细微的差别，但是应该不太影响？如果未来需要改成pFind中的值，可以采用注释中的element_dict
element_dict = {
    "C": 12.0000000,
    "H": 1.00782503207,
    "Pm": 1.00727647012,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "S": 31.972071
}
# element_dict={
#     "C": 12.0000000,
#     "H": 1.007825035,
#     "Pm": 1.00727647012,
#     "N": 14.003074,
#     "O": 15.99491463,
#     "S": 31.972072,
#     "C13": 13.00335
# }


amino_acid_dict = {
    "A": element_dict["C"]*3 + element_dict["H"]*5 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "B": element_dict["C"]*0,
    "C": element_dict["C"]*3 + element_dict["H"]*5 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*1,
    "D": element_dict["C"]*4 + element_dict["H"]*5 + element_dict["N"]*1 + element_dict["O"]*3 + element_dict["S"]*0,
    "E": element_dict["C"]*5 + element_dict["H"]*7 + element_dict["N"]*1 + element_dict["O"]*3 + element_dict["S"]*0,
    "F": element_dict["C"]*9 + element_dict["H"]*9 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "G": element_dict["C"]*2 + element_dict["H"]*3 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "H": element_dict["C"]*6 + element_dict["H"]*7 + element_dict["N"]*3 + element_dict["O"]*1 + element_dict["S"]*0,
    "I": element_dict["C"]*6 + element_dict["H"]*11 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "J": element_dict["C"]*0,
    "K": element_dict["C"]*6 + element_dict["H"]*12 + element_dict["N"]*2 + element_dict["O"]*1 + element_dict["S"]*0,
    "L": element_dict["C"]*6 + element_dict["H"]*11 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "M": element_dict["C"]*5 + element_dict["H"]*9 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*1,
    "N": element_dict["C"]*4 + element_dict["H"]*6 + element_dict["N"]*2 + element_dict["O"]*2 + element_dict["S"]*0,
    "O": element_dict["C"]*0,
    "P": element_dict["C"]*5 + element_dict["H"]*7 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "Q": element_dict["C"]*5 + element_dict["H"]*8 + element_dict["N"]*2 + element_dict["O"]*2 + element_dict["S"]*0,
    "R": element_dict["C"]*6 + element_dict["H"]*12 + element_dict["N"]*4 + element_dict["O"]*1 + element_dict["S"]*0,
    "S": element_dict["C"]*3 + element_dict["H"]*5 + element_dict["N"]*1 + element_dict["O"]*2 + element_dict["S"]*0,
    "T": element_dict["C"]*4 + element_dict["H"]*7 + element_dict["N"]*1 + element_dict["O"]*2 + element_dict["S"]*0,
    "U": element_dict["C"]*0,
    "V": element_dict["C"]*5 + element_dict["H"]*9 + element_dict["N"]*1 + element_dict["O"]*1 + element_dict["S"]*0,
    "W": element_dict["C"]*11 + element_dict["H"]*10 + element_dict["N"]*2 + element_dict["O"]*1 + element_dict["S"]*0,
    "X": element_dict["C"]*0,
    "Y": element_dict["C"]*9 + element_dict["H"]*9 + element_dict["N"]*1 + element_dict["O"]*2 + element_dict["S"]*0,
    "Z": element_dict["C"]*0,
}

h2o_mass = element_dict["H"]*2 + element_dict["O"]*1
proton_mass = element_dict['Pm']

ion_common_dict = {
    "Him": amino_acid_dict['H'] - element_dict["C"] - element_dict["O"] + element_dict['Pm'],
    "Fim": amino_acid_dict['F'] - element_dict["C"] - element_dict["O"] + element_dict['Pm'],
    "Yim": amino_acid_dict['Y'] - element_dict["C"] - element_dict["O"] + element_dict['Pm'],
    "Kpm": amino_acid_dict['K'] + element_dict['Pm'],
    # "Kpm2": element_dict["C"]*6 + element_dict["H"]*11 + element_dict["N"]*1 + element_dict["O"]*2 + element_dict['Pm'],
    # "Rim": element_dict["C"]*5 + element_dict["H"]*12 + element_dict["N"]*4 + element_dict["O"]*0 + element_dict['Pm'],
    # "Ky1": element_dict["C"]*6 + element_dict["H"]*14 + element_dict["N"]*2 + element_dict["O"]*2 + element_dict['Pm'],
    # "Wim": element_dict["C"]*10 + element_dict["H"]*10 + element_dict["N"]*2 + element_dict["O"]*0 + element_dict['Pm'],
    # "Ry1": element_dict["C"]*6 + element_dict["H"]*14 + element_dict["N"]*4 + element_dict["O"]*2 + element_dict['Pm'],
    # "common_a2_b2": element_dict["C"]*9 + element_dict["H"]*16 + element_dict["N"]*2 + element_dict["O"]*3 + element_dict['Pm'],
    # "common_a2_b2_2": element_dict["C"]*10 + element_dict["H"]*18 + element_dict["N"]*2 + element_dict["O"]*3 + element_dict['Pm'],
    # "Qim": element_dict["C"]*4 + element_dict["H"]*8 + element_dict["N"]*2 + element_dict["O"]*1 + element_dict['Pm'],
    # "Qim2": element_dict["C"]*4 + element_dict["H"]*7 + element_dict["N"]*1 + element_dict["O"]*2 + element_dict['Pm'],
}


# 读取选择的常见修饰，返回dict
def common_dict_create(current_path):
    modification_path = os.path.join(current_path, 'modification-null.ini')
    with open(modification_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 1
    common_dict = {}
    while i < len(lines):
        if len(lines[i]) < 2:
            break
        mod_name = lines[i].split()[0]
        eq_idx = mod_name.find('=')
        mod_name = mod_name[eq_idx+1:]
        mod_mass = lines[i+1].split()[2]
        common_dict[mod_name] = float(mod_mass)
        i += 2
    return common_dict


if __name__ == "__main__":
    print(amino_acid_dict)
    print(ion_common_dict)
    print(334.212130-element_dict['H']+element_dict['S'] +
          3*element_dict['H']+element_dict['Pm'])

    # 计算MH+
    seq = "SKDDQVTVIGAGVTLHEALAAAELLK"
    sum = 0.0
    for i in seq:
        sum += amino_acid_dict[i]
    sum += h2o_mass
    sum += element_dict['Pm']
    print(sum)

    t = 2453.224623
    t += element_dict['Pm']
    t /= 2
    print(t)
