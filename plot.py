'''
Email: pengyaping21@mails.ucas.ac.cn
Author: pengyaping21
LastEditors: pengyaping21
Date: 2023-05-15 15:38:45
LastEditTime: 2023-05-15 16:03:39
FilePath: \pChem-main\plot.py
Description: Do not edit
'''
import matplotlib.pyplot as plt
import os


class Plot:
    def __init__(self, x=None, y=None, title=None, xlabel=None, ylabel=None, save_path=None, save_name=None):
        self.x = x
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.save_path = save_path
        self.save_name = save_name

    def line_plot(self):
        plt.plot(self.x, self.y)
        self._add_labels()
        self._save_fig()

    def scatter_plot(self):
        plt.scatter(self.x, self.y)
        self._add_labels()
        self._save_fig()

    def bar_plot(self):
        plt.bar(self.x, self.y)
        self._add_labels()

    def _add_labels(self):
        if self.title:
            plt.title(self.title)
        if self.xlabel:
            plt.xlabel(self.xlabel)
        if self.ylabel:
            plt.ylabel(self.ylabel)

    def _save_fig(self):
        plt.savefig(os.path.join(self.save_path, self.save_name))
        plt.close()


# x_values = [1, 2, 3, 4, 5]
# y_values = [10, 8, 6, 4, 2]

# plot_obj = Plot(x=x_values, y=y_values, title="Simple Plot", xlabel="X Axis",
#                 ylabel="Y Axis", save_path="G:/pChem_ion-2023-4/pChem-main/tmp", save_name="ion.png")

# # 绘制线图
# plot_obj.line_plot()

# # 绘制散点图
# plot_obj.scatter_plot()

# # 绘制柱状图
# plot_obj.bar_plot()

# 模拟二级质谱数据
data = {
    600.1234213: 100,
    'Fragment Ions': {
        126.12773: 20,
        110.07754: 40,
        800.2013445: 15,
        1000.202321: 25
    }
}

# 提取母离子和碎片离子的质量数和相对丰度信息
parent_ion_mass, parent_ion_intensity = list(data.items())[0]
fragment_ion_masses, fragment_ion_intensities = zip(
    *data['Fragment Ions'].items())

# 设置画布大小
fig, ax = plt.subplots(figsize=(800, 60))

# 绘制母离子峰
ax.vlines(parent_ion_mass, 0, parent_ion_intensity,
          colors='black', linewidth=2)

# 绘制每个碎片离子峰
for i in range(len(fragment_ion_masses)):
    ax.vlines(
        fragment_ion_masses[i], 0, fragment_ion_intensities[i], colors='red', linewidth=2)

# 添加垂直参考线和标签
ax.vlines(fragment_ion_masses, 0, max(
    fragment_ion_intensities), colors='gray', linestyle='--')
ax.set_xticks(list(data.keys()))
ax.tick_params(axis='x', rotation=90)

# 设置图表标题和坐标轴标签
ax.set_title('MSMS')
ax.set_xlabel('m/z')
ax.set_ylabel('relative intensity')

plt.show()
