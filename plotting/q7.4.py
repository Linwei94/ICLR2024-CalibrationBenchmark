import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
df = pd.read_csv('./cifar10_tss.csv')
acc = df["acc"]
ece = df["pre_ECE_15"]
beta = 1
df["hcs_ece_beta"] = (1 + beta) * (acc * (1 - ece))/(beta * acc + (1-ece))
print(df[['arch', "hcs_ece_beta"]].sort_values(by="hcs_ece_beta", ascending=False).head(20))
# 提取architecture

def extract_arch(arch_string, position):
    arch_list = arch_string.split('|')[1:-1]
    arch_list.remove('+')
    arch_list.remove('+')
    print(arch_list)
    # arch = arch_list[position].split('~')[0]
    arch = arch_list[position]
    print(arch)
    if 'conv' in arch:
        if '1x1' in arch:
            return 'conv1x1'
        elif '3x3' in arch:
            return 'conv3x3'
    elif 'pool' in arch:
        return 'pool'
    elif 'none' in arch:
        return 'none'
    else:
        return 'skip'

for i in range(6):
    df[f'edge{i+1}'] = df['arch'].apply(lambda x: extract_arch(x, i))
# print(df['edge2'])

top20_df = df.nlargest(20, 'hcs_ece_beta')

fig, axs = plt.subplots(1, 6, figsize=(17,5), sharey=True)
cbar_ax = fig.add_axes([.89, .15, .01, 0.78])

architectures = ['none','skip','pool','conv1x1','conv3x3']
colors = mpl.cm.viridis(np.linspace(0, 1, 20))

for i, ax in enumerate(axs.flat):
    edge_name = f'edge{i+1}'
    counter = 0
    for _, row in top20_df.iterrows():
        architecture = row[edge_name]
        ax.scatter(architectures.index(architecture), row['hcs_ece_beta'], color=colors[counter])
        counter += 1
    ax.set_title(f'Edge {i+1}')
    if i == 0:
        ax.set_ylabel('HCS', fontsize=16)
    # ax.set_xlabel('Architecture')
    for j, _ in enumerate(architectures):
        ax.axvline(j, linestyle='--', color='gray', alpha=0.5)
    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels(architectures, rotation=45)  # 旋转x轴标签


archs = top20_df['config'].tolist()
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.Normalize(vmin=0, vmax=len(archs)-1)
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, ticks=range(len(archs)), label='Top-20 Architecture on Cifar-10')
cb.ax.set_yticklabels(archs)

plt.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('./cifar10_top20.pdf', dpi=1000)
plt.show()

