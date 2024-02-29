import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from robustness_dataset import RobustnessDataset
from matplotlib.cm import ScalarMappable

# Load the data into a pandas DataFrame
df = pd.read_csv("../cifar10_tss.csv")
# df.sort_values(by='accuracy', ascending=False, inplace=True)
# df.reset_index()
# df = df.head(1500)
print(df)

data = RobustnessDataset(path="robustness-dataset")
no_paras = []
for idx in df.index:
    arch = data.id_to_string(int(idx))
    # print(arch)
    edge = [arch.split('+')[0][1:-1], arch.split('+')[1][1:-1].split("|")[0], arch.split('+')[1][1:-1].split("|")[1],
            arch.split('+')[2][1:-1].split("|")[0], arch.split('+')[2][1:-1].split("|")[1],
            arch.split('+')[2][1:-1].split("|")[2]]
    temp_paras = 0
    for item in edge:
        if item.startswith("nor_conv_3x3"):
            temp_paras += 9
        elif item.startswith("nor_conv_1x1"):
            temp_paras += 1
    no_paras.append(temp_paras)
    # print(edge)
df["no_paras"] = no_paras
acc = df["acc"]
ece = df["pre_ECE_15"]
# adaece = df["15bins-adaece-pre"]
# cwece = df["15bins-classece-pre"]
hrs_ece = 2 * (acc * (1 - ece))/(acc + (1-ece))
# hrs_adece = 2 * (acc * (1 - adaece))/(acc + (1 - adaece))
# hrs_cwece = 2 * (acc * (1 - cwece))/(acc + (1 - cwece))
df["hrs_ece"] = hrs_ece
# df["hrs_adaece"] = hrs_adece
# df["hrs_cwece"] = hrs_cwece

beta = 1
hrs_ece_beta = (1 + beta) * (acc * (1 - ece))/(beta * acc + (1-ece))
# hrs_adece_beta = (1 + beta) * (acc * (1 - adaece))/(beta * acc + (1 - adaece))
# hrs_cwece_beta = (1 + beta) * (acc * (1 - cwece))/(beta * acc + (1 - cwece))
df["hrs_ece_beta"] = hrs_ece_beta
# df["hrs_adaece_beta"] = hrs_adece_beta
# df["hrs_cwece_beta"] = hrs_cwece_beta

# df.sort_values(by = 'hrs_ece_beta', ascending=False, inplace=True)
# df.reset_index()
# print(df[["hrs_ece_beta", '15bins-ece-pre', 'accuracy']])


# Set the seaborn style and font scale
sns.set(style="whitegrid")
sns.set(font_scale=1.4)
min_no_paras, max_no_paras = df['no_paras'].min(), df['no_paras'].max()
# Create a color palette based on reds
red_map = sns.color_palette("Blues_r", as_cmap=True)

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Scatter plot 1
sns.scatterplot(x="no_paras", y="pre_ECE_15", data=df, ax=ax[0], linewidth=0,
                palette=red_map)
ax[0].set_xlabel("Cell Kernel Parameters")
ax[0].set_ylabel("ECE(%)")


# Scatter plot 2
scatter_plot = sns.scatterplot(y="pre_ECE_15", x="acc", hue="no_paras", data=df,
                ax=ax[1], palette=red_map, alpha=0.4, edgecolor='face', legend=False)
ax[1].set_xlabel("Accuracy(%)")
ax[1].set_ylabel("ECE(%)")
scatter_plot.set_xlim(0.4, 1)

# Scatter plot 3
scatter_plot = sns.scatterplot(y="hrs_ece_beta", x="acc", hue="no_paras", data=df,
                ax=ax[2], palette=red_map, alpha=0.4, edgecolor='face', legend=False)
ax[2].set_xlabel("Accuracy(%)")
ax[2].set_ylabel("HCS")
scatter_plot.set_ylim(0.55, 1)
scatter_plot.set_xlim(0.4, 1)

# Add color bar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(ScalarMappable(cmap=red_map, norm=plt.Normalize(vmin=min_no_paras, vmax=max_no_paras)), cax=cbar_ax)
cbar.set_label("Cell Kernel Parameters")

formatter = ticker.FuncFormatter(lambda x, _: f'{x*100:.0f}')
tick_fontsize = 12

for i, axes in enumerate(ax):
    axes.xaxis.set_major_formatter(formatter)
    if axes != ax[-1]:
        axes.yaxis.set_major_formatter(formatter)
    axes.tick_params(axis='both', labelsize=tick_fontsize)


plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the layout to make room for the color bar
plt.savefig('./cell_params.pdf')
plt.show()

