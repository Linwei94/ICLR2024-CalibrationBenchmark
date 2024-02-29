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





# print(no_paras)
#
# # Calculate the average pre AUC for CIFAR10-C and SVHN
# df["average_pre_auc"] = (df["pre_auc_cifar10c"] + df["pre_auc_svhn"]) / 2
#
# # Plot the scatter plots
# # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# red_map = sns.color_palette("Blues", as_cmap=True)
# sns.scatterplot(x="no_paras", y="15bins-ece-pre" ,data=df, ax=ax[0], linewidth=0,
#                 palette=red_map, alpha=0.3)
# ax[0].set_xlabel("Cell Kernel Parameters")
# ax[0].set_ylabel("ECE")
# ax[0].set_title("Cell Kernel Parameters vs. ECE")
#
# # sns.scatterplot(x="no_paras", y="15bins-adaece-pre", hue="average_pre_auc", data=df, ax=ax[1], linewidth=0,
# #                 palette=red_map, alpha=0.3)
# # ax[1].set_xlabel("Cell Kernel Parameters")
# # ax[1].set_ylabel("ada-ECE")
# # ax[1].set_title("15bins-adaece-pre vs. ada-ECE")
# #
# # sns.scatterplot(x="no_paras", y="15bins-classece-pre", hue="average_pre_auc", data=df, ax=ax[2], linewidth=0,
# #                 palette=red_map, alpha=0.3)
# # ax[2].set_xlabel("Cell Kernel Parameters")
# # ax[2].set_ylabel("class-wise ECE")
# # ax[2].set_title("Cell Kernel Parameters vs. class-wise ECE")
# #
# # plt.tight_layout()
# # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# # plt.savefig('./no_paras_ece_top1500ece.png')
# # plt.show()
#
#
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# inverted_flare_map = sns.color_palette("flare_r", as_cmap=True)
#
# # create a ScalarMappable object for the color map with min and max values from "no_paras"
# min_no_paras, max_no_paras = df['no_paras'].min(), df['no_paras'].max()
# sm = ScalarMappable(cmap=inverted_flare_map, norm=plt.Normalize(vmin=min_no_paras, vmax=max_no_paras))
#
#
# # Scatter plot 1
# scatter_plot = sns.scatterplot(y="hrs_ece_beta", x="acc", hue="no_paras", data=df,
#                 ax=ax[0], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[0].set_xlabel("accuracy")
# ax[0].set_ylabel("hrs_ece")
# ax[0].set_title("hrs_ece vs. accuracy")
# ax[0].grid(True, linestyle='--')   # Enable gridlines for the first scatter plot
#
# # Add color bar for the first scatter plot
# fig.colorbar(sm, ax=ax[0])
# scatter_plot.set_xlim(0.4, 1)
# scatter_plot.set_ylim(0.6, 1)
#
# # Scatter plot 2
# scatter_plot = sns.scatterplot(y="hrs_adaece_beta", x="acc", hue="no_paras", data=df,
#                 ax=ax[1], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[1].set_xlabel("accuracy")
# ax[1].set_ylabel("hrs_adaece")
# ax[1].set_title("hrs_adaece vs. accuracy")
# ax[1].grid(True, linestyle='--')  # Enable gridlines for the second scatter plot
#
# # Add color bar for the second scatter plot
# fig.colorbar(sm, ax=ax[1])
# scatter_plot.set_xlim(0.4, 1)
# scatter_plot.set_ylim(0.6, 1)
#
# # Scatter plot 3
# scatter_plot = sns.scatterplot(y="hrs_cwece_beta", x="acc", hue="no_paras", data=df,
#                 ax=ax[2], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[2].set_xlabel("accuracy")
# ax[2].set_ylabel("hrs_cwece")
# ax[2].set_title("hrs_cwece ECE vs. accuracy")
# ax[2].grid(True, linestyle='--')  # Enable gridlines for the third scatter plot
#
# # Add color bar for the third scatter plot
# fig.colorbar(sm, ax=ax[2])
# scatter_plot.set_xlim(0.4, 1)
# scatter_plot.set_ylim(0.6, 1)
#
# plt.tight_layout()
# # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# # plt.savefig('./hrs_acc_params.png')
# plt.show()
#
#
#
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# inverted_flare_map = sns.color_palette("flare_r", as_cmap=True)
#
# # create a ScalarMappable object for the color map with min and max values from "no_paras"
# min_no_paras, max_no_paras = df['no_paras'].min(), df['no_paras'].max()
# sm = ScalarMappable(cmap=inverted_flare_map, norm=plt.Normalize(vmin=min_no_paras, vmax=max_no_paras))
#
# # Scatter plot 1
# scatter_plot = sns.scatterplot(y="15bins-ece-pre", x="acc", hue="no_paras", data=df,
#                 ax=ax[0], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[0].set_xlabel("accuracy")
# ax[0].set_ylabel("ECE")
# ax[0].set_title("ECE vs. accuracy")
# ax[0].grid(True, linestyle='--')   # Enable gridlines for the first scatter plot
#
# # Add color bar for the first scatter plot
# fig.colorbar(sm, ax=ax[0])
# scatter_plot.set_xlim(0.4, 1)
#
#
# # Scatter plot 2
# scatter_plot = sns.scatterplot(y="15bins-adaece-pre", x="acc", hue="no_paras", data=df,
#                 ax=ax[1], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[1].set_xlabel("accuracy")
# ax[1].set_ylabel("ada-ECE")
# ax[1].set_title("ada-ECE vs. accuracy")
# ax[1].grid(True, linestyle='--')  # Enable gridlines for the second scatter plot
#
# # Add color bar for the second scatter plot
# fig.colorbar(sm, ax=ax[1])
# scatter_plot.set_xlim(0.4, 1)
#
# # Scatter plot 3
# scatter_plot = sns.scatterplot(y="15bins-classece-pre", x="acc", hue="no_paras", data=df,
#                 ax=ax[2], palette=inverted_flare_map, alpha=0.4, edgecolor='face', legend=False)
# ax[2].set_xlabel("accuracy")
# ax[2].set_ylabel("cw-ECE")
# ax[2].set_title("class-wise ECE vs. accuracy")
# ax[2].grid(True, linestyle='--')  # Enable gridlines for the third scatter plot
#
# # Add color bar for the third scatter plot
# fig.colorbar(sm, ax=ax[2])
# scatter_plot.set_xlim(0.4, 1)
#
# plt.tight_layout()
# # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# # plt.savefig('./ece_acc_with_params_auroc.png')
# plt.show()


#
# df.sort_values(by='average_pre_auc', ascending=False, inplace=True)
# df.reset_index()
# df = df.head(1200)
# print(df)
#
# # Plot the scatter plots
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# red_map = sns.color_palette("Blues", as_cmap=True)
# sns.scatterplot(x="no_paras", y="average_pre_auc", hue="15bins-ece-pre", linewidth=0, data=df, ax=ax[0], palette=red_map)
# ax[0].set_xlabel("Cell Kernel Parameters")
# ax[0].set_ylabel("average_pre_auc")
# ax[0].set_title("Cell Kernel Parameters vs. average_pre_auc")
#
# sns.scatterplot(x="no_paras", y="average_pre_auc", hue="15bins-adaece-pre", data=df, ax=ax[1], linewidth=0, palette=red_map)
# ax[1].set_xlabel("Cell Kernel Parameters")
# ax[1].set_ylabel("average_pre_auc")
# ax[1].set_title("15bins-adaece-pre vs. average_pre_auc")
#
# sns.scatterplot(x="no_paras", y="average_pre_auc", hue="15bins-classece-pre", data=df, ax=ax[2], linewidth=0, palette=red_map)
# ax[2].set_xlabel("Cell Kernel Parameters")
# ax[2].set_ylabel("average_pre_auc")
# ax[2].set_title("Cell Kernel Parameters vs. average_pre_auc")
#
# plt.tight_layout()
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# # plt.savefig('./no_paras_aucroc.png')
# plt.show()


# fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=800)
# red_map = sns.color_palette("Blues", as_cmap=True)
# scatter_plot = sns.scatterplot( y="15bins-ece-pre", x="accuracy", hue="no_paras", data=df, ax=ax[0], linewidth=0)
# ax[0].set_xlabel("accuracy")
# ax[0].set_ylabel("ECE")
# ax[0].set_title("ECE vs. accuracy")
# scatter_plot.set_ylim(40, 100)
#
# scatter_plot = sns.scatterplot(y="15bins-adaece-pre", x="accuracy", hue="no_paras", size="no_paras", data=df,
#                  ax=ax[1], linewidth=0)
# ax[1].set_xlabel("accuracy")
# ax[1].set_ylabel("ada-ECE")
# ax[1].set_title("ada-ECE vs. accuracy")
# scatter_plot.set_ylim(40, 100)
#
# scatter_plot = sns.scatterplot(y="15bins-classece-pre", x="accuracy", hue="no_paras", data=df,
#                  ax=ax[2], linewidth=0)
# ax[2].set_xlabel("accuracy")
# ax[2].set_ylabel("ada-ECE")
# ax[2].set_title("class-wise ECE vs. accuracy")
# scatter_plot.set_ylim(40, 100)
#
# plt.tight_layout()
# plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
# # plt.savefig('./ece_acc_with_params_auroc.png')
# # plt.show()