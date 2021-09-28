import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import os

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-f", "--file",
        metavar="F",
        default="result.csv",
        help="The Result file"
    )
    argparser.add_argument(
        "-d", "--output_dir",
        metavar="D",
        help="Output Directory"
    )
    subparsers = argparser.add_subparsers(dest='subcommand')
    # subparser for model dims
    parser_dims = subparsers.add_parser('dims')
    parser_dims.add_argument(
        "-n", "--num_stacks",
        default=6,
        help="Number of stacks"
    )

    # subparser for num_stacks
    parser_stacks = subparsers.add_parser("stacks")
    parser_stacks.add_argument(
        "-a", "--autoencoder_dims",
        default=200,
        help="Autoencoder dims"
    )
    parser_stacks.add_argument(
        "-l", "--l_win",
        default=100,
        help="Transformer input dims"
    )
    args = argparser.parse_args()
    return args

args = get_args()

RESULT_FILE=args.file
OUTPUT_DIR=args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
result = pd.read_csv(RESULT_FILE, index_col=0)
color_list = ["tab:red", "tab:green", "tab:purple", "tab:blue", "tab:orange", "tab:brown", "tab:pink", "tab:cyan", "tab:olive", "tab:gray"]

def visualize(args):
    type = args.subcommand
    if type == "stacks":
        return visualize_stack(args.autoencoder_dims, args.l_win)
    if type == "dims":
        return visualize_dims(args.num_stacks)

def visualize_stack(auto, trans):
    out_df = result[(result["autoencoder_dims"] == int(auto)) & (result["l_win"] == int(trans))]
    print(out_df)
    for data in ["scada1", "scada2"]:
        out_df_scada = out_df[out_df["data_dir"] == data]
        out_df_scada = out_df_scada.sort_values(by=["num_stacks"])
        output_file_scada = os.path.join(OUTPUT_DIR, "auto_{}_trans_{}_{}.png".format(auto, trans, data))
        plt.figure()
        l = list(out_df_scada["num_stacks"].to_numpy())
        plt.plot(l, out_df_scada["precision"].to_numpy(), color_list[0], marker=".", label="precision")
        plt.plot(l, out_df_scada["recall"].to_numpy(), color_list[1], marker=".", label="recall")
        plt.plot(l, out_df_scada["F1"].to_numpy(), color_list[2], marker=".", label="f1")
        plt.plot(l, out_df_scada["AUC"].to_numpy(), color_list[3], marker=".", label="auc")
        plt.ylim(0., 1.)
        plt.xlim(1, 6)
        plt.legend()
        plt.title("Performance by number of transformer stacks ({},{})".format(auto, trans))
        plt.savefig(output_file_scada)
        plt.close()
    return 0

def visualize_dims(num_stacks):
    out_df = result[result["num_stacks"] == int(num_stacks)]
    out_df = out_df.sort_values(by=["autoencoder_dims"])
    for data in ["scada1", "scada2"]:
        out_df_scada = out_df[out_df["data_dir"] == data]
        OUTPUT_FILE_SCADA = os.path.join(OUTPUT_DIR, "stacks_{}_{}.png".format(num_stacks, data))
        fig, ax = plt.subplots()
        tup = out_df_scada[["autoencoder_dims", "l_win"]].itertuples()
        l = []
        for t in tup:
            l.append("({},{})".format(t[1], t[2]))
        plt.plot(l, out_df_scada["precision"].to_numpy(), color_list[0], marker=".", label="precision")
        plt.plot(l, out_df_scada["recall"].to_numpy(), color_list[1], marker=".", label="recall")
        plt.plot(l, out_df_scada["F1"].to_numpy(), color_list[2], marker=".", label="F1")
        plt.plot(l, out_df_scada["AUC"].to_numpy(), color_list[3], marker=".", label="AUC")
        plt.legend()
        plt.ylim(0., 1.)
        ax.set_xticklabels(rotation = (45), fontsize = 8, va='bottom', ha='left', labels=l)
        plt.title("Performance by model dims (stacks = {})".format(num_stacks))
        fig.savefig(OUTPUT_FILE_SCADA)
        plt.close(fig)
visualize(args)
