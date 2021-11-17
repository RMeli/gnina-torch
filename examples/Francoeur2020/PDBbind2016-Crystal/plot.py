from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

nfolds = 6

folds = [f"fold{i}" for i in range(1, nfolds + 1)]

for variable in ["loss", "rmse"]:
    plt.figure()

    df_test = pd.read_csv(
        f"results/test-{variable}.csv",
        header=None,
        names=["epoch"] + folds,
    )
    df_test["mode"] = "test"

    df_train = pd.read_csv(
        f"results/train-{variable}.csv",
        header=None,
        names=["epoch"] + folds,
    )
    df_train["mode"] = "train"

    df = pd.concat((df_test, df_train))

    df = df.melt(
        id_vars=["epoch", "mode"],
        value_vars=folds,
        var_name="fold",
        value_name=variable,
    )

    sns.lineplot(
        data=df,
        x="epoch",
        y=variable,
        hue="mode",
        style="mode",
        markers=True,
        dashes=False,
    )
    plt.savefig(f"results/{variable}.png")

pearson = defaultdict(list)
rmse = defaultdict(list)

df_types = pd.read_csv(
    "../data/types/ref_uff_test0.types",
    header=None,
    sep=" ",
    names=["label", "affinity", "rmsd", "rec", "lig", "#", "vina"],
)
df_types["system"] = df_types.apply(lambda r: r["rec"][:4], axis=1)
df_types.drop(columns=["label", "affinity", "lig", "rec", "#"], inplace=True)

for fold in range(1, nfolds + 1):
    df_inference = pd.read_csv(f"out{fold}/inference.csv", index_col=0)
    df_inference["fold"] = fold

    df = pd.concat((df_inference, df_types), axis=1)
    print(df)

    # assert np.allclose( df_inference["affinity_exp"].to_numpy(), np.abs(df_types["affinity"].to_numpy()))

    ids_best = df.groupby(by=["system"])["rmsd"].idxmin()
    ids_worst = df.groupby(by=["system"])["rmsd"].idxmax()
    ids_vina = df.groupby(by=["system"])["vina"].idxmin()
    ids_cnnaffinity = df.groupby(by=["system"])["affinity_pred"].idxmax()

    pearson["best"].append(
        stats.pearsonr(
            df.loc[ids_best, "affinity_pred"], df.loc[ids_best, "affinity_exp"]
        )[0]
    )
    pearson["worst"].append(
        stats.pearsonr(
            df.loc[ids_worst, "affinity_pred"], df.loc[ids_worst, "affinity_exp"]
        )[0]
    )
    pearson["CNNaffinity"].append(
        stats.pearsonr(
            df.loc[ids_cnnaffinity, "affinity_pred"],
            df.loc[ids_cnnaffinity, "affinity_exp"],
        )[0]
    )

    def frmse(a, b):
        return np.sqrt(np.mean(np.square(a - b)))

    rmse["best"].append(
        frmse(df.loc[ids_best, "affinity_pred"], df.loc[ids_best, "affinity_exp"])
    )
    rmse["worst"].append(
        frmse(df.loc[ids_worst, "affinity_pred"], df.loc[ids_worst, "affinity_exp"])
    )
    rmse["CNNaffinity"].append(
        frmse(
            df.loc[ids_cnnaffinity, "affinity_pred"],
            df.loc[ids_cnnaffinity, "affinity_exp"],
        )
    )

df_pearson = pd.DataFrame(pearson)
df_rmse = pd.DataFrame(rmse)

order = ["best", "CNNaffinity", "worst"]

plt.figure()
sns.boxplot(data=df_pearson, order=order)
sns.swarmplot(data=df_pearson, order=order)
plt.xlabel("Selection Criteria")
plt.ylabel("Pearson's $r$")
plt.ylim(0, 0.8)
plt.savefig("results/pearson-box.png")

plt.figure()
sns.boxplot(data=df_rmse, order=order)
sns.swarmplot(data=df_rmse, order=order)
plt.xlabel("Selection Criteria")
plt.ylabel("RMSE")
plt.ylim(1.3, 3.0)
plt.savefig("results/rmse-box.png")
