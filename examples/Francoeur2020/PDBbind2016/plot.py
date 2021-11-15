import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

for variable in ["loss", "rmse"]:
    plt.figure()

    df = pd.read_csv(
        f"results/{variable}.csv", header=None, names=["epoch", "test", "train"]
    )
    df = df.melt(
        id_vars="epoch",
        value_vars=["test", "train"],
        var_name="mode",
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

df = pd.read_csv("out/inference.csv")
plt.figure()
sns.regplot(data=df, x="affinity_exp", y="affinity_pred")
plt.savefig("results/inference.png")
