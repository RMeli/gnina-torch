import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df_all = []

for a in ["augmentation", "no-augmentation"]:
    for p in ["train", "test"]:
        df = pd.read_csv(f"results/{a}-{p}.csv", header=None).rename(
            columns={0: "Epoch"}
        )

        df["Phase"] = p
        df["Augmentation"] = "yes" if a == "augmentation" else "no"

        df = df.melt(
            id_vars=["Epoch", "Phase", "Augmentation"], value_vars=[1, 2, 3]
        ).rename(columns={"variable": "Fold", "value": "ROC-AUC"})

        df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

sns.lineplot(
    x="Epoch", y="ROC-AUC", hue="Augmentation", style="Phase", data=df_all, markers=True
)
plt.legend(loc="lower center")
plt.savefig("results/ROC-AUC.png")
plt.savefig("results/ROC-AUC.pdf")
