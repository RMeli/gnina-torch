import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

epochs = 50

df_all = pd.DataFrame(columns=["Fold", "ROC-AUC", "Phase", "Augmentation"])

for a in ["augmentation", "no-augmentation"]:
    for p in ["train", "test"]:
        df = pd.read_csv(f"results/{a}-{p}.csv", header=None)

        df["Epoch"] = np.arange(0, epochs, 1)
        df["Phase"] = p
        df["Augmentation"] = "yes" if a == "augmentation" else "no"

        df = df.melt(
            id_vars=["Epoch", "Phase", "Augmentation"], value_vars=[0, 1, 2]
        ).rename(columns={"variable": "Fold", "value": "ROC-AUC"})

        df_all = df_all.append(df, ignore_index=True)

# print(df_all)

sns.lineplot(x="Epoch", y="ROC-AUC", hue="Augmentation", style="Phase", data=df_all)
plt.savefig("ROC-AUC.png")
plt.savefig("ROC-AUC.pdf")
