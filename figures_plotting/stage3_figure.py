#!/usr/bin/env python3
"""Stage 3 figure: injection<->tumour aiming alignment, train vs test, image vs scalar."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
SP = os.path.dirname(os.path.abspath(__file__))
res = pd.read_csv(f"{SP}/stage3_alignment.csv")

order = ["I1  (img cells)", "I2  (img cells+subs)", "I2m (img cells+subs+m1m2)",
         "S3m (scalar cells+m1m2)", "S3s (scalar cells+subs)", "S3sm(scalar cells+subs+m1m2)"]
res["mode"] = pd.Categorical(res["mode"], order, ordered=True)
res = res.sort_values("mode")

fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(len(order))
w = 0.36
tr = [res[(res["mode"] == m) & (res.split == "train")].align_mean.values[0] for m in order]
te = [res[(res["mode"] == m) & (res.split == "test")].align_mean.values[0] for m in order]
tr_sd = [res[(res["mode"] == m) & (res.split == "train")].align_sd.values[0] for m in order]
te_sd = [res[(res["mode"] == m) & (res.split == "test")].align_sd.values[0] for m in order]

ax.bar(x - w/2, tr, w, yerr=tr_sd, capsize=3, label="train (rectangle)", color="#4C72B0")
ax.bar(x + w/2, te, w, yerr=te_sd, capsize=3, label="test (network-field, OOD)", color="#C44E52")
ax.axvline(2.5, color="k", ls=":", lw=0.8)
ax.text(1, 0.03, "image obs", ha="center", fontsize=9, style="italic")
ax.text(4, 0.03, "scalar obs", ha="center", fontsize=9, style="italic")
ax.set_xticks(x)
ax.set_xticklabels([m.split("(")[0].strip() for m in order], rotation=0)
ax.set_ylabel("injection–tumour distance\n(normalised, lower = better aiming)")
ax.set_title("Drug-aiming alignment transfers under geometry shift for image, not scalar, observations")
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, 0.8)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{SP}/fig_aiming_transfer.{ext}", dpi=150)
print("saved fig_aiming_transfer.pdf/.png")
