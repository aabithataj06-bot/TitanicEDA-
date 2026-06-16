import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "#0a0e1a",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1e2d45",
    "axes.labelcolor": "#94a3b8",
    "axes.titlecolor": "#e2e8f0",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1e2d45",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "text.color": "#e2e8f0",
    "font.family": "monospace",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

COLORS = {
    "blue":   "#3b82f6",
    "cyan":   "#06b6d4",
    "purple": "#8b5cf6",
    "green":  "#10b981",
    "red":    "#ef4444",
    "amber":  "#f59e0b",
    "pink":   "#ec4899",
}
PALETTE = list(COLORS.values())

print("✅ Imports done. Libraries loaded.")
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

print(f"Shape       : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print("=" * 55)
print("  DESCRIPTIVE STATISTICS")
print("=" * 55)
print(df.describe().round(2).to_string())

print("\n\n" + "=" * 55)
print("  MISSING VALUES")
print("=" * 55)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "% Missing": missing_pct})
print(missing_df[missing_df["Missing"] > 0].sort_values("% Missing", ascending=False))

print("\n\n" + "=" * 55)
print("  SURVIVAL RATE")
print("=" * 55)
print(df["Survived"].value_counts())
print(f"\nOverall Survival Rate: {df['Survived'].mean()*100:.2f}%")

df["AgeBin"] = pd.cut(df["Age"],
                       bins=[0, 12, 18, 35, 60, 100],
                       labels=["Child", "Teen", "Adult", "Middle", "Senior"])


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


df["IsAlone"] = (df["FamilySize"] == 1).astype(int)


df["LogFare"] = np.log1p(df["Fare"])

print("✅ Feature engineering done.")
print(df[["AgeBin", "FamilySize", "IsAlone", "LogFare"]].head(8))



fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Titanic EDA  ·  Feature Distributions", fontsize=14, fontweight="bold", y=1.01)


ax = axes[0, 0]
counts = df["Pclass"].value_counts().sort_index()
bars = ax.bar(["1st Class", "2nd Class", "3rd Class"], counts,
              color=[COLORS["blue"], COLORS["cyan"], COLORS["purple"]],
              width=0.55, zorder=3)
ax.bar_label(bars, padding=4, color="#e2e8f0", fontsize=9)
ax.set_title("Passenger Class Distribution")
ax.set_ylabel("Count")
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)

ax = axes[0, 1]
sex_counts = df["Sex"].value_counts()
wedges, texts, autotexts = ax.pie(
    sex_counts, labels=["Male", "Female"],
    colors=[COLORS["blue"], COLORS["pink"]],
    autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(width=0.6, edgecolor="#0a0e1a", linewidth=2)
)
for at in autotexts: at.set_color("#e2e8f0")
ax.set_title("Gender Split")


ax = axes[0, 2]
ax.hist(df["Age"].dropna(), bins=25, color=COLORS["cyan"], alpha=0.85, edgecolor="#0a0e1a", zorder=3)
ax.axvline(df["Age"].median(), color=COLORS["amber"], linestyle="--", lw=1.5, label=f"Median: {df['Age'].median():.1f}")
ax.axvline(df["Age"].mean(),   color=COLORS["red"],   linestyle="--", lw=1.5, label=f"Mean:   {df['Age'].mean():.1f}")
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.legend(fontsize=8)
ax.grid(axis="y", zorder=0)


ax = axes[1, 0]
ax.hist(df["LogFare"], bins=30, color=COLORS["green"], alpha=0.85, edgecolor="#0a0e1a", zorder=3)
ax.set_title("Log(Fare) Distribution")
ax.set_xlabel("log(Fare + 1)")
ax.grid(axis="y", zorder=0)


ax = axes[1, 1]
fam_counts = df["FamilySize"].value_counts().sort_index()
ax.bar(fam_counts.index, fam_counts.values, color=COLORS["purple"], alpha=0.9,
       width=0.6, zorder=3, edgecolor="#0a0e1a")
ax.set_title("Family Size Distribution")
ax.set_xlabel("Family Size")
ax.set_ylabel("Count")
ax.grid(axis="y", zorder=0)


ax = axes[1, 2]
emb = df["Embarked"].value_counts()
ax.bar(emb.index, emb.values,
       color=[COLORS["blue"], COLORS["cyan"], COLORS["purple"]],
       width=0.5, zorder=3)
ax.set_title("Embarkation Port")
ax.set_ylabel("Count")
ax.grid(axis="y", zorder=0)

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.show()
print("✅ Saved: eda_distributions.png")



fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Titanic EDA  ·  Survival Analysis", fontsize=14, fontweight="bold")

surv_colors = [COLORS["red"], COLORS["green"]]


ax = axes[0, 0]
surv_class = df.groupby("Pclass")["Survived"].mean()
bars = ax.bar(["1st Class", "2nd Class", "3rd Class"], surv_class * 100,
              color=[COLORS["green"], COLORS["amber"], COLORS["red"]],
              width=0.55, zorder=3)
ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#e2e8f0", fontsize=9)
ax.set_title("Survival Rate by Class")
ax.set_ylabel("Survival %")
ax.set_ylim(0, 80)
ax.grid(axis="y", zorder=0)


ax = axes[0, 1]
surv_sex = df.groupby("Sex")["Survived"].mean()
bars = ax.bar(["Female", "Male"], surv_sex[["female", "male"]] * 100,
              color=[COLORS["pink"], COLORS["blue"]], width=0.4, zorder=3)
ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#e2e8f0", fontsize=9)
ax.set_title("Survival Rate by Gender")
ax.set_ylabel("Survival %")
ax.set_ylim(0, 90)
ax.grid(axis="y", zorder=0)


ax = axes[0, 2]
surv_age = df.groupby("AgeBin", observed=True)["Survived"].mean()
bars = ax.bar(surv_age.index, surv_age * 100,
              color=PALETTE[:len(surv_age)], width=0.55, zorder=3)
ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#e2e8f0", fontsize=9)
ax.set_title("Survival Rate by Age Group")
ax.set_ylabel("Survival %")
ax.set_ylim(0, 75)
ax.grid(axis="y", zorder=0)


ax = axes[1, 0]
survived_fare   = df[df["Survived"] == 1]["LogFare"]
nosurvive_fare  = df[df["Survived"] == 0]["LogFare"]
bp = ax.boxplot([nosurvive_fare, survived_fare],
                patch_artist=True,
                medianprops=dict(color="#fff", linewidth=2))
bp["boxes"][0].set_facecolor(COLORS["red"] + "99")
bp["boxes"][1].set_facecolor(COLORS["green"] + "99")
ax.set_xticklabels(["Not Survived", "Survived"])
ax.set_title("log(Fare) by Survival")
ax.set_ylabel("log(Fare + 1)")
ax.grid(axis="y", zorder=0)


ax = axes[1, 1]
surv_emb = df.groupby("Embarked")["Survived"].mean().sort_values(ascending=False)
bars = ax.bar(surv_emb.index, surv_emb * 100,
              color=[COLORS["cyan"], COLORS["purple"], COLORS["blue"]], width=0.45, zorder=3)
ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#e2e8f0", fontsize=9)
ax.set_title("Survival Rate by Port")
ax.set_ylabel("Survival %")
ax.set_ylim(0, 70)
ax.grid(axis="y", zorder=0)


ax = axes[1, 2]
surv_fam = df.groupby("FamilySize")["Survived"].mean()
ax.plot(surv_fam.index, surv_fam * 100,
        color=COLORS["cyan"], marker="o", lw=2, markersize=7, zorder=3)
ax.fill_between(surv_fam.index, surv_fam * 100, alpha=0.15, color=COLORS["cyan"])
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family Size")
ax.set_ylabel("Survival %")
ax.grid(zorder=0)

plt.tight_layout()
plt.savefig("eda_survival.png", dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.show()
print("✅ Saved: eda_survival.png")



fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Titanic EDA  ·  Correlations", fontsize=14, fontweight="bold")


numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "LogFare"]
corr = df[numeric_cols].corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

ax = axes[0]
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor="#0a0e1a",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8, "color": "white"})
ax.set_title("Correlation Heatmap (Lower Triangle)")
ax.set_facecolor("#111827")
ax.tick_params(colors="#94a3b8", labelsize=9)


ax = axes[1]
surv_corr = corr["Survived"].drop("Survived").sort_values()
bar_colors = [COLORS["red"] if v < 0 else COLORS["green"] for v in surv_corr]
bars = ax.barh(surv_corr.index, surv_corr.values, color=bar_colors,
               alpha=0.85, zorder=3, edgecolor="#0a0e1a")
ax.axvline(0, color="#64748b", lw=1)
ax.set_title("Feature Correlation with Survival")
ax.set_xlabel("Pearson Correlation")
ax.grid(axis="x", zorder=0)
for bar, val in zip(bars, surv_corr.values):
    ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
            fontsize=8, color="#e2e8f0")

plt.tight_layout()
plt.savefig("eda_correlations.png", dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.show()
print("✅ Saved: eda_correlations.png")



print("\n" + "=" * 55)
print("  SURVIVAL RATE: CLASS × GENDER")
print("=" * 55)
pivot = df.pivot_table("Survived", index="Sex", columns="Pclass", aggfunc="mean")
print((pivot * 100).round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0a0e1a")
pivot_plot = (df.groupby(["Pclass", "Sex"])["Survived"].mean() * 100).unstack()
pivot_plot.plot(kind="bar", ax=ax,
                color=[COLORS["pink"], COLORS["blue"]],
                width=0.6, zorder=3, edgecolor="#0a0e1a")
ax.set_title("Survival Rate by Class & Gender")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival %")
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
ax.legend(["Female", "Male"], framealpha=0.2)
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("eda_class_gender.png", dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.show()
print("✅ Saved: eda_class_gender.png")



print("\n" + "=" * 55)
print("  KEY FINDINGS SUMMARY")
print("=" * 55)

findings = [
    ("Gender",      "Female 74.2% vs Male 18.9% survival — biggest factor"),
    ("Class",       "1st 63% | 2nd 47% | 3rd 24% — wealth = lifeboat access"),
    ("Age",         "Children 58% survival (highest), Seniors 22.7% (lowest)"),
    ("Fare",        "Survivors paid 2.2x more ($48 vs $22 avg)"),
    ("Port",        "Cherbourg 55.4% — mostly 1st class boardings"),
    ("FamilySize",  "Solo travelers & very large families survived least"),
    ("Missing",     "Cabin 77% missing, Age 20% — needs imputation"),
]

for feature, insight in findings:
    print(f"  {'▶ ' + feature:<14} {insight}")

print("\n📁 Files saved:")
for f in ["eda_distributions.png", "eda_survival.png",
          "eda_correlations.png", "eda_class_gender.png"]:
    print(f"   · {f}")
