"""
Netflix Dataset - Exploratory Data Analysis
============================================
Run: python netflix_eda.py
Output: All charts saved to outputs/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/netflix_titles.csv"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "blue":   "#3266ad",
    "coral":  "#d85a30",
    "teal":   "#1D9E75",
    "amber":  "#BA7517",
    "purple": "#7F77DD",
    "pink":   "#D4537E",
    "gray":   "#888780",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        11,
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Load & clean ───────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")

df["date_added"]   = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df["year_added"]   = df["date_added"].dt.year
df["month_added"]  = df["date_added"].dt.month

movies = df[df["type"] == "Movie"].copy()
tv     = df[df["type"] == "TV Show"].copy()

movies["duration_min"] = movies["duration"].str.extract(r"(\d+)").astype(float)
tv["seasons"]          = tv["duration"].str.extract(r"(\d+)").astype(float)


# ── 2. Summary stats ──────────────────────────────────────────────────────────
print("\n[2] Summary statistics")
print(df.describe(include="all").to_string())

print("\nMissing values:")
print(df.isnull().sum())


# ── 3. Chart: Content type split ──────────────────────────────────────────────
print("\n[3] Plotting charts...")
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["type"].value_counts()
bars = ax.barh(counts.index, counts.values,
               color=[PALETTE["blue"], PALETTE["coral"]], height=0.5)
for bar, v in zip(bars, counts.values):
    ax.text(v + 30, bar.get_y() + bar.get_height()/2,
            f"{v:,}  ({v/len(df)*100:.1f}%)", va="center", fontsize=10)
ax.set_xlabel("Number of titles")
ax.set_title("Content type distribution", fontweight="bold")
ax.set_xlim(0, counts.max() * 1.22)
save(fig, "01_content_type.png")


# ── 4. Chart: Content added per year ─────────────────────────────────────────
yearly = df["year_added"].value_counts().sort_index()
yearly = yearly[yearly.index >= 2015]

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(yearly.index.astype(int), yearly.values, color=PALETTE["blue"], width=0.6)
for x, y in zip(yearly.index.astype(int), yearly.values):
    ax.text(x, y + 20, str(int(y)), ha="center", fontsize=9)
ax.set_xlabel("Year")
ax.set_ylabel("Titles added")
ax.set_title("Content added to Netflix per year", fontweight="bold")
save(fig, "02_content_per_year.png")


# ── 5. Chart: Movies vs TV added per year ────────────────────────────────────
m_year = movies["year_added"].value_counts().sort_index()
t_year = tv["year_added"].value_counts().sort_index()
years  = sorted(set(m_year.index) | set(t_year.index))
years  = [y for y in years if y >= 2015]

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(years))
w = 0.4
ax.bar(x - w/2, [m_year.get(y, 0) for y in years], w,
       label="Movies", color=PALETTE["blue"])
ax.bar(x + w/2, [t_year.get(y, 0) for y in years], w,
       label="TV Shows", color=PALETTE["coral"])
ax.set_xticks(x)
ax.set_xticklabels([int(y) for y in years])
ax.set_ylabel("Titles added")
ax.set_title("Movies vs TV Shows added per year", fontweight="bold")
ax.legend()
save(fig, "03_movies_vs_tv_yearly.png")


# ── 6. Chart: Top 10 countries ────────────────────────────────────────────────
top_countries = df["country"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1],
               color=PALETTE["teal"])
for bar, v in zip(bars, top_countries.values[::-1]):
    ax.text(v + 20, bar.get_y() + bar.get_height()/2,
            f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("Number of titles")
ax.set_title("Top 10 countries by content volume", fontweight="bold")
save(fig, "04_top_countries.png")


# ── 7. Chart: Content ratings ────────────────────────────────────────────────
ratings = df["rating"].value_counts().head(9)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(ratings.index, ratings.values, color=PALETTE["coral"])
for i, (r, v) in enumerate(zip(ratings.index, ratings.values)):
    ax.text(i, v + 30, str(v), ha="center", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Content ratings distribution", fontweight="bold")
save(fig, "05_ratings.png")


# ── 8. Chart: Top 15 genres ──────────────────────────────────────────────────
genres = Counter()
for g in df["listed_in"].dropna():
    for item in g.split(","):
        genres[item.strip()] += 1

top_genres = pd.Series(dict(genres.most_common(15)))

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=PALETTE["purple"])
for i, v in enumerate(top_genres.values[::-1]):
    ax.text(v + 10, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Number of titles")
ax.set_title("Top 15 genres on Netflix", fontweight="bold")
save(fig, "06_top_genres.png")


# ── 9. Chart: Movie duration distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(movies["duration_min"].dropna(), bins=30,
        color=PALETTE["pink"], edgecolor="white", linewidth=0.5)
ax.axvline(movies["duration_min"].mean(), color=PALETTE["coral"],
           linestyle="--", linewidth=1.5, label=f"Mean: {movies['duration_min'].mean():.1f} min")
ax.axvline(movies["duration_min"].median(), color=PALETTE["blue"],
           linestyle="--", linewidth=1.5, label=f"Median: {movies['duration_min'].median():.0f} min")
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Count")
ax.set_title("Movie duration distribution", fontweight="bold")
ax.legend()
save(fig, "07_movie_duration.png")


# ── 10. Chart: TV show seasons ───────────────────────────────────────────────
season_counts = tv["seasons"].value_counts().sort_index().head(10)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(season_counts.index.astype(int), season_counts.values,
       color=PALETTE["amber"], width=0.6)
for x, y in zip(season_counts.index.astype(int), season_counts.values):
    ax.text(x, y + 10, str(int(y)), ha="center", fontsize=9)
ax.set_xlabel("Number of seasons")
ax.set_ylabel("Number of TV shows")
ax.set_title("TV show season distribution", fontweight="bold")
save(fig, "08_tv_seasons.png")


# ── 11. Chart: Missing values heatmap ────────────────────────────────────────
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]

fig, ax = plt.subplots(figsize=(7, 3))
bars = ax.barh(missing.index[::-1], missing.values[::-1],
               color=[PALETTE["coral"] if v > 500 else PALETTE["amber"] for v in missing.values[::-1]])
for bar, v in zip(bars, missing.values[::-1]):
    ax.text(v + 5, bar.get_y() + bar.get_height()/2,
            f"{int(v):,}", va="center", fontsize=9)
ax.set_xlabel("Missing count")
ax.set_title("Missing values per column", fontweight="bold")
save(fig, "09_missing_values.png")


# ── 12. Chart: Monthly content additions ─────────────────────────────────────
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = df["month_added"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(monthly.index, monthly.values, marker="o",
        color=PALETTE["teal"], linewidth=2, markersize=6)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=PALETTE["teal"])
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_ylabel("Titles added")
ax.set_title("Content additions by month (all years)", fontweight="bold")
save(fig, "10_monthly_additions.png")


# ── 13. Correlation heatmap (numeric cols) ───────────────────────────────────
num_df = df[["release_year", "year_added", "month_added"]].dropna()
num_df["is_movie"] = (df["type"] == "Movie").astype(int)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation matrix (numeric features)", fontweight="bold")
save(fig, "11_correlation_heatmap.png")


print("\n✅ All 11 charts saved to outputs/")
print("\nKey insights:")
print(f"  • Total titles     : {len(df):,}")
print(f"  • Movies           : {len(movies):,} ({len(movies)/len(df)*100:.1f}%)")
print(f"  • TV Shows         : {len(tv):,} ({len(tv)/len(df)*100:.1f}%)")
print(f"  • Avg movie length : {movies['duration_min'].mean():.1f} min")
print(f"  • Top country      : {df['country'].value_counts().index[0]}")
print(f"  • Peak year added  : {int(yearly.idxmax())} ({int(yearly.max())} titles)")
print(f"  • Most common rating: {df['rating'].value_counts().index[0]}")
print(f"  • Missing directors: {df['director'].isnull().sum():,} ({df['director'].isnull().mean()*100:.1f}%)")
