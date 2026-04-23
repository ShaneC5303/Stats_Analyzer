#  Stats Analyzer
**Course:** 23DEA3202 – Data Exploration and Preparation  
**Project:** Abstract Cricket – Test Player Performance Analysis

---

##  Project Structure
```
cricket_project/
├── data/
│   └── cricket_players.csv   ← Dataset (22 international players)
├── src/
│   └── analysis.py           ← Main analysis script
├── outputs/                  ← Generated charts & CSVs (auto-created)
├── requirements.txt
└── README.md
```

---

##  Quick Start (VS Code)

### Step 1 – Install dependencies
Open the **VS Code Terminal** (`Ctrl + `` ` ``) and run:
```bash
pip install -r requirements.txt
```

### Step 2 – Run the analysis
```bash
python src/analysis.py
```

### Step 3 – View outputs
- `outputs/cricket_analysis.png` → 6 charts (role pie, runs by country, scatter, heatmap …)
- `outputs/player_rankings.csv` → Ranked players by composite performance score

---

##  What the Script Does

| Step | Description |
|------|-------------|
| 1. Load | Reads `cricket_players.csv` into a Pandas DataFrame |
| 2. Overview | Prints dtypes, missing values, and basic statistics |
| 3. Clean | Removes duplicates, adds `years_active`, replaces zero bowling stats with NaN |
| 4. EDA | Top run scorers, top wicket takers, country-wise averages |
| 5. Visualise | 6 matplotlib/seaborn charts saved to `outputs/` |
| 6. Score | Computes a composite performance score (weighted normalisation) |
| 7. Summary | Prints dataset highlights |

---

##  Performance Score Formula
```
score = runs(30%) + average(25%) + strike_rate(15%) + wickets(20%) + catches(10%)
```
Each metric is min-max normalised to [0, 1] before weighting.

---

##  Dependencies
- `pandas` – data manipulation  
- `numpy` – numerical operations  
- `matplotlib` – plotting  
- `seaborn` – statistical visualisations  
