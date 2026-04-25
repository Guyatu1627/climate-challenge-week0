import json
from pathlib import Path
path = Path(r'c:\Users\Guyatu\climate-challenge-week0\notebooks\ethiopia_eda.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
updated = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code' and any('from scipy import stats' in line for line in cell.get('source', [])):
        cell['source'] = [
            'from scipy import stats\n',
            'import numpy as np\n',
            'import pandas as pd\n',
            '\n',
            "# 0. Ensure dataframe is loaded\n",
            "if 'df' not in globals():\n",
            '    df = pd.read_csv("../data/ethiopia.csv")\n',
            '    df["Country"] = "Ethiopia"\n',
            '    df["Date"] = pd.to_datetime(df["YEAR"] * 1000 + df["DOY"], format="%Y%j")\n',
            '    df["Month"] = df["Date"].dt.month\n',
            '    df = df.replace(-999, np.nan)\n',
            '\n',
            "# 1. Define the columns we want to track for weird values\n",
            "cols_to_check = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX']\n",
            '\n',
            '# 2. Calculate Z-scores\n',
            '# we use np.abs to turn negative scores into positive ones\n',
            "z_scores = np.abs(stats.zscore(df[cols_to_check], nan_policy='omit'))\n",
            '\n',
            '# 3. Find rows where any column has a Z-score > 3\n',
            'outliers = (z_scores > 3).any(axis=1)\n',
            'print(f"Number of outliers rows detected: {outliers.sum()}")\n',
            '\n',
            '# 4. Show a few outliers to see if they look like errors or just extreme weather\n',
            'df[outliers].head()\n'
        ]
        updated = True
if not updated:
    raise SystemExit('No matching cell found to update')
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Updated notebook cell successfully')
