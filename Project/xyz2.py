"""
=============================================================================
  ADVANCED BUG ANALYSIS SCRIPT
  - Dataset 1: bugs-XXXX.csv  (Bug ID, Type, Summary, Product, Component,
                                Assignee, Status, Resolution, Updated)
  - Dataset 2: raw comments   (author, creation_time, bug_id, text,
                                Bug report, author_id, status, resolution,
                                contains_steps_to_reproduce)

  Analyses:
    1. Systematic Bug Lifecycle  – open / close / re-open patterns per bug_id
    2. Comment / Summary Quality – genuine vs filler classification (rule +
                                   ML hybrid)

  Run:
      pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
      python advanced_bug_analysis.py
=============================================================================
"""

import re
import os
import warnings
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – change to "TkAgg" if you want popups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION  (edit paths here if needed)
# ─────────────────────────────────────────────────────────────────────────────

DS1_PATH = "bugs-2025-02-23 (1).csv"          # summary-level CSV  (doc-1 style)
DS2_PATH = "100k_filtered_raw_bug_reports.xlsx" # comment-level XLSX (doc-2 style)
OUT_DIR  = Path("bug_analysis_output")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & HARMONISE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset1(path: str) -> pd.DataFrame:
    """
    Loads the summary-level CSV.
    Expected columns (case-insensitive):
        Bug ID, Type, Summary, Product, Component, Assignee,
        Status, Resolution, Updated
    """
    print(f"\n[1] Loading Dataset-1 from: {path}")
    try:
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        print(f"  ⚠  File not found: {path} — generating synthetic demo data.")
        df = _synthetic_ds1()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename_map = {
        "bug_id": "bug_id", "id": "bug_id",
        "type": "type",
        "summary": "summary",
        "product": "product",
        "component": "component",
        "assignee": "assignee",
        "status": "status",
        "resolution": "resolution",
        "updated": "updated",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    if "bug_id" not in df.columns:
        raise ValueError("Dataset-1 must have a 'Bug ID' column.")

    df["bug_id"] = df["bug_id"].astype(str).str.strip()
    df["source"] = "ds1"
    print(f"  Rows: {len(df):,}   Unique bug_ids: {df['bug_id'].nunique():,}")
    return df


def load_dataset2(path: str) -> pd.DataFrame:
    """
    Loads the comment-level XLSX / CSV.
    Expected columns (case-insensitive):
        author, creation_time, bug_id, text, Bug report, author_id,
        status, resolution, contains_steps_to_reproduce
    Falls back to CSV if xlsx fails / not found.
    """
    print(f"\n[2] Loading Dataset-2 from: {path}")
    try:
        if path.endswith(".xlsx"):
            df = pd.read_excel(path, dtype=str)
        else:
            df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        print(f"  ⚠  File not found: {path} — generating synthetic demo data.")
        df = _synthetic_ds2()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Harmonise column names
    for old, new in [
        ("bug_report", "is_bug_report"),
        ("contains_steps_to_reproduce", "has_steps"),
        ("text", "comment_text"),
    ]:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "bug_id" not in df.columns:
        raise ValueError("Dataset-2 must have a 'bug_id' column.")

    df["bug_id"] = df["bug_id"].astype(str).str.strip()

    # Parse creation_time
    if "creation_time" in df.columns:
        df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce", utc=True)

    df["source"] = "ds2"
    print(f"  Rows: {len(df):,}   Unique bug_ids: {df['bug_id'].nunique():,}")
    return df


# ─── synthetic fallback data (so the script runs even without real files) ────

def _synthetic_ds1() -> pd.DataFrame:
    rows = [
        ["1801502","defect","Toolbar overflow menu flickers","Firefox","Extensions","alice","VERIFIED","FIXED","2022-11-19"],
        ["1801501","defect","Mixed content frame from https","Firefox","Security","bob","RESOLVED","FIXED","2022-11-19"],
        ["1801500","defect","CSS race condition","Core","Layout","carol","RESOLVED","FIXED","2022-11-19"],
        ["1801502","defect","Toolbar overflow menu flickers","Firefox","Extensions","alice","REOPENED","---","2022-11-20"],
        ["1801502","defect","Toolbar overflow menu flickers","Firefox","Extensions","alice","RESOLVED","FIXED","2022-11-22"],
        ["1801488","defect","AVIF thumbnail not shown","Firefox","File Dialog","dave","RESOLVED","FIXED","2022-11-19"],
        ["1801488","defect","AVIF thumbnail not shown","Firefox","File Dialog","dave","REOPENED","---","2022-11-21"],
        ["1801488","defect","AVIF thumbnail not shown","Firefox","File Dialog","dave","VERIFIED","FIXED","2022-11-23"],
        ["1801479","defect","Extension test failure","Firefox","Extensions","eve","VERIFIED","FIXED","2022-11-19"],
        ["1801471","defect","Unified extension panel crash","Firefox","Extensions","frank","RESOLVED","FIXED","2022-11-19"],
    ]
    return pd.DataFrame(rows, columns=["Bug ID","Type","Summary","Product",
                                        "Component","Assignee","Status",
                                        "Resolution","Updated"])


def _synthetic_ds2() -> pd.DataFrame:
    import random, string
    random.seed(42)
    bugs = ["1801502","1801501","1801500","1801488","1801479","1801471"]
    statuses = ["RESOLVED","VERIFIED","REOPENED","NEW","ASSIGNED"]
    rows = []
    for bug in bugs:
        for i in range(random.randint(2,6)):
            rows.append({
                "author": f"user{random.randint(1,20)}@example.com",
                "creation_time": f"2022-11-{random.randint(18,25):02d}T{random.randint(0,23):02d}:00:00Z",
                "bug_id": bug,
                "comment_text": random.choice([
                    "Steps to reproduce: 1. Open Firefox 2. Go to about:config actual: crash expected: no crash",
                    "created attachment 12345 screenshot",
                    "see detailed analysis at https://bugzilla.mozilla.org/...",
                    ".",
                    "bump",
                    "This is a regression. Last good build: 2022-11-15 first bad: 2022-11-16",
                    "Fixed in nightly. The issue was caused by a race condition in the IPC layer.",
                    "x",
                    "user agent: Mozilla/5.0 windows nt 10.0 actual: wrong behavior expected: correct behavior steps: 1. do this 2. do that",
                ]),
                "is_bug_report": "TRUE",
                "author_id": str(random.randint(100000, 999999)),
                "status": random.choice(statuses),
                "resolution": random.choice(["FIXED","---","DUPLICATE"]),
                "has_steps": str(random.choice(["TRUE","FALSE"])),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LIFECYCLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Status transition graph (based on Mozilla Bugzilla conventions)
OPEN_STATUSES   = {"NEW", "ASSIGNED", "REOPENED", "UNCONFIRMED", "IN_PROGRESS"}
CLOSED_STATUSES = {"RESOLVED", "VERIFIED", "CLOSED"}

def classify_status(status: str) -> str:
    s = str(status).upper().strip()
    if s in OPEN_STATUSES:
        return "open"
    if s in CLOSED_STATUSES:
        return "closed"
    return "unknown"


def analyse_lifecycle(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Combines both datasets, groups by bug_id, and detects
    open → closed → open (reopen) patterns.

    Returns a dict with:
      - lifecycle_df   : per-bug summary DataFrame
      - transition_df  : all (bug_id, time, from_status, to_status) transitions
      - pattern_counts : dict of pattern label → count
    """
    print("\n[3] Running Lifecycle Analysis …")

    frames = []

    # --- Dataset 1: one row per status snapshot ---
    if "updated" in df1.columns and "status" in df1.columns:
        tmp = df1[["bug_id","status","updated"]].copy()
        tmp.rename(columns={"updated":"event_time"}, inplace=True)
        tmp["event_time"] = pd.to_datetime(tmp["event_time"], errors="coerce", utc=True)
        frames.append(tmp)

    # --- Dataset 2: comment-level status column ---
    if "status" in df2.columns and "creation_time" in df2.columns:
        tmp = df2[["bug_id","status","creation_time"]].copy()
        tmp.rename(columns={"creation_time":"event_time"}, inplace=True)
        frames.append(tmp)

    if not frames:
        print("  ⚠  No time-stamped status data found; skipping lifecycle.")
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["event_time","status"])
    combined["status_class"] = combined["status"].apply(classify_status)
    combined = combined[combined["status_class"] != "unknown"]
    combined = combined.sort_values(["bug_id","event_time"])

    # Deduplicate: keep unique (bug_id, status_class, event_time) rows
    combined = combined.drop_duplicates(subset=["bug_id","status_class","event_time"])

    # Build per-bug transition sequences
    results = []
    all_transitions = []

    for bug_id, grp in combined.groupby("bug_id"):
        seq = grp["status_class"].tolist()
        times = grp["event_time"].tolist()

        # Count transitions
        opens   = seq.count("open")
        closes  = seq.count("closed")
        reopens = 0
        pattern = "simple"

        prev = None
        for i, s in enumerate(seq):
            if prev == "closed" and s == "open":
                reopens += 1
            if i > 0:
                all_transitions.append({
                    "bug_id":      bug_id,
                    "event_time":  times[i],
                    "from_status": seq[i-1],
                    "to_status":   s,
                })
            prev = s

        if reopens >= 2:
            pattern = "chronic_reopen"
        elif reopens == 1:
            pattern = "single_reopen"
        elif closes >= 1 and opens >= 1:
            pattern = "normal_close"
        elif opens >= 1 and closes == 0:
            pattern = "never_closed"

        results.append({
            "bug_id":         bug_id,
            "total_events":   len(seq),
            "open_count":     opens,
            "close_count":    closes,
            "reopen_count":   reopens,
            "pattern":        pattern,
            "status_sequence": " → ".join(seq),
        })

    lifecycle_df   = pd.DataFrame(results)
    transition_df  = pd.DataFrame(all_transitions)
    pattern_counts = lifecycle_df["pattern"].value_counts().to_dict()

    print(f"  Bugs analysed : {len(lifecycle_df):,}")
    print(f"  Pattern counts: {pattern_counts}")

    return {
        "lifecycle_df":   lifecycle_df,
        "transition_df":  transition_df,
        "pattern_counts": pattern_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMMENT / SUMMARY QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. Rule-based quality scorer ──────────────────────────────────────────

FILLER_PATTERNS = [
    r"^\s*\.\s*$",
    r"^\s*(bump|ping|triage|me too|same here|same issue|same problem|\+1|thanks)\s*$",
    r"^(x|ok|yes|no|done|fixed|see above|see below|see comment)\s*$",
    r"^created attachment \d+\s*$",
]

QUALITY_KEYWORDS = [
    "steps to reproduce", "str:", "actual result", "expected result",
    "regression", "error", "crash", "exception", "traceback", "stack trace",
    "assert", "fail", "bug", "issue", "fix", "patch", "workaround",
    "user agent", "build id", "version", "nightly", "release", "bisect",
    "log", "console", "screenshot", "attachment",
]

STRONG_QUALITY_RE = re.compile(
    r"(steps?\s+to\s+repro|actual\s+result|expected\s+result|str\s*\d*\s*[:\.])",
    re.IGNORECASE,
)

def rule_quality_score(text: str) -> dict:
    """Returns a dict with raw_score (0-6) and label (genuine/borderline/filler)."""
    if not isinstance(text, str):
        text = str(text) if text else ""

    text_lower = text.lower().strip()

    # Immediate filler detection
    for pat in FILLER_PATTERNS:
        if re.match(pat, text_lower, re.IGNORECASE):
            return {"raw_score": 0, "label": "filler",
                    "has_steps": False, "word_count": len(text_lower.split())}

    words      = text_lower.split()
    word_count = len(words)

    score = 0

    # Length-based
    if word_count >= 10:  score += 1
    if word_count >= 30:  score += 1
    if word_count >= 80:  score += 1

    # Keyword hit
    keyword_hits = sum(1 for kw in QUALITY_KEYWORDS if kw in text_lower)
    if keyword_hits >= 1: score += 1
    if keyword_hits >= 3: score += 1

    # Strong structural signals (steps to reproduce, etc.)
    has_steps = bool(STRONG_QUALITY_RE.search(text))
    if has_steps: score += 1

    label = "genuine" if score >= 4 else ("borderline" if score >= 2 else "filler")

    return {
        "raw_score":  score,
        "label":      label,
        "has_steps":  has_steps,
        "word_count": word_count,
    }


# ── 3b. NLP / TF-IDF cluster-based quality ─────────────────────────────────

def clean_for_nlp(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # drop very short tokens
    return " ".join(w for w in text.split() if len(w) > 2)


def nlp_quality_analysis(texts: pd.Series,
                         n_clusters: int = 3) -> pd.DataFrame:
    """
    Runs TF-IDF + SVD (LSA) + KMeans on the text corpus.
    Returns a DataFrame with columns: tfidf_cluster, svd_x, svd_y
    """
    print("  Running TF-IDF + LSA + KMeans …")
    clean = texts.apply(clean_for_nlp)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=2)),
        ("svd",   TruncatedSVD(n_components=50, random_state=42)),
    ])
    X = pipe.fit_transform(clean)

    # 2-D projection for plotting
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2   = svd2.fit_transform(
        TfidfVectorizer(max_features=3000, min_df=2,
                        sublinear_tf=True).fit_transform(clean)
    )

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    return pd.DataFrame({
        "tfidf_cluster": labels,
        "svd_x": X2[:, 0],
        "svd_y": X2[:, 1],
    })


def analyse_quality(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Applies rule-based + NLP quality analysis to:
      - df1 summaries
      - df2 comment texts
    """
    print("\n[4] Running Quality Analysis …")

    # ── DS1: summaries ──
    df1 = df1.copy()
    if "summary" not in df1.columns:
        df1["summary"] = ""
    scores1 = df1["summary"].apply(rule_quality_score).apply(pd.Series)
    
    # FIX: Drop overlapping columns to prevent duplicate column names
    cols_to_drop1 = [c for c in scores1.columns if c in df1.columns]
    df1 = pd.concat([df1.drop(columns=cols_to_drop1).reset_index(drop=True),
                     scores1.reset_index(drop=True)], axis=1)

    # ── DS2: comment text ──
    df2 = df2.copy()
    text_col = "comment_text" if "comment_text" in df2.columns else "text" if "text" in df2.columns else None
    if text_col is None:
        df2["comment_text"] = ""
        text_col = "comment_text"

    # Deduplicate comments per bug
    df2 = df2.drop_duplicates(subset=["bug_id", text_col])

    scores2 = df2[text_col].apply(rule_quality_score).apply(pd.Series)
    
    # FIX: Drop overlapping columns to prevent duplicate column names
    cols_to_drop2 = [c for c in scores2.columns if c in df2.columns]
    df2 = pd.concat([df2.drop(columns=cols_to_drop2).reset_index(drop=True),
                     scores2.reset_index(drop=True)], axis=1)

    # NLP on DS2 comments (usually larger dataset)
    if len(df2) >= 10:
        n_clust = min(5, max(2, len(df2) // 20))
        nlp_res = nlp_quality_analysis(df2[text_col], n_clusters=n_clust)
        
        # Ensure no duplicate columns here either
        cols_to_drop_nlp = [c for c in nlp_res.columns if c in df2.columns]
        df2 = pd.concat([df2.drop(columns=cols_to_drop_nlp).reset_index(drop=True),
                         nlp_res.reset_index(drop=True)], axis=1)
    else:
        df2["tfidf_cluster"] = 0
        df2["svd_x"] = 0.0
        df2["svd_y"] = 0.0

    print(f"  DS1 summary quality: {df1['label'].value_counts().to_dict()}")
    print(f"  DS2 comment quality: {df2['label'].value_counts().to_dict()}")

    # ── Per-bug quality aggregation ──
    # FIX: Pre-calculate flags to use simple vectorized aggregations instead of lambdas
    df2["is_genuine"] = (df2["label"] == "genuine").astype(int)
    df2["is_filler"] = (df2["label"] == "filler").astype(int)
    df2["is_borderline"] = (df2["label"] == "borderline").astype(int)
    df2["has_steps_bool"] = df2["has_steps"].fillna(False).astype(bool)

    bug_quality = (
        df2.groupby("bug_id")
           .agg(
               comment_count    = ("bug_id", "count"),
               genuine_count    = ("is_genuine", "sum"),
               filler_count     = ("is_filler", "sum"),
               borderline_count = ("is_borderline", "sum"),
               has_any_steps    = ("has_steps_bool", "max"), # max on bool acts identically to any()
               avg_word_count   = ("word_count", "mean"),
               avg_score        = ("raw_score", "mean"),
           )
           .reset_index()
    )
    bug_quality["genuine_ratio"] = (
        bug_quality["genuine_count"] / bug_quality["comment_count"].clip(lower=1)
    )

    return {
        "df1_quality": df1,
        "df2_quality": df2,
        "bug_quality": bug_quality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "genuine":    "#2ecc71",
    "borderline": "#f39c12",
    "filler":     "#e74c3c",
    "normal_close":    "#3498db",
    "single_reopen":   "#e67e22",
    "chronic_reopen":  "#c0392b",
    "never_closed":    "#9b59b6",
    "simple":          "#95a5a6",
}


def plot_lifecycle(lc: dict, out_dir: Path):
    lifecycle_df   = lc.get("lifecycle_df", pd.DataFrame())
    transition_df  = lc.get("transition_df", pd.DataFrame())
    pattern_counts = lc.get("pattern_counts", {})

    if lifecycle_df.empty:
        print("  ⚠  No lifecycle data to plot.")
        return

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── (a) Pattern distribution pie ──
    ax1 = fig.add_subplot(gs[0, 0])
    labels  = list(pattern_counts.keys())
    sizes   = list(pattern_counts.values())
    colors  = [PALETTE.get(l, "#bdc3c7") for l in labels]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 9},
    )
    ax1.set_title("Bug Lifecycle Patterns", fontweight="bold")

    # ── (b) Reopen count distribution ──
    ax2 = fig.add_subplot(gs[0, 1])
    reopen_counts = lifecycle_df["reopen_count"].value_counts().sort_index()
    bars = ax2.bar(
        reopen_counts.index.astype(str),
        reopen_counts.values,
        color=["#3498db" if i == 0 else "#e74c3c" for i in reopen_counts.index],
        edgecolor="white", linewidth=0.8,
    )
    ax2.set_title("Re-open Frequency per Bug", fontweight="bold")
    ax2.set_xlabel("Re-open Count")
    ax2.set_ylabel("Number of Bugs")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 str(int(bar.get_height())),
                 ha="center", va="bottom", fontsize=8)

    # ── (c) Open vs Close vs Reopen scatter ──
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(
        lifecycle_df["open_count"],
        lifecycle_df["close_count"],
        c=lifecycle_df["reopen_count"],
        cmap="Reds", s=60, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax3, label="Reopen count")
    ax3.set_title("Open vs Close Events\n(colour = reopen count)", fontweight="bold")
    ax3.set_xlabel("Open Events")
    ax3.set_ylabel("Close Events")

    # ── (d) Transition heatmap ──
    ax4 = fig.add_subplot(gs[1, 0])
    if not transition_df.empty:
        pivot = (
            transition_df.groupby(["from_status","to_status"])
                         .size()
                         .unstack(fill_value=0)
        )
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.5, ax=ax4,
                    cbar_kws={"shrink": 0.8})
        ax4.set_title("Status Transition Matrix", fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No transition data", ha="center", va="center")
        ax4.axis("off")

    # ── (e) Chronic re-openers table ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    chronic = (
        lifecycle_df[lifecycle_df["reopen_count"] >= 1]
        .sort_values("reopen_count", ascending=False)
        .head(15)
    )
    if not chronic.empty:
        cols = ["bug_id","open_count","close_count","reopen_count","pattern"]
        table_data = [cols] + chronic[cols].values.tolist()
        col_labels = ["Bug ID","Opens","Closes","Reopens","Pattern"]
        tbl = ax5.table(
            cellText=chronic[cols].values,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        # Colour header
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        ax5.set_title("Top Re-opened Bugs", fontweight="bold", pad=12)

    fig.suptitle("Bug Lifecycle Analysis", fontsize=16, fontweight="bold", y=1.01)
    path = out_dir / "lifecycle_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")


def plot_quality(qr: dict, out_dir: Path):
    df1_q   = qr.get("df1_quality", pd.DataFrame())
    df2_q   = qr.get("df2_quality", pd.DataFrame())
    bug_q   = qr.get("bug_quality", pd.DataFrame())

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── (a) DS1 summary quality bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    if "label" in df1_q.columns:
        vc = df1_q["label"].value_counts()
        bars = ax1.bar(vc.index, vc.values,
                       color=[PALETTE.get(l,"#bdc3c7") for l in vc.index],
                       edgecolor="white")
        ax1.set_title("DS1: Summary Quality", fontweight="bold")
        ax1.set_ylabel("Count")
        for b in bars:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                     str(int(b.get_height())), ha="center", fontsize=8)

    # ── (b) DS2 comment quality bar ──
    ax2 = fig.add_subplot(gs[0, 1])
    if "label" in df2_q.columns:
        vc = df2_q["label"].value_counts()
        bars = ax2.bar(vc.index, vc.values,
                       color=[PALETTE.get(l,"#bdc3c7") for l in vc.index],
                       edgecolor="white")
        ax2.set_title("DS2: Comment Quality", fontweight="bold")
        ax2.set_ylabel("Count")
        for b in bars:
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                     str(int(b.get_height())), ha="center", fontsize=8)

    # ── (c) Quality score distribution ──
    ax3 = fig.add_subplot(gs[0, 2])
    if "raw_score" in df2_q.columns:
        df2_q["raw_score"].value_counts().sort_index().plot(
            kind="bar", ax=ax3, color="#3498db", edgecolor="white"
        )
        ax3.set_title("DS2: Raw Quality Score Distribution", fontweight="bold")
        ax3.set_xlabel("Score (0–6)")
        ax3.set_ylabel("Count")

    # ── (d) Word count by quality label ──
    ax4 = fig.add_subplot(gs[1, 0])
    if "word_count" in df2_q.columns and "label" in df2_q.columns:
        for label, grp in df2_q.groupby("label"):
            ax4.hist(grp["word_count"].clip(upper=300), bins=30,
                     alpha=0.6, label=label,
                     color=PALETTE.get(label,"#bdc3c7"))
        ax4.set_title("Word Count by Quality Label", fontweight="bold")
        ax4.set_xlabel("Word Count")
        ax4.set_ylabel("Frequency")
        ax4.legend()

    # ── (e) NLP cluster scatter ──
    ax5 = fig.add_subplot(gs[1, 1])
    if "svd_x" in df2_q.columns:
        for label, grp in df2_q.groupby("label"):
            ax5.scatter(grp["svd_x"], grp["svd_y"],
                        c=PALETTE.get(label,"#bdc3c7"),
                        label=label, alpha=0.5, s=15, edgecolors="none")
        ax5.set_title("NLP Text Space (LSA projection)", fontweight="bold")
        ax5.set_xlabel("LSA dim-1")
        ax5.set_ylabel("LSA dim-2")
        ax5.legend(markerscale=2, fontsize=8)

    # ── (f) Per-bug genuine ratio ──
    ax6 = fig.add_subplot(gs[1, 2])
    if "genuine_ratio" in bug_q.columns:
        ax6.hist(bug_q["genuine_ratio"], bins=20,
                 color="#2ecc71", edgecolor="white")
        ax6.set_title("Per-Bug Genuine Comment Ratio", fontweight="bold")
        ax6.set_xlabel("Genuine Ratio (0–1)")
        ax6.set_ylabel("Number of Bugs")
        ax6.axvline(bug_q["genuine_ratio"].mean(), color="red",
                    linestyle="--", label=f"Mean: {bug_q['genuine_ratio'].mean():.2f}")
        ax6.legend()

    # ── (g) steps-to-reproduce prevalence ──
    ax7 = fig.add_subplot(gs[2, 0])
    if "has_steps" in df2_q.columns:
        vc = df2_q["has_steps"].value_counts()
        ax7.pie(vc.values, labels=["Has Steps" if v else "No Steps" for v in vc.index],
                colors=["#2ecc71","#e74c3c"], autopct="%1.1f%%", startangle=90)
        ax7.set_title("Comments with\nSteps-to-Reproduce", fontweight="bold")

    # ── (h) per-bug quality heatmap (top 20 bugs) ──
    ax8 = fig.add_subplot(gs[2, 1:])
    if not bug_q.empty:
        top20 = bug_q.nlargest(20, "comment_count")
        heat_data = top20[["genuine_count","borderline_count",
                            "filler_count"]].set_index(top20["bug_id"])
        sns.heatmap(heat_data, annot=True, fmt=".0f",
                    cmap="RdYlGn", ax=ax8,
                    linewidths=0.5, cbar_kws={"shrink": 0.6})
        ax8.set_title("Quality Breakdown – Top 20 Most-Commented Bugs",
                      fontweight="bold")
        ax8.set_xticklabels(["Genuine","Borderline","Filler"], rotation=0)

    fig.suptitle("Bug Comment / Summary Quality Analysis",
                 fontsize=16, fontweight="bold", y=1.01)
    path = out_dir / "quality_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE RESULTS TO CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results(lc: dict, qr: dict, out_dir: Path):
    print("\n[5] Saving CSV results …")

    if lc.get("lifecycle_df") is not None and not lc["lifecycle_df"].empty:
        p = out_dir / "lifecycle_per_bug.csv"
        lc["lifecycle_df"].to_csv(p, index=False)
        print(f"  → {p}")

    if lc.get("transition_df") is not None and not lc["transition_df"].empty:
        p = out_dir / "lifecycle_transitions.csv"
        lc["transition_df"].to_csv(p, index=False)
        print(f"  → {p}")

    if qr.get("df1_quality") is not None and not qr["df1_quality"].empty:
        p = out_dir / "ds1_summary_quality.csv"
        qr["df1_quality"].to_csv(p, index=False)
        print(f"  → {p}")

    if qr.get("df2_quality") is not None and not qr["df2_quality"].empty:
        p = out_dir / "ds2_comment_quality.csv"
        qr["df2_quality"].to_csv(p, index=False)
        print(f"  → {p}")

    if qr.get("bug_quality") is not None and not qr["bug_quality"].empty:
        p = out_dir / "per_bug_quality_summary.csv"
        qr["bug_quality"].to_csv(p, index=False)
        print(f"  → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PRINT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(lc: dict, qr: dict):
    divider = "═" * 70

    print(f"\n{divider}")
    print("  FINAL ANALYSIS REPORT")
    print(divider)

    # ── Lifecycle ──
    print("\n  ── BUG LIFECYCLE ──")
    pc = lc.get("pattern_counts", {})
    if pc:
        total = sum(pc.values())
        for pat, cnt in sorted(pc.items(), key=lambda x: -x[1]):
            print(f"    {pat:<25} {cnt:>6}  ({cnt/total*100:.1f}%)")

        chronic = lc["lifecycle_df"][lc["lifecycle_df"]["reopen_count"] >= 1]
        print(f"\n    Bugs reopened at least once : {len(chronic):,}")
        if not chronic.empty:
            print(f"    Max reopen count            : {chronic['reopen_count'].max()}")
            print(f"    Most re-opened bug IDs      : "
                  f"{', '.join(chronic.nlargest(5,'reopen_count')['bug_id'].tolist())}")
    else:
        print("    (no lifecycle data available)")

    # ── Quality ──
    print("\n  ── COMMENT / SUMMARY QUALITY ──")
    for name, key in [("DS1 Summaries","df1_quality"), ("DS2 Comments","df2_quality")]:
        df = qr.get(key, pd.DataFrame())
        if df.empty or "label" not in df.columns:
            continue
        vc = df["label"].value_counts()
        total = vc.sum()
        print(f"\n  {name}:")
        for lbl, cnt in vc.items():
            print(f"    {lbl:<15} {cnt:>6}  ({cnt/total*100:.1f}%)")

    bq = qr.get("bug_quality", pd.DataFrame())
    if not bq.empty:
        print(f"\n  Per-bug quality (DS2):")
        print(f"    Avg genuine ratio   : {bq['genuine_ratio'].mean():.2f}")
        print(f"    Bugs with any steps : {bq['has_any_steps'].sum()}")
        worst = bq.nsmallest(5, "genuine_ratio")
        print(f"    Lowest quality bugs : "
              f"{', '.join(worst['bug_id'].tolist())}")

    print(f"\n{divider}")
    print("  ✅  Analysis complete.  Outputs saved to: bug_analysis_output/")
    print(divider)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*70)
    print("  ADVANCED BUG ANALYSIS  –  Lifecycle + Quality")
    print("═"*70)

    # Load
    df1 = load_dataset1(DS1_PATH)
    df2 = load_dataset2(DS2_PATH)

    # Analyse
    lc = analyse_lifecycle(df1, df2)
    qr = analyse_quality(df1, df2)

    # Visualise
    print("\n[6] Generating visualisations …")
    if lc:
        plot_lifecycle(lc, OUT_DIR)
    plot_quality(qr, OUT_DIR)

    # Save CSVs
    save_results(lc, qr, OUT_DIR)

    # Report
    print_report(lc, qr)


if __name__ == "__main__":
    main()