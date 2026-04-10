import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================
# LOAD DATASETS
# ============================
file1 = "100k_filtered_raw_bug_reports.xlsx"
file2 = "bugs-2025-02-23 (1).csv"

df1 = pd.read_excel(file1)
df2 = pd.read_csv(file2)

df = pd.concat([df1, df2], ignore_index=True)

print("Original shape:", df.shape)

# ============================
# CLEAN COLUMN NAMES
# ============================
df.columns = df.columns.str.lower()

# REMOVE DUPLICATE COLUMNS
df = df.loc[:, ~df.columns.duplicated()]

print("Columns after cleaning:", df.columns)

# ============================
# FIX BUG_ID COLUMN
# ============================
if 'bug_id' not in df.columns:
    for col in df.columns:
        if 'id' in col:
            df.rename(columns={col: 'bug_id'}, inplace=True)
            break

# Ensure bug_id is 1D
if isinstance(df['bug_id'], pd.DataFrame):
    df['bug_id'] = df['bug_id'].iloc[:, 0]

df['bug_id'] = pd.to_numeric(df['bug_id'], errors='coerce')

# ============================
# FIX STATUS COLUMN
# ============================
if 'status' not in df.columns:
    raise Exception("No status column found!")

# Ensure status is 1D
if isinstance(df['status'], pd.DataFrame):
    df['status'] = df['status'].iloc[:, 0]

df['status'] = df['status'].astype(str)

# ============================
# TEXT COLUMN HANDLING
# ============================
df['summary_final'] = df.get('summary', df.get('bug report', ""))
df['comment_final'] = df.get('text', "")

df['combined_text'] = df['summary_final'].fillna('') + " " + df['comment_final'].fillna('')

# ============================
# REMOVE DUPLICATE ROWS
# ============================
df = df.drop_duplicates(subset=['bug_id', 'combined_text'])

print("After duplicate removal:", df.shape)

# ============================
# PART 1: LIFECYCLE ANALYSIS
# ============================
print("\n===== BUG LIFECYCLE ANALYSIS (FINAL) =====")

lifecycle_df = df.groupby('bug_id').agg(
    unique_status_count=('status', 'nunique'),
    entry_count=('status', 'count')
).reset_index()

print(lifecycle_df.describe())

# ============================
# CLUSTERING
# ============================
features = lifecycle_df[['unique_status_count', 'entry_count']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
lifecycle_df['cluster'] = kmeans.fit_predict(scaled)

print("\nCluster Distribution:")
print(lifecycle_df['cluster'].value_counts())

# Plot
plt.figure()
sns.scatterplot(
    x=lifecycle_df['entry_count'],
    y=lifecycle_df['unique_status_count'],
    hue=lifecycle_df['cluster']
)
plt.title("Bug Behavior Clusters (FINAL)")
plt.xlabel("Entries per Bug")
plt.ylabel("Unique Status Count")
plt.show()

# ============================
# PART 2: TEXT QUALITY ANALYSIS
# ============================
print("\n===== COMMENT QUALITY ANALYSIS =====")

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [w for w in words if len(w) > 2]
    
    return " ".join(words)

df['clean_text'] = df['combined_text'].apply(clean_text)

def quality_score(text):
    words = text.split()
    length = len(words)

    keywords = ['error', 'fail', 'crash', 'exception', 'bug', 'issue']

    score = 0
    if length > 5:
        score += 1
    if length > 15:
        score += 1
    if any(k in text for k in keywords):
        score += 1

    return score

df['quality_score'] = df['clean_text'].apply(quality_score)

df['quality_label'] = df['quality_score'].apply(
    lambda x: 'genuine' if x >= 2 else 'filler'
)

print("\nQuality Distribution:")
print(df['quality_label'].value_counts())

# ============================
# NLP CLUSTERING
# ============================
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])

kmeans_text = KMeans(n_clusters=2, random_state=42, n_init=10)
df['text_cluster'] = kmeans_text.fit_predict(X)

# ============================
# SAVE RESULTS
# ============================
lifecycle_df.to_csv("bug_lifecycle_analysis_FINAL.csv", index=False)
df.to_csv("bug_quality_analysis_FINAL.csv", index=False)

print("\n✅ FINAL ANALYSIS COMPLETE!")
print("Saved files:")
print(" - bug_lifecycle_analysis_FINAL.csv")
print(" - bug_quality_analysis_FINAL.csv")