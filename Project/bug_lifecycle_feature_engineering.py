import pandas as pd

# =============================
# 1. Load Dataset from Local CSV
# =============================
file_path = "dataset.csv"   # <-- change path if needed
df = pd.read_csv(file_path)
print(df.head())
# Convert timestamp to datetime if not already
df["date"] = pd.to_datetime(df["date"])

# Sort values for correct lifecycle order
df = df.sort_values(["bug_id", "date"])


# =============================
# 2. Feature Engineering Function
# =============================

def extract_bug_features(group):
    
    bug_id = group.name
    
    # First timestamps of key events
    new_time = group[group["status"] == "NEW"]["date"].min()
    assign_time = group[group["status"] == "ASSIGNED"]["date"].min()
    resolve_time = group[group["status"] == "RESOLVED"]["date"].min()
    verify_time = group[group["status"] == "VERIFIED"]["date"].min()
    
    # Final status
    final_status = group["status"].iloc[-1]
    
    # Time calculations (in hours)
    time_to_assign = (assign_time - new_time).total_seconds()/3600 if pd.notna(assign_time) and pd.notna(new_time) else None
    time_to_resolve = (resolve_time - new_time).total_seconds()/3600 if pd.notna(resolve_time) and pd.notna(new_time) else None
    time_to_verify = (verify_time - resolve_time).total_seconds()/3600 if pd.notna(verify_time) and pd.notna(resolve_time) else None
    
    lifecycle_time = (group["date"].max() - new_time).total_seconds()/3600 if pd.notna(new_time) else None
    
    # Reopen count
    reopen_count = (group["status"] == "REOPENED").sum()
    
    # Number of transitions
    num_status_changes = len(group)
    
    # Number of unique developers
    num_developers = group["user_id"].nunique()
    
    return pd.Series({
        "bug_id": bug_id,
        "time_to_assign_hr": time_to_assign,
        "time_to_resolve_hr": time_to_resolve,
        "time_to_verify_hr": time_to_verify,
        "lifecycle_time_hr": lifecycle_time,
        "reopen_count": reopen_count,
        "num_status_changes": num_status_changes,
        "num_developers": num_developers,
        "final_status": final_status,
        "reopened_flag": 1 if reopen_count > 0 else 0
    })


# =============================
# 3. Apply Feature Engineering
# =============================
features_df = df.groupby("bug_id").apply(extract_bug_features).reset_index(drop=True)


# =============================
# 4. Preview Results
# =============================
print(features_df.head())
print(features_df.shape)


# =============================
# 5. Save Engineered Dataset
# =============================
features_df.to_csv("engineered_bug_dataset.csv", index=False)

print("✅ Feature engineering completed and saved.")
