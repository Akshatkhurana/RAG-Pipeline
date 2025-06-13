import pandas as pd
import os

# Path to your Excel file
excel_path = "../data/documents/network_elements_with_yearwise_versions.xlsx"

# Folder to save text files
output_dir = "../data/documents"
os.makedirs(output_dir, exist_ok=True)

# Load Excel file
df = pd.read_excel(excel_path)

# Preview columns
print("Columns:", df.columns)

# Write each row as a separate text file
for idx, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
    with open(os.path.join(output_dir, f"doc_{idx+1}.txt"), "w", encoding="utf-8") as f:
        f.write(content)

print(f"✅ Done! Saved {len(df)} text files to {output_dir}")