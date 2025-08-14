import pandas as pd
from sqlalchemy import create_engine
import os

# Print current working directory to debug file path issues
print("Current working directory:", os.getcwd())

# Define your SQLite DB URL
database_url = "sqlite:///database.db"
engine = create_engine(database_url)

# Define CSV file names and their corresponding SQL table names
csv_to_table_map = {
    "train.csv": "train_table",
    "ideal.csv": "ideal_table",
    "test.csv": "test_table"
}

# If files are in the same directory as script, use "."
csv_directory = "."  # Change this if your files are elsewhere

# Loop over each CSV and insert into the database
for csv_file, table_name in csv_to_table_map.items():
    csv_path = os.path.join(csv_directory, csv_file)
    print(f"Checking for file at: {csv_path}")

    if os.path.exists(csv_path):
        print(f"Processing {csv_file} -> {table_name}")
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        print(f"✔️  {table_name} created in database.db")
    else:
        print(f"❌ File not found: {csv_path}")
