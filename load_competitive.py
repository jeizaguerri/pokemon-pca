import requests
import pandas as pd
import re

SMOGON_STATS_URL = "https://www.smogon.com/stats/2024-12/gen9ou-1825.txt"
def download_data(url):
    try:
        print(f"Fetching data from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def parse_smogon_usage_table(text):
    lines = text.splitlines()

    # Find the start of the table
    table_start = False
    data_rows = []

    for line in lines:
        if re.match(r"\s*\|\s*Rank\s*\|\s*Pokemon", line):
            table_start = True
            continue
        if table_start:
            if re.match(r"\s*\+-+", line):  # Skip separator lines
                continue
            if line.strip() == '':
                break  # End of table
            parts = [part.strip() for part in line.strip('| \n').split('|')]
            if len(parts) == 7:  # Expected number of columns
                data_rows.append(parts)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=[
        'Rank', 'Pokemon', 'Usage %', 'Raw', 'Raw %', 'Real', 'Real %'
    ])

    # Convert numeric columns
    for col in ['Usage %', 'Raw', 'Raw %', 'Real', 'Real %']:
        df[col] = df[col].str.replace('%', '').str.replace(',', '').astype(float)

    df['Rank'] = df['Rank'].astype(int)
    
    return df


def main():
    # Download data
    data = download_data(SMOGON_STATS_URL)
    if data is None:
        print("Failed to download data.")
        return
    
    # Parse data
    parsed_data = parse_smogon_usage_table(data)
    if parsed_data is None:
        print("Failed to parse data.")
        return
    
    # Save to CSV
    parsed_data.to_csv('smogon_usage_stats.csv', index=False)
    print("Data saved to smogon_usage_stats.csv")

if __name__ == "__main__":
    main()
    