import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np

# Google Sheets setup
PATH_CREDENTIALS = "/home/fsoto/Documents/LCsSSL/logs/eval/true-area-463715-q9-83968305bd29.json"  # Replace with your credentials file path
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(PATH_CREDENTIALS, scope)
client = gspread.authorize(creds)

# Open the existing Google Sheets file
spreadsheet = client.open("LightCurves Experiments")  # Replace with your Google Sheets file name

# Function to upload CSV to a specific sheet
def upload_csv_to_sheet(csv_path, sheet_name):
    # Check if the sheet exists; if not, create it
    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")

    # Read CSV and upload to the sheet
    df = pd.read_csv(csv_path)
    df = df.replace([np.nan,np.inf,-np.inf], '')  # Replace NaN, inf, -inf with empty strings
    sheet.clear()  # Clear existing content
    sheet.update([df.columns.values.tolist()] + df.values.tolist())
    print(f"Uploaded {csv_path} to sheet: {sheet_name}")


