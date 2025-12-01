import urllib.request
import zipfile
import os
from pathlib import Path

# Configuration
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    Download and extract the SMS Spam Collection dataset.
    
    Args:
        url: URL to download the zip file from
        zip_path: Local path to save the downloaded zip file
        extracted_path: Directory to extract files to
        data_file_path: Final path for the renamed data file
    """
    # Check if file already exists
    if data_file_path.exists():
        print(
            f"{data_file_path} already exists. "
            "Skipping download and extraction."
        )
        return
    
    # Download the zip file
    with urllib.request.urlopen(url) as response:  # 1
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:  # 2
        zip_ref.extractall(extracted_path)
    
    # Rename the extracted file to .tsv extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)  # 3
    
    print(f"File downloaded and saved as {data_file_path}")


# Execute the download
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)