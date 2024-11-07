import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Load the CSV file
csv_file = "pinstripe1.csv"  # Replace with your actual CSV file
data = pd.read_csv(csv_file)

# Specify columns
image_url_column = "Image Link"  # Replace with your image URL column name
label_column = "labels"    # Replace with your label column name

# Directory to save images
output_dir = "pinstripeimages"
os.makedirs(output_dir, exist_ok=True)

# Function to sanitize the URL
def sanitize_url(url):
    if isinstance(url, str):  # Ensure the URL is a string
        clean_url = url.split(",")[0]  # Split by comma and take the first part
        return clean_url.strip()  # Remove any leading/trailing whitespace
    else:
        return None  # Return None for invalid or missing URLs

# Function to download and save an image
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)  # Download the image
        response.raise_for_status()  # Check for HTTP errors
        img = Image.open(BytesIO(response.content))  # Open the image
        img.save(save_path)  # Save the image
        print(f"Image saved: {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Iterate through rows
for index, row in data.iterrows():
    raw_url = row[image_url_column]
    url = sanitize_url(raw_url)  # Clean the URL
    if not url:  # Skip if the URL is invalid or None
        print(f"Skipping row {index}: Missing or invalid URL")
        continue
    
    label = row[label_column]
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Save path for the image
    save_path = os.path.join(label_dir, f"image_{index}.jpg")

    # Download the image
    download_image(url, save_path)

print("Image download process complete!")

