import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Step 1: Load the CSV file
csv_file = "your_file.csv"  # Replace with your actual CSV file
data = pd.read_csv(csv_file)

# Step 2: Set up parameters
image_url_column = "clicklink"  # Replace with your image URL column name
output_dir = "downloaded_images"  # Directory to save images
os.makedirs(output_dir, exist_ok=True)

# Step 3: Define the image downloading function
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)  # Download the image
        response.raise_for_status()  # Check for HTTP request errors
        img = Image.open(BytesIO(response.content))  # Open the image
        img.save(save_path)  # Save the image
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Step 4: Download images
for index, row in data.iterrows():
    url = row[image_url_column]
    save_path = os.path.join(output_dir, f"image_{index}.jpg")  # Save with index-based naming
    download_image(url, save_path)

print("Image download complete!")
