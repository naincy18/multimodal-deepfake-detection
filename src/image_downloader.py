from pathlib import Path
import pandas as pd
import requests
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Start with a small test, not the entire dataset
df = pd.read_csv(
    RAW_DIR / "train.tsv",
    sep="\t",
    usecols=["id", "image_url", "2_way_label"]
)

df = df.dropna(subset=["id", "image_url"])
df = df.drop_duplicates(subset=["id"]).head(20)

session = requests.Session()
session.headers.update({
    "User-Agent": "DeepfakeDetectionStudentProject/1.0"
})

success = 0
failed = 0

for _, row in df.iterrows():
    image_id = str(row["id"])
    image_url = str(row["image_url"])

    # Use the post ID as the image filename
    output_path = IMAGE_DIR / f"{image_id}.jpg"

    if output_path.exists():
        print(f"Already downloaded: {image_id}")
        continue

    try:
        response = session.get(
            image_url,
            timeout=15
        )
        response.raise_for_status()

        # Confirm the URL returned image data
        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if not content_type.startswith("image/"):
            print(f"Not an image: {image_id}")
            failed += 1
            continue

        output_path.write_bytes(response.content)
        print(f"Downloaded: {image_id}")
        success += 1

    except (requests.RequestException, OSError) as error:
        print(f"Failed: {image_id} - {error}")
        failed += 1

    time.sleep(0.2)

print("\nDownload test complete.")
print("Successful:", success)
print("Failed:", failed)
print("Images folder:", IMAGE_DIR)
