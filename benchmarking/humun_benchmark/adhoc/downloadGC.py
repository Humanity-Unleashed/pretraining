from google.cloud import storage
import sys
import os


# Get key from this notion page -
# https://humanity-unleashed.notion.site/FRED-Time-Series-Scraping-Tutorial-51774df4e0a5484e8458ae4665e53664#:~:text=service%2Daccount%2Dkey.json%20(a%20setup%20file%20with%20keys%20allowing%20you%20to%20access%20the%20GC%20space)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcs.json"  #!! make sure to edit this


def download_from_gcs(gcs_path, local_directory):
    """Downloads a file from GCS to a local directory using google-cloud-storage."""
    # Initialize a storage client
    client = storage.Client()

    # Extract the bucket name and blob name from the GCS path
    bucket_name, blob_name = gcs_path[5:].split("/", 1)

    # Construct the full local file path (local directory + filename)
    filename = os.path.basename(blob_name)
    local_path = os.path.join(local_directory, filename)

    try:
        # Get the bucket and blob
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download the blob to the local path
        blob.download_to_filename(local_path)

        print(f"Successfully downloaded {gcs_path} to {local_path}")
    except Exception as e:
        print(f"Failed to download {gcs_path} to {local_path}: {e}")


# Usage example:

# METADATA FILE PATH ON GCS - DO NOT CHANGE!
gcs_path = "gs://humun-storage/path/in/bucket/all_fred_metadata.csv"
# CHANGE TO LOCATION FOR WHERE YOU WANT THE METADATA FILE TO BE SAVED
local_directory = "."
download_from_gcs(gcs_path, local_directory)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python downloadGC.py <local_directory>")
        sys.exit(1)

    local_directory = sys.argv[1]
    download_from_gcs(gcs_path, local_directory)
