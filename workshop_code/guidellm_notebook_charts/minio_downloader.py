import boto3
import os
from datetime import datetime

def download_latest_benchmark_file(endpoint_url, access_key, secret_key, bucket_name, local_path="."):
    """
    Connects to a MinIO S3 bucket, finds the most recent benchmark file,
    and downloads it.

    The file naming convention is expected to be 'benchmark_<TIMESTAMP>_rate.txt',
    where <TIMESTAMP> is in the format 'YYYYMMDDHHMMSS'.

    Args:
        endpoint_url (str): The S3 API endpoint URL of the MinIO server.
        access_key (str): The access key (username) for MinIO.
        secret_key (str): The secret key (password) for MinIO.
        bucket_name (str): The name of the bucket to connect to.
        local_path (str, optional): The local directory to save the file to.
                                     Defaults to the current directory.
    """
    # --- 1. Establish a connection to the MinIO S3 bucket ---
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto3.session.Config(signature_version='s3v4')
        )
        print("Successfully connected to MinIO.")
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        return

    # --- 2. List all objects in the bucket ---
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            print(f"Bucket '{bucket_name}' is empty or does not exist.")
            return
        objects = response['Contents']
    except Exception as e:
        print(f"Error listing objects in bucket '{bucket_name}': {e}")
        return

    # --- 3. Filter for benchmark files and find the most recent one ---
    latest_file = None
    latest_timestamp = None
    file_prefix = "benchmark_"
    file_suffix = ".txt"

    for obj in objects:
        file_name = obj['Key']
        if file_name.startswith(file_prefix):
            try:
                # Extract the timestamp from the filename
                timestamp_str = file_name[len(file_prefix):-len(file_suffix)]
                
                current_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if latest_timestamp is None or current_timestamp > latest_timestamp:
                    latest_timestamp = current_timestamp
                    latest_file = file_name
            except ValueError:
                # Ignore files that don't have a valid timestamp format
                print(f"Skipping file with invalid timestamp format: {file_name}")
                continue

    if not latest_file:
        print("No files matching the 'benchmark_<TIMESTAMP>_rate.txt' format were found.")
        return

    print(f"Found the latest benchmark file: {latest_file}")

    # --- 4. Download the most recent file ---
    try:
        local_file_path = os.path.join(local_path, latest_file)
        s3_client.download_file(bucket_name, latest_file, local_file_path)
        print(f"Successfully downloaded '{latest_file}' to '{local_file_path}'")
        return latest_file
    except Exception as e:
        print(f"Error downloading file '{latest_file}': {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace these with your actual MinIO credentials and details.
    MINIO_ENDPOINT = "http://minio-service.rhaiis-demo.svc.cluster.local:9000"  # e.g., "http://192.168.1.100:9000"
    MINIO_ACCESS_KEY = "minio"
    MINIO_SECRET_KEY = "minio123"
    MINIO_BUCKET_NAME = "guidellm-benchmark"
    DOWNLOAD_DIRECTORY = "./downloads" # A directory to save the downloaded file

    # Create the download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)

    download_latest_benchmark_file(
        endpoint_url=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET_NAME,
        local_path=DOWNLOAD_DIRECTORY
    )
