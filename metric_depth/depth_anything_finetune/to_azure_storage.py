# python3 to_azure_storage.py image-depth-models

'''Takes torch data from local machine and stores them on azure'''
import os
import sys
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def write_to_storage(
                        container_name,
                        blob_name
                    ):
    """
    pip install azure-storage-blob
    
    Get storage account and storage key:
        1) Go to Azure
        2) Storage accounts
        3) Click on our storage account name
        4) Get storage key
            i) Security + networking
            ii) Access keys
            iii) Choose either of the two keys

    In terminal type the following:
    export AZURE_STORAGE_ACCOUNT='Enter storage account name'
    export AZURE_STORAGE_KEY='Enter storage account key'
    """

    # Create a blob service client
    blob_service_client = \
        BlobServiceClient.from_connection_string(connection_string)

    container_lower = container_name.replace('_','-')
    container_lower = container_lower.lower()
    # Create the container client if the container doesn't already exist
    try:
        container_client = blob_service_client.create_container(container_lower)
    except Exception:
        print(f"Container {container_name} already exists or cannot be created.")

    # Create a blob client
    blob_client = blob_service_client.get_blob_client(container=container_lower, blob=blob_name)
    print("Blob client created.")

    local_file_path = f'{os.getcwd()}/{container_name}/{blob_name}'
    # Upload the file
    print(f"Uploading {blob_name} to {container_name} in Azure Blob Storage.")
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data)

    print(f"File {local_file_path} uploaded to {blob_name} in Azure Blob Storage.")


########################################
#      Create storage environment
########################################

# Azure storage account credentials
acc_name = os.getenv('AZURE_STORAGE_ACCOUNT')
acc_key = os.getenv('AZURE_STORAGE_KEY')

connection_string = f"DefaultEndpointsProtocol=https;" \
                    f"AccountName={acc_name};" \
                    f"AccountKey={acc_key};" \
                    f"EndpointSuffix=core.windows.net"

path = os.getcwd()
container = "image-depth-models" # e.g. "CNN_09-15" --> The name of the directory is the container name as well
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container)
all_blobs = container_client.list_blobs()
all_blobs = set([blob.name for blob in all_blobs])
path = f'{path}/{container}'
print(f'This is the path: {path}')
for blob in os.listdir(path):
    if blob not in all_blobs:
        write_to_storage(container, blob)