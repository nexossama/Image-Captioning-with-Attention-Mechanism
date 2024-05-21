import os
import requests
import gdown

def download_file_if_not_exists(url, destination):
    """
    Downloads a file from a URL to a destination if it does not already exist.
    
    Parameters:
    - url (str): The URL of the file to download.
    - destination (str): The path to save the downloaded file.
    
    Returns:
    - None
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    
    try:
        if 'drive.google.com' in url:
            gdown.download(url, destination, quiet=False)
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check if the request was successful

            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
            print(f"File downloaded successfully and saved to {destination}")
    except Exception as e:
        print(f"Failed to download file. Error: {e}")

def download_model():
    url = 'https://drive.google.com/uc?/export=download&id=1K5nICnO0SnK6HOTqTuSvZsArpbmRf9jr'
    src_path=os.path.dirname(os.path.dirname(__file__))
    destination_path = os.path.join(src_path,"assets","attention_model_state_200_long_memory.pth")
    download_file_if_not_exists(url, destination_path)