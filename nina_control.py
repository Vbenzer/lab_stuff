import requests
import time

# Base URL for the N.I.N.A. API
API_BASE_URL = "http://localhost:1888/v2/api"  # Replace with your N.I.N.A. API host and port if needed

# Optional: If the API requires an API key, include it here
HEADERS = {
#    "Authorization": "",  # Replace with your actual API key
    "Content-Type": "application/json"
}

def connect_equipment():
    """Connect all equipment in N.I.N.A."""
    url = f"{API_BASE_URL}/equipment/camera/connect"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        print("Equipment connected successfully!")
    else:
        print(f"Failed to connect equipment: {response.status_code} - {response.text}")

def start_sequence(sequence_file):
    """Start a sequence in N.I.N.A."""
    url = f"http://localhost:1888/v2/api/sequence/start"
    data = {
        "path": sequence_file  # Replace with the path to your sequence file
    }
    response = requests.get(url,data)
    if response.status_code == 200:
        print("Sequence started successfully!")
    else:
        print(f"Failed to start sequence: {response.status_code} - {response.text}")

# Example usage:
connect_equipment()
start_sequence("C:/Users/stuermer/Documents/N.I.N.A/Test.json")  # Update with the correct file path
time.sleep(10)
start_sequence("C:/Users/stuermer/Documents/N.I.N.A/Test2.json")
#response = requests.get('http://localhost:1888/v2/api/sequence/start')
#