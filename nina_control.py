import requests
import time

# Configuration
NINA_API_URL = "http://localhost:59590/api"  # Replace with the correct URL and port if needed
API_KEY = ""  # Add your API key if you set one


def connect_equipment():
    """Connect all equipment in N.I.N.A."""
    url = f"{NINA_API_URL}/equipment/connect"
    headers = {"X-Api-Key": API_KEY} if API_KEY else {}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print("Equipment connected successfully.")
    else:
        print(f"Failed to connect equipment: {response.status_code} - {response.text}")


def start_sequence():
    """Start the current sequence in N.I.N.A."""
    url = f"{NINA_API_URL}/sequence/start"
    headers = {"X-Api-Key": API_KEY} if API_KEY else {}
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print("Sequence started successfully.")
    else:
        print(f"Failed to start sequence: {response.status_code} - {response.text}")


def main():
    print("Connecting equipment...")
    connect_equipment()
    time.sleep(5)  # Wait for equipment to connect

    print("Starting sequence...")
    start_sequence()


if __name__ == "__main__":
    main()
