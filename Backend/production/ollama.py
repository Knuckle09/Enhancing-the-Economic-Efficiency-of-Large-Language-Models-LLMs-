import requests

OLLAMA_URL = "http://localhost:11434"

def is_ollama_running():
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3.2:3b", "prompt": "Ping", "stream": False},
            timeout=5
        )
        if response.status_code == 200:
            print("Ollama is running ✅")
            return True
        else:
            print(f"Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("Cannot connect to Ollama ❌")
        return False

is_ollama_running()
