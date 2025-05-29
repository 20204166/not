import requests

url = "http://localhost:3000/api/notes/process"
payload = {
    "text_input": "Today we discussed the use of artificial intelligence in education and how it can help automate summarization of lecture notes."
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    assert "summary" in result["data"], "Missing 'summary' in response"
    print("✅ API returned summary successfully:", result["data"]["summary"])
except Exception as e:
    print("❌ API test failed:", e)
    exit(1)
