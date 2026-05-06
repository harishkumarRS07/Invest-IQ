#!/usr/bin/env python
"""Quick test script to check batch signals"""
import requests
import json

API_URL = "http://localhost:8000/api/v1"

# Test health
print("Testing health endpoint...")
try:
    resp = requests.get(f"{API_URL}/health")
    print(f"Health: {resp.status_code} -> {resp.json()}\n")
except Exception as e:
    print(f"Health failed: {e}\n")

# Try login first
print("Attempting login...")
resp = requests.post(f"{API_URL}/auth/login", json={
    "email": "test@test.com",
    "password": "test123"
})

if resp.status_code == 200:
    token = resp.json().get("token")
    print(f"Login successful, got token\n")
else:
    # Try register
    print("Login failed, attempting register...")
    resp = requests.post(f"{API_URL}/auth/register", json={
        "email": "test@test.com",
        "password": "test123",
        "name": "Test User"
    })
    if resp.status_code in [200, 201]:
        token = resp.json().get("token")
        print(f"Registered successfully, got token\n")
    else:
        print(f"Auth failed: {resp.status_code} -> {resp.json()}\n")
        exit(1)

# Test batch signals
print("Testing batch signals endpoint...")
headers = {"Authorization": f"Bearer {token}"}
stocks = ["HDFCBANK", "INFY", "TCS", "RELIANCE", "ICICIBANK"]

resp = requests.post(f"{API_URL}/signals/batch", 
    json={"symbols": stocks},
    headers=headers
)

print(f"Status: {resp.status_code}")
print(f"\nResponse:\n{json.dumps(resp.json(), indent=2)}")

if resp.status_code == 200:
    data = resp.json()
    signals = data.get("signals", [])
    print(f"\n{'='*60}")
    print(f"RESULTS: Got {len(signals)} signals")
    print(f"{'='*60}")
    for signal in signals:
        ticker = signal.get('symbol') or signal.get('ticker', 'N/A')
        change = signal.get('pct_change', 0)
        print(f"{ticker:12} | Signal: {signal['signal']:6} | Change: {change:+.2f}% | Confidence: {signal['signal_confidence']:.2f}")
else:
    print(f"\nError: {resp.text}")
