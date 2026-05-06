#!/bin/bash
# Script to extract ngrok public URL and update frontend config

CONFIG_FILE="./InvestIQ-App/config/api.js"

echo "Fetching ngrok public URL..."

# Get ngrok tunnels API response
RESPONSE=$(curl -s http://127.0.0.1:4040/api/tunnels)

if [ -z "$RESPONSE" ]; then
    echo "Error: Could not connect to ngrok API"
    echo "Make sure ngrok is running: ngrok http 8000"
    exit 1
fi

# Extract HTTPS URL
PUBLIC_URL=$(echo "$RESPONSE" | grep -o '"public_url":"[^"]*https[^"]*' | head -1 | cut -d'"' -f4)

if [ -z "$PUBLIC_URL" ]; then
    echo "No HTTPS tunnel found"
    exit 1
fi

echo "Found ngrok URL: $PUBLIC_URL"

API_URL="$PUBLIC_URL/api/v1"

# Update config file (macOS compatible)
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s|export const API_NGROK_URL = '[^']*';|export const API_NGROK_URL = '$API_URL';|g" "$CONFIG_FILE"
else
    sed -i "s|export const API_NGROK_URL = '[^']*';|export const API_NGROK_URL = '$API_URL';|g" "$CONFIG_FILE"
fi

echo "Updated $CONFIG_FILE"
echo "API URL is now: $API_URL"
