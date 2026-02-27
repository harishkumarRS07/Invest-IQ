
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.features.sentiment import sentiment_analyzer

def test_sentiment():
    print("Testing Sentiment Analysis...")
    
    examples = [
        ("Apple reports record-breaking quarterly revenue, exceeding expectations.", 1.0), # Positive
        ("Tech stocks tumble as inflation fears grip the market.", -1.0), # Negative
        ("The Federal Reserve kept interest rates unchanged today.", 0.0), # Neutral
        ("Company X faces lawsuit over patent infringement, stock drops 5%.", -1.0), # Negative
        ("New product launch receives mixed reviews from critics.", 0.0), # Mixed/Neutral
        # Long text example
        ("Amazon.com Inc. (AMZN) is firing hundreds of employees in its cloud computing division Amazon Web Services (AWS) as part of a strategic shift. The move comes as AWS sales growth has slowed in recent quarters. However, the company remains optimistic about the long-term prospects of cloud computing and artificial intelligence.", -0.5) # Mixed/Negative leaning
    ]
    
    for text, expected_sign in examples:
        score = sentiment_analyzer.analyze(text)
        print(f"\nText: {text[:50]}...")
        print(f"Score: {score:.4f} (Expected sign: {expected_sign})")

if __name__ == "__main__":
    test_sentiment()
