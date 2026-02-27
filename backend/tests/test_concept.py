
from transformers import pipeline
import torch

def test_concept():
    print("Testing Sentiment Concept...")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device, top_k=None)
    
    examples = [
        "Apple reports record-breaking quarterly revenue, exceeding expectations.", 
        "Tech stocks tumble as inflation fears grip the market.",
        "The Federal Reserve kept interest rates unchanged today.",
        "Company X faces lawsuit over patent infringement, stock drops 5%.",
        "New product launch receives mixed reviews from critics.",
        "Amazon.com Inc. (AMZN) is firing hundreds of employees in its cloud computing division Amazon Web Services (AWS) as part of a strategic shift. The move comes as AWS sales growth has slowed in recent quarters. However, the company remains optimistic about the long-term prospects of cloud computing and artificial intelligence."
    ]
    
    for text in examples:
        results = pipe(text)
        # results is a list of list of dicts (because top_k=None)
        # [[{'label': 'positive', 'score': 0.9}, {'label': 'negative', 'score': 0.05}, ...]]
        
        scores = {item['label']: item['score'] for item in results[0]}
        
        pos = scores.get('positive', 0)
        neg = scores.get('negative', 0)
        neu = scores.get('neutral', 0)
        
        compound = pos - neg
        
        print(f"\nText: {text[:50]}...")
        print(f"Pos: {pos:.4f}, Neg: {neg:.4f}, Neu: {neu:.4f}")
        print(f"Compound Score: {compound:.4f}")

if __name__ == "__main__":
    test_concept()
