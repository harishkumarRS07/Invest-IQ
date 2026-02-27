import sys
import os

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import torch
from backend.core.logging import logger
from typing import List, Dict, Union
import numpy as np

class SentimentAnalyzer:
    """
    Analyzes financial text sentiment using FinBERT.
    """
    def __init__(self):
        # Check for GPU
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing FinBERT on device: {'GPU' if device == 0 else 'CPU'}")
        
        try:
            # We use top_k=None to get all scores (positive, negative, neutral)
            self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device, top_k=None)
            self.tokenizer = self.pipe.tokenizer
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.pipe = None
            self.tokenizer = None

    def _process_single_text(self, text: str) -> float:
        """
        Process a single text string, handling chunking if necessary.
        Returns a compound score: P(positive) - P(negative).
        """
        if not text or not text.strip():
            return 0.0
            
        try:
            # Tokenize
            tokens = self.tokenizer(text, return_tensors='pt', truncation=False, padding=False)
            input_ids = tokens['input_ids'][0]
            
            # Chunking settings
            max_len = 512
            stride = 256
            
            if len(input_ids) <= max_len:
                # Process normally
                chunks = [text]
            else:
                # Create overlapping chunks
                # Note: Decoding tokens back to text might assume the tokenizer can handle it perfectly.
                # Alternatively, we can pass input_ids to the model if using model directly, 
                # but valid inputs for pipeline are usually strings.
                # So we will slide window over input_ids and decode back to string chunks.
                chunks = []
                for i in range(0, len(input_ids), stride):
                    chunk_ids = input_ids[i : i + max_len]
                    if len(chunk_ids) < 10: # Skip very small chunks at the end
                        continue
                    # Skip special tokens if in middle? encode adds them. decode might not handle overlap perfectly if we split mid-sentence.
                    # Ideally we use stride on tokens.
                    chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    if i + max_len >= len(input_ids):
                         break

            # Predict on chunks
            # results will be a list of lists (one list per chunk, containing the scores)
            # [[{'label': 'positive', 'score': 0.9}, ...], ...]
            results = self.pipe(chunks, padding=True, truncation=True, max_length=512)
            
            avg_pos = 0.0
            avg_neg = 0.0
            
            for chunk_res in results:
                # chunk_res is list of dicts: [{'label': 'positive', 'score': X}, ...]
                scores = {item['label']: item['score'] for item in chunk_res}
                avg_pos += scores.get('positive', 0.0)
                avg_neg += scores.get('negative', 0.0)
            
            n_chunks = len(chunks)
            if n_chunks > 0:
                avg_pos /= n_chunks
                avg_neg /= n_chunks
                
            return avg_pos - avg_neg

        except Exception as e:
            logger.error(f"Error processing text chunk: {e}")
            return 0.0

    def analyze(self, text: Union[str, List[str]]) -> float:
        """
        Analyze sentiment of text(s).
        Returns an average score between -1.0 (Negative) and 1.0 (Positive).
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not installed. Sentinel Analysis disabled.")
            return 0.0

        if not self.pipe:
            logger.warning("FinBERT not initialized, returning neutral score.")
            return 0.0

        if isinstance(text, str):
            text = [text]

        total_score = 0.0
        count = 0
        
        for t in text:
            score = self._process_single_text(t)
            total_score += score
            count += 1
            
        if count == 0:
            return 0.0
            
        return total_score / count

sentiment_analyzer = SentimentAnalyzer()
