# src/utils/metrics.py
from rouge import Rouge

class RougeEvaluator:
    def __init__(self):
        """
        Initialize ROUGE scorer.
        """
        self.rouge = Rouge()

    def compute_rouge(self, predictions, references):
        """
        Compute ROUGE scores for a batch of predictions and references.
        Args:
            predictions (list of str): List of predicted summaries.
            references (list of str): List of reference summaries.
        Returns:
            dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        scores = self.rouge.get_scores(predictions, references, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
