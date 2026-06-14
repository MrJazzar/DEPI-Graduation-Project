"""
Person State Module
===================
Holds the per-person tracking state for multi-person monitoring.
Each detected person gets their own PersonState instance that tracks
their identity, focus voting, and liveness independently.
"""

from collections import deque
from models.liveness_detector import LivenessDetector


class PersonState:
    """
    Encapsulates all tracking state for a single monitored person.

    Each person has:
        - A name (from face recognition).
        - A vote buffer for majority-vote focus classification.
        - Their own LivenessDetector instance.
        - An unknown_streak counter for face ID stability.

    Args:
        name:               Initial name (default 'Unknown').
        vote_buffer_size:   Number of samples in the majority vote window (default 3).
    """

    def __init__(self, name: str = "Unknown", vote_buffer_size: int = 3):
        self.name = name
        self.vote_buffer: deque = deque(maxlen=vote_buffer_size)
        self.liveness = LivenessDetector()
        self.unknown_streak = 0
        self.last_voted_label = "waiting"
        self.liveness_status = "Checking..."
        self.last_importances: dict = {}
        self.last_seen: float = 0.0  # timestamp of last detection
        self.bbox: tuple = (0, 0, 0, 0)  # (x1, y1, x2, y2) bounding box

    def majority_vote(self) -> str:
        """Return the most common label in the vote buffer."""
        if not self.vote_buffer:
            return "waiting"
        focused_votes = sum(1 for v in self.vote_buffer if v == "focused")
        return "focused" if focused_votes > len(self.vote_buffer) / 2 else "distracted"

    def update_name(self, new_name: str) -> None:
        """
        Update identity with a stability filter.
        Only resets to 'Unknown' after 5 consecutive failures.
        """
        if new_name != "Unknown":
            self.name = new_name
            self.unknown_streak = 0
        else:
            self.unknown_streak += 1
            if self.unknown_streak > 5:
                self.name = "Unknown"

    @property
    def display_name(self) -> str:
        return self.name.capitalize()
