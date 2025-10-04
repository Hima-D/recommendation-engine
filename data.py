import pandas as pd
import numpy as np
from typing import Dict

class UserActivitySimulator:
    """
    Simulates user website activity (e.g., views/clicks as ratings).
    In prod: Replace with Kafka consumer or DB query for real events.
    """
    def __init__(self, num_users: int = 100, num_items: int = 50):
        self.num_users = num_users
        self.num_items = num_items
        self.users = list(range(num_users))
        self.items = list(range(num_items))

    def generate_ratings(self) -> pd.DataFrame:
        """Generate sparse user-item interactions (ratings 1-5)."""
        data = []
        for user in self.users:
            num_interactions = np.random.poisson(5)
            interacted_items = np.random.choice(
                self.items, size=min(num_interactions, self.num_items), replace=False
            )
            for item in interacted_items:
                rating = np.random.randint(1, 6)
                data.append({"user_id": user, "item_id": item, "rating": rating})
        df = pd.DataFrame(data)
        return df

    def get_user_profile(self, user_id: int, df: pd.DataFrame) -> Dict:
        """Extract activity profile for a user."""
        profile = df[df["user_id"] == user_id]
        return profile.to_dict("records")