import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from typing import List
from collections import defaultdict

class RecommendationModel:
    """
    User-based Collaborative Filtering using Cosine Similarity.
    Modern extension: Integrate PyTorch for sequence models (e.g., SASRec).
    """
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarities = None

    def fit(self, ratings_df: pd.DataFrame):
        """Build matrix and compute user similarities."""
        self.user_item_matrix = ratings_df.pivot(
            index='user_id', columns='item_id', values='rating'
        ).fillna(0)
        # Cosine similarities
        self.user_similarities = pd.DataFrame(
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        for i in self.user_item_matrix.index:
            for j in self.user_item_matrix.index:
                if i == j:
                    self.user_similarities.loc[i, j] = 1.0
                else:
                    sim = 1 - cosine(
                        self.user_item_matrix.loc[i].values,
                        self.user_item_matrix.loc[j].values
                    )
                    self.user_similarities.loc[i, j] = sim
        print("Model fitted. User similarities computed.")

    def recommend(self, user_id: int, n: int = 5) -> List[int]:
        """Top-N recs from similar users."""
        if self.user_item_matrix is None:
            raise ValueError("Model not fitted")
        if user_id not in self.user_item_matrix.index:
            raise ValueError("User not found")
        
        # Top similar users
        similar_users = self.user_similarities[user_id].sort_values(
            ascending=False
        ).index[1:n*2]
        recommendations = defaultdict(float)
        
        for sim_user in similar_users:
            sim_score = self.user_similarities.loc[user_id, sim_user]
            liked_items = self.user_item_matrix.loc[sim_user][
                self.user_item_matrix.loc[sim_user] > 0
            ]
            not_liked = self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] == 0
            ].index
            for item in liked_items.index.intersection(not_liked):
                recommendations[item] += sim_score * liked_items[item]
        
        top_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        return [item for item, _ in top_items]