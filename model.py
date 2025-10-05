import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from typing import List, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import datetime

class NeuralCF(nn.Module):
    """Enhanced NCF with Transformer for sequences and dropout."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(100, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, positions=None):
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        x = u_emb * i_emb
        if positions is not None:
            pos_emb = self.position_embedding(positions)
            x += pos_emb
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) if x.dim() > 2 else x
        x = self.fc(x).squeeze()
        return self.sigmoid(x)

class RecommendationModel:
    """Full hybrid with MMR diversity, LLM augmentation, debiasing, cascade ranking."""
    def __init__(self, num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.ncf_model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.user_item_matrix = None
        self.user_similarities = None
        self.item_features = None
        self.content_sim = None
        self.item_metadata = None
        self.llm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.llm_model = AutoModel.from_pretrained("distilbert-base-uncased")

    def fit(self, ratings_df: pd.DataFrame, item_metadata: pd.DataFrame, epochs: int = 50):
        self.item_metadata = item_metadata.sort_values('item_id')
        if ratings_df.duplicated(subset=['user_id', 'item_id']).any():
            ratings_df = ratings_df.groupby(['user_id', 'item_id']).agg({
                'rating': 'mean',
                'timestamp': 'last',
                'user_type': 'first',
                'session_id': 'first'
            }).reset_index()
            print("Aggregated duplicates in ratings_df.")

        # Validate and filter user and item IDs
        valid_ratings_df = ratings_df[
            (ratings_df['user_id'] >= 0) & (ratings_df['user_id'] < self.num_users) &
            (ratings_df['item_id'] >= 0) & (ratings_df['item_id'] < self.num_items)
        ]
        if len(valid_ratings_df) < len(ratings_df):
            print(f"Filtered out {len(ratings_df) - len(valid_ratings_df)} rows with invalid user/item IDs")
        ratings_df = valid_ratings_df
        if ratings_df.empty:
            raise ValueError("No valid ratings data after filtering")

        ratings_df['recency_weight'] = ratings_df.groupby('user_id')['timestamp'].transform(
            lambda x: np.exp(-(x.max() - x).dt.days / 60)
        )
        ratings_df['weighted_rating'] = ratings_df['rating'] * ratings_df['recency_weight']

        unique_users = ratings_df['user_id'].nunique()
        unique_items = ratings_df['item_id'].nunique()
        self.ncf_model = NeuralCF(unique_users, unique_items)
        self.optimizer = torch.optim.Adam(self.ncf_model.parameters(), lr=0.001)

        grouped = ratings_df.sort_values('timestamp').groupby('user_id')
        max_seq_len = 50
        users_seq, items_seq, positions_seq, ratings_seq = [], [], [], []
        for _, group in grouped:
            seq_items = group['item_id'].values[-max_seq_len:]
            seq_ratings = (group['weighted_rating'].values[-max_seq_len:] / 5.0).astype(np.float32)
            seq_positions = np.arange(len(seq_items)) % 100
            pad_len = max_seq_len - len(seq_items)
            if pad_len > 0:
                seq_items = np.pad(seq_items, (0, pad_len), mode='constant', constant_values=0)
                seq_ratings = np.pad(seq_ratings, (0, pad_len), mode='constant', constant_values=0)
                seq_positions = np.pad(seq_positions, (0, pad_len), mode='constant', constant_values=0)
            users_seq.append(np.full(max_seq_len, group['user_id'].iloc[0]))
            items_seq.append(seq_items)
            positions_seq.append(seq_positions)
            ratings_seq.append(seq_ratings)

        users = torch.tensor(np.stack(users_seq), dtype=torch.long)
        items = torch.tensor(np.stack(items_seq), dtype=torch.long)
        positions = torch.tensor(np.stack(positions_seq), dtype=torch.long)
        ratings = torch.tensor(np.stack(ratings_seq), dtype=torch.float32)

        self.ncf_model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            preds = self.ncf_model(users.view(-1), items.view(-1), positions.view(-1))
            item_pop = ratings_df['item_id'].value_counts(normalize=True)
            propensity = torch.tensor([item_pop.get(i.item(), 1e-6) for i in items.view(-1)], dtype=torch.float32)
            loss = (self.criterion(preds, ratings.view(-1)) / propensity).mean()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.user_item_matrix = ratings_df.pivot(
            index='user_id', columns='item_id', values='weighted_rating'
        ).fillna(0)
        user_matrix = self.user_item_matrix.values
        self.user_similarities = pd.DataFrame(
            cosine_similarity(user_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        if 'features' in item_metadata.columns:
            features = item_metadata['features'].values
            self.item_features = np.stack([
                x if isinstance(x, np.ndarray) and x.shape == (16,) else np.zeros(16)
                for x in features
            ])
            descriptions = item_metadata['description'].values
            inputs = self.llm_tokenizer(descriptions.tolist(), padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.llm_model(**inputs)
                llm_features = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embedding
            self.item_features = np.hstack([self.item_features, llm_features])
            if not np.issubdtype(self.item_features.dtype, np.number):
                raise ValueError("Item features must be numeric")
            self.content_sim = cosine_similarity(self.item_features)
        else:
            self.content_sim = np.eye(self.num_items)
            self.item_features = np.zeros((self.num_items, 768))  # Adjust for DistilBERT embedding size

        print("Full hybrid model fitted.")

    def simulate_users(self, num_sim: int = 100):
        """Agentic simulation for lifelong training."""
        sim_df = pd.DataFrame({
            'user_id': np.random.randint(0, self.num_users, num_sim),
            'item_id': np.random.randint(0, self.num_items, num_sim),
            'rating': np.random.uniform(1, 5, num_sim),
            'timestamp': [datetime.datetime.now() for _ in range(num_sim)]
        })
        self.fit(sim_df, self.item_metadata, epochs=5)

    def _ncf_score(self, user_id: int, item_id: int) -> float:
        if self.ncf_model is None:
            return 0.0
        self.ncf_model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long)
            item_tensor = torch.tensor([item_id], dtype=torch.long)
            pred = self.ncf_model(user_tensor, item_tensor).item()
        return pred

    def _mmr_diversity(self, candidates: List[int], selected: List[int], user_id: int, n: int, lambda_div: float = 0.5) -> List[int]:
        """Iterative MMR for diversity."""
        selected = selected[:]
        for _ in range(min(n, len(candidates))):
            if not candidates:
                break
            scores = {}
            for cand in candidates:
                rel_score = self._ncf_score(user_id, cand)
                if selected:
                    div_scores = [1 - self.content_sim[cand, s] for s in selected]
                    div_score = np.mean(div_scores) if div_scores else 0
                else:
                    div_score = 0
                scores[cand] = lambda_div * rel_score + (1 - lambda_div) * div_score
            if not scores:
                break
            next_item = max(scores, key=scores.get)
            selected.append(next_item)
            candidates.remove(next_item)
        return selected

    def recommend(self, user_id: int, n: int = 5, alpha_cf: float = 0.4, alpha_ncf: float = 0.3, alpha_content: float = 0.3, diversity: bool = True) -> Tuple[List[int], str]:
        if self.user_item_matrix is None:
            raise ValueError("Model not fitted")
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found, using cold-start strategy")
            liked_items = [np.random.choice(self.user_item_matrix.columns)]
        else:
            liked_items = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index.tolist()
            if not liked_items:
                liked_items = [np.random.choice(self.user_item_matrix.columns)]
                print(f"User {user_id} has no interactions, using random item: {liked_items}")

        scores = defaultdict(float)
        all_items = list(self.user_item_matrix.columns)
        interacted = set(liked_items)
        print(f"User {user_id} interacted items: {interacted}")

        # Cascade Pre-ranking
        all_items_tensor = torch.tensor(all_items, dtype=torch.long)
        user_ids_pre = torch.full((len(all_items),), user_id, dtype=torch.long)
        with torch.no_grad():
            pre_scores = self.ncf_model(user_ids_pre, all_items_tensor).squeeze().numpy()
        pre_candidates = np.argsort(pre_scores)[-100:]
        candidate_items = [all_items[i] for i in pre_candidates if all_items[i] not in interacted]
        print(f"Pre-ranking candidates: {len(candidate_items)} items")

        if not candidate_items:
            candidate_items = list(set(all_items) - interacted)
            print(f"Fallback to {len(candidate_items)} non-interacted items")

        # Fallback: Popularity-based if still empty
        if not candidate_items:
            item_pop = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
            candidate_items = item_pop.index[:100].tolist()
            print(f"Fallback to popularity-based: {len(candidate_items)} items")

        # Full scoring
        for item in candidate_items:
            similar_scores = self.user_similarities[user_id].sort_values(ascending=False)[1:15]
            for sim_user, sim_score in similar_scores.items():
                user_ratings = self.user_item_matrix.loc[sim_user]
                if item in user_ratings and user_ratings[item] > 0:
                    scores[item] += alpha_cf * sim_score * user_ratings[item]
            scores[item] += alpha_ncf * self._ncf_score(user_id, item) * 5
            liked_indices = [l for l in liked_items if l < self.content_sim.shape[0]]
            content_sim_mean = np.mean([self.content_sim[l, item] for l in liked_indices]) if liked_indices else 0
            scores[item] += alpha_content * content_sim_mean * 5

        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n*3]
        candidate_items = [int(item) for item, _ in candidates]
        print(f"Scored candidates: {len(candidate_items)} items")

        if diversity and len(candidate_items) > 1:
            recs = self._mmr_diversity(candidate_items, [], user_id, n)
        else:
            recs = candidate_items[:n]

        if not recs:
            recs = list(set(all_items) - interacted)[:n]
            print(f"Fallback to random non-interacted items: {recs}")

        explanation = "Hybrid recs from CF, neural predictions, and content sim."
        if diversity:
            explanation += " (diversified with MMR)."
        if recs and self.item_metadata is not None:
            top_cat = self.item_metadata.loc[recs[0], 'category'] if recs[0] in self.item_metadata.index else "unknown"
            explanation += f" Focus on {top_cat}."

        print(f"Final recommendations for user {user_id}: {recs}")
        return recs, explanation