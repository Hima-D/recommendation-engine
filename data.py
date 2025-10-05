import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable, List
import datetime
import os
import asyncio
from enum import Enum
from db import SessionLocal, df_to_db, df_to_db_batch, db_to_df, engine, Rating
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Env config for integration
DB_URL = os.getenv('DB_URL', 'sqlite:///ratings.db')
CLEANUP_DAYS = int(os.getenv('CLEANUP_DAYS', 365))

class UserType(Enum):
    """Realistic user segments."""
    NEW_USER = "new_user"      # <5 interactions
    CASUAL_BROWSER = "casual"  # 5-15
    FREQUENT_BUYER = "frequent" # 15+

class ItemCategory(Enum):
    """Amazon-like categories."""
    ELECTRONICS = "Electronics"
    BOOKS = "Books"
    CLOTHING = "Clothing"
    HOME = "Home & Garden"

class UserActivitySimulator:
    """
    Realistic simulator with sessions, user types, and enriched metadata.
    Integrates: DB persistence, model callbacks, API-friendly outputs.
    """
    def __init__(self, num_users: int = 100, num_items: int = 50, now: datetime.datetime = datetime.datetime(2025, 10, 5)):
        self.num_users = num_users
        self.num_items = num_items
        self.users = list(range(num_users))
        self.session = SessionLocal(bind=engine)
        self.items = self._generate_realistic_items(num_items)
        self.item_metadata = self._generate_item_metadata()
        self.now = now
        self.on_log_callback: Optional[Callable] = None
        logger.info(f"Initialized simulator with {num_users} users and {num_items} items")

    def _generate_realistic_items(self, num_items: int) -> List[str]:
        """Generate Amazon-like item names."""
        electronics = ["iPhone 15", "MacBook Pro", "AirPods Pro", "Samsung TV"]
        books = ["The Great Gatsby", "1984", "Sapiens", "Atomic Habits"]
        clothing = ["Levi's Jeans", "Nike Air Max", "Adidas Hoodie", "Zara Dress"]
        home = ["IKEA Lamp", "Dyson Vacuum", "Nest Thermostat", "KitchenAid Mixer"]
        all_items = electronics * (num_items//16 + 1) + books * (num_items//16 + 1) + \
                    clothing * (num_items//16 + 1) + home * (num_items//16 + 1)
        return all_items[:num_items]

    def _generate_item_metadata(self) -> pd.DataFrame:
        """Enriched metadata: name, category, price, desc snippet, features (16-dim)."""
        categories = list(ItemCategory)
        data = []
        for idx, item_name in enumerate(self.items):
            cat = np.random.choice(categories)
            price = np.random.uniform(10, 1000)
            desc = f"High-quality {cat.value.lower()}: {item_name} - Customer favorite."
            cat_onehot = np.zeros(len(categories))
            cat_onehot[categories.index(cat)] = 1
            features = np.concatenate([cat_onehot, [price/1000], np.random.rand(11)])  # 16-dim
            data.append({
                'item_id': idx,
                'item_name': item_name,
                'category': cat.value,
                'price': price,
                'description': desc,
                'in_stock': np.random.choice([True, False], p=[0.9, 0.1]),
                'features': features
            })
        df = pd.DataFrame(data)
        df_to_db(df, self.session, table_name='item_metadata')
        logger.info(f"Generated metadata for {len(df)} items")
        return df

    async def _cleanup_old_data(self):
        """Async cleanup: Remove interactions > CLEANUP_DAYS old."""
        cutoff = self.now - datetime.timedelta(days=CLEANUP_DAYS)
        try:
            deleted = self.session.query(Rating).filter(Rating.timestamp < cutoff).delete()
            self.session.commit()
            logger.info(f"Cleaned up {deleted} old ratings asynchronously")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Async cleanup failed: {e}")

    async def generate_ratings(self) -> pd.DataFrame:
        """Generate session-based interactions with user types."""
        data = []
        await self._cleanup_old_data()

        for user in self.users:
            user_type = np.random.choice(
                list(UserType),
                p=[0.3, 0.5, 0.2]
            )
            base_interactions = {'new_user': 3, 'casual': 10, 'frequent': 25}[user_type.value]
            num_sessions = {'new_user': 1, 'casual': 3, 'frequent': 8}[user_type.value]
            
            for session in range(num_sessions):
                session_start = self.now - datetime.timedelta(
                    days=np.random.exponential(90) * (session / num_sessions + 0.1)
                )
                num_in_session = np.random.poisson(base_interactions / num_sessions)
                interacted_items = np.random.choice(
                    range(self.num_items), size=min(num_in_session, self.num_items), replace=False
                )
                for item_id in interacted_items:
                    offset_min = np.random.uniform(0, 30)
                    timestamp = session_start + datetime.timedelta(minutes=offset_min)
                    rating_value = np.random.randint(1, 6) if user_type != UserType.NEW_USER else np.random.randint(3, 6)
                    data.append({
                        "user_id": user,
                        "item_id": item_id,
                        "rating": rating_value,
                        "timestamp": timestamp,
                        "user_type": user_type.value,
                        "session_id": f"{user}_{session}"
                    })
        
        df = pd.DataFrame(data).sort_values(['user_id', 'timestamp'])
        df_to_db_batch(df, self.session)
        logger.info(f"Generated {len(df)} realistic interactions for {self.num_users} users")
        return df

    def get_user_profile(self, user_id: int, df: Optional[pd.DataFrame] = None) -> Dict:
        """Get profile with sessions and type."""
        if df is None:
            df = db_to_df(self.session)
        profile = df[df["user_id"] == user_id].sort_values('timestamp')
        sessions = profile.groupby('session_id').agg({
            'item_id': 'nunique',
            'rating': 'mean',
            'timestamp': ['min', 'max']
        }).to_dict('records')
        return {
            'user_id': user_id,
            'interactions': profile.to_dict("records"),
            'sessions': sessions,
            'user_type': profile['user_type'].iloc[0] if not profile.empty else UserType.NEW_USER.value,
            'total_interactions': len(profile)
        }

    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """API-friendly: Return session summaries."""
        profile = self.get_user_profile(user_id)
        return profile['sessions']

    async def log_activity(self, activity: Dict, trigger_retrain: bool = True) -> pd.DataFrame:
        """
        Log with validation; optional retrain callback.
        Integrates with API: Validates rating, handles new items.
        """
        required = ['user_id', 'item_id']
        if not all(k in activity for k in required):
            raise ValueError("Missing required fields: user_id, item_id")
        if activity['user_id'] >= self.num_users or activity['user_id'] < 0:
            raise ValueError(f"User ID {activity['user_id']} out of range [0, {self.num_users-1}]")
        rating_value = activity.get('rating', 5)
        if not 1 <= rating_value <= 5:
            raise ValueError("Rating must be 1-5")
        if activity['item_id'] >= self.num_items:
            logger.info(f"Adding new item ID {activity['item_id']}")
            new_name = f"New Item {activity['item_id']}"
            self.items.append(new_name)
            categories = list(ItemCategory)
            cat = np.random.choice(categories)
            price = np.random.uniform(10, 1000)
            cat_onehot = np.zeros(len(categories))
            cat_onehot[categories.index(cat)] = 1
            new_features = np.concatenate([cat_onehot, [price/1000], np.random.rand(11)])  # 16-dim
            new_row = pd.DataFrame([{
                'item_id': activity['item_id'],
                'item_name': new_name,
                'category': cat.value,
                'price': price,
                'description': f"High-quality {cat.value.lower()}: {new_name} - Newly added item.",
                'in_stock': True,
                'features': new_features
            }])
            self.item_metadata = pd.concat([self.item_metadata, new_row], ignore_index=True)
            df_to_db(new_row, self.session, table_name='item_metadata')

        if 'timestamp' not in activity:
            activity['timestamp'] = self.now
        if 'user_type' not in activity:
            current_df = db_to_df(self.session)
            count = len(current_df[current_df['user_id'] == activity['user_id']])
            activity['user_type'] = 'new_user' if count < 5 else ('casual' if count < 15 else 'frequent')

        try:
            new_row = pd.DataFrame([{
                'user_id': activity['user_id'],
                'item_id': activity['item_id'],
                'rating': rating_value,
                'timestamp': pd.to_datetime(activity['timestamp']),
                'user_type': activity['user_type'],
                'session_id': activity.get('session_id', f"{activity['user_id']}_{pd.Timestamp.now().timestamp()}")
            }])
            df_to_db(new_row, self.session)
        except Exception as e:
            self.session.rollback()
            logger.error(f"DB insert failed: {e}")
            raise RuntimeError(f"DB insert failed: {e}")

        updated_df = db_to_df(self.session)
        
        if trigger_retrain and self.on_log_callback:
            self.on_log_callback()

        logger.info(f"Logged activity for user {activity['user_id']}. Total: {len(updated_df)}")
        return updated_df

    def set_retrain_callback(self, callback: Callable):
        """Hook for model integration."""
        self.on_log_callback = callback

    def get_latest_df(self) -> pd.DataFrame:
        """For model fit(): Get fresh data."""
        df = db_to_df(self.session)
        logger.info(f"Loaded {len(df)} ratings from database")
        return df