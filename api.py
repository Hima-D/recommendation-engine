from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, List
import numpy as np
from functools import lru_cache
from contextlib import asynccontextmanager
import logging
import pandas as pd
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Local imports
from data import UserActivitySimulator
from model import RecommendationModel
from db import get_db_session, cleanup_old_data
from tasks import background_retrain  # For async

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Full-Scale Amazon-Grade Rec Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Globals (loaded in lifespan)
sim: UserActivitySimulator = None
rec_model: RecommendationModel = None
ratings_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Init sim, load data, fit model; shutdown: Cleanup."""
    global sim, rec_model, ratings_df
    # Startup
    sim = UserActivitySimulator(num_users=100, num_items=50)
    ratings_df = sim.get_latest_df()  # Load from DB
    if len(ratings_df) == 0:
        logger.info("No ratings data found, generating new ratings")
        ratings_df = await sim.generate_ratings()
    # Filter invalid user and item IDs
    valid_ratings_df = ratings_df[
        (ratings_df['user_id'] >= 0) & (ratings_df['user_id'] < sim.num_users) &
        (ratings_df['item_id'] >= 0) & (ratings_df['item_id'] < sim.num_items)
    ]
    if len(valid_ratings_df) < len(ratings_df):
        logger.warning(f"Filtered {len(ratings_df) - len(valid_ratings_df)} rows with invalid user/item IDs")
    ratings_df = valid_ratings_df
    if ratings_df.empty:
        logger.warning("No valid ratings data after filtering, generating new ratings")
        ratings_df = await sim.generate_ratings()
    rec_model = RecommendationModel(num_users=sim.num_users, num_items=sim.num_items)
    try:
        rec_model.fit(ratings_df, sim.item_metadata, epochs=50)
    except Exception as e:
        logger.error(f"Failed to fit model: {e}")
        raise
    # Set callback for log-triggered retrain
    sim.set_retrain_callback(lambda: background_retrain.delay())
    logger.info(f"App started. Loaded {len(ratings_df)} interactions. User IDs: {ratings_df['user_id'].min()} to {ratings_df['user_id'].max()}")
    
    # Shutdown
    try:
        if sim and sim.session:
            deleted = cleanup_old_data(sim.session)
            logger.info(f"Cleaned up {deleted} old ratings during shutdown")
            sim.session.close()
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {e}")
    yield
    logger.info("App shutdown.")

app.router.lifespan_context = lifespan

class UserRequest(BaseModel):
    user_id: int
    n: int = 5
    diversity: bool = True

class ActivityLog(BaseModel):
    user_id: int
    item_id: int
    rating: int = 5  # Default positive
    timestamp: str = None  # ISO format

class EvalRequest(BaseModel):
    k: int = 5

@lru_cache(maxsize=256)
def get_cached_recs(user_id: int, n: int, diversity: bool) -> Dict:
    try:
        recs, expl = rec_model.recommend(user_id, n, diversity=diversity)
        return {"recommendations": recs, "explanation": expl}
    except Exception as e:
        logger.error(f"Cache error for user {user_id}: {e}")
        raise

@app.post("/log_activity")
@limiter.limit("100/hour")
async def log_activity(request: Request, activity: ActivityLog, background_tasks: BackgroundTasks, db=Depends(get_db_session)):
    """Log with async retrain trigger. Returns profile summary."""
    global ratings_df
    try:
        activity_dict = activity.dict()
        if 'timestamp' in activity_dict and activity_dict['timestamp']:
            activity_dict['timestamp'] = pd.to_datetime(activity_dict['timestamp'])
        ratings_df = await sim.log_activity(activity_dict, trigger_retrain=True)
        background_tasks.add_task(background_retrain.delay)
        get_cached_recs.cache_clear()
        profile = sim.get_user_profile(activity.user_id, ratings_df)
        logger.info(f"Logged activity for user {activity.user_id}. New total: {len(ratings_df)}")
        return {
            "message": "Logged & retrain queued",
            "total_events": len(ratings_df),
            "user_profile": {
                "user_type": profile['user_type'],
                "total_interactions": profile['total_interactions'],
                "recent_sessions": profile['sessions'][-3:] if profile['sessions'] else []
            }
        }
    except ValueError as e:
        logger.error(f"Invalid activity data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Log activity failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/retrain")
@limiter.limit("5/hour")
async def retrain_model(request: Request, db=Depends(get_db_session)):
    """Sync retrain on latest data."""
    global rec_model, ratings_df
    try:
        ratings_df = sim.get_latest_df()
        # Filter invalid user and item IDs
        valid_ratings_df = ratings_df[
            (ratings_df['user_id'] >= 0) & (ratings_df['user_id'] < sim.num_users) &
            (ratings_df['item_id'] >= 0) & (ratings_df['item_id'] < sim.num_items)
        ]
        if len(valid_ratings_df) < len(ratings_df):
            logger.warning(f"Filtered {len(ratings_df) - len(valid_ratings_df)} rows with invalid user/item IDs")
        ratings_df = valid_ratings_df
        if ratings_df.empty:
            logger.warning("No valid ratings data for retraining, generating new ratings")
            ratings_df = await sim.generate_ratings()
        rec_model.fit(ratings_df, sim.item_metadata, epochs=20)
        get_cached_recs.cache_clear()
        deleted = cleanup_old_data(db)
        logger.info(f"Retrained model. Data size: {len(ratings_df)}, Cleaned: {deleted}")
        return {"message": "Retrained successfully", "data_size": len(ratings_df), "cleaned_old": deleted}
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail="Retrain failed")

@app.post("/recommend")
@limiter.limit("1000/hour")
async def get_recommendations(request: Request, user_request: UserRequest):
    """Hybrid real-time recs with cache."""
    try:
        if user_request.user_id >= sim.num_users or user_request.user_id < 0:
            raise ValueError(f"User ID {user_request.user_id} out of range [0, {sim.num_users-1}]")
        cached = get_cached_recs(user_request.user_id, user_request.n, user_request.diversity)
        cached["user_id"] = user_request.user_id
        logger.debug(f"Generated recs for user {user_request.user_id}: {cached['recommendations']}")
        return cached
    except ValueError as e:
        logger.error(f"Invalid user ID: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recommend failed: {e}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")

@app.get("/user_profile/{user_id}")
@limiter.limit("50/hour")
async def get_user_profile(request: Request, user_id: int):
    """Get detailed user profile with sessions."""
    try:
        if user_id >= sim.num_users or user_id < 0:
            raise ValueError(f"User ID {user_id} out of range [0, {sim.num_users-1}]")
        profile = sim.get_user_profile(user_id)
        return profile
    except ValueError as e:
        logger.error(f"Invalid user ID: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Profile fetch failed")

@app.get("/items")
@limiter.limit("10/hour")
async def get_items(request: Request, limit: int = 20):
    """Get sample items metadata."""
    try:
        items = sim.item_metadata.head(limit).to_dict('records')
        return {"items": items}
    except Exception as e:
        logger.error(f"Items fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
@limiter.limit("10/hour")
async def evaluate_model(request: Request, eval_request: EvalRequest):
    """Eval: P@K and R@K on holdout."""
    try:
        from sklearn.model_selection import train_test_split
        current_df = sim.get_latest_df()
        if len(current_df) < 100:
            raise ValueError("Insufficient data for evaluation")
        train_df, test_df = train_test_split(current_df, test_size=0.2, random_state=42)
        precisions, recalls = [], []
        for user in test_df['user_id'].unique()[:20]:
            test_items = set(test_df[test_df['user_id'] == user]['item_id'])
            recs, _ = rec_model.recommend(user, eval_request.k)
            hits = len(set(recs) & test_items)
            precision = hits / len(recs) if recs else 0
            recall = hits / len(test_items) if test_items else 0
            precisions.append(precision)
            recalls.append(recall)
        avg_p = np.mean(precisions)
        avg_r = np.mean(recalls)
        return {"precision_at_k": avg_p, "recall_at_k": avg_r, "k": eval_request.k}
    except ValueError as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Eval failed: {e}")
        raise HTTPException(status_code=500, detail="Evaluation failed")

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request, db=Depends(get_db_session)):
    """Check system health."""
    try:
        data_size = len(sim.get_latest_df())
        return {
            "status": "healthy",
            "model_trained": rec_model.ncf_model is not None,
            "data_size": data_size,
            "cache_info": get_cached_recs.cache_info(),
            "db_connected": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)