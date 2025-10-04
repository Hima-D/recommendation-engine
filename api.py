from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from data import UserActivitySimulator  # Import as needed
from model import RecommendationModel

app = FastAPI(title="Modern Recommendation Engine API")

# Global instances (in prod: Use dependency injection or load from DB)
sim = UserActivitySimulator(num_users=100, num_items=50)
ratings_df = sim.generate_ratings()
rec_model = RecommendationModel()
rec_model.fit(ratings_df)

class UserRequest(BaseModel):
    user_id: int
    n: int = 5

@app.post("/recommend")
async def get_recommendations(request: UserRequest):
    """Real-time recommendation endpoint."""
    recs = rec_model.recommend(request.user_id, request.n)
    return {
        "user_id": request.user_id,
        "recommendations": recs,
        "explanation": "Based on similar user activities on the website."
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_trained": rec_model.user_item_matrix is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)