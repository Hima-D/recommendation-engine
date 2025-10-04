# Recommendation Engine

A modular, modern recommendation system based on user website activity using FastAPI, Pandas, and SciPy.

## Setup
1. `pip install -r requirements.txt`
2. `python api.py`

## Usage
POST /recommend: `{"user_id": 0, "n": 5}`

## Modules
- `data.py`: Activity simulation.
- `model.py`: CF model.
- `api.py`: Serving.