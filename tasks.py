from celery import Celery
import time  # Sim delay

app = Celery('tasks', broker='redis://localhost:6379/0')  # In prod: Redis

@app.task
def background_retrain():
    """Async retrain task."""
    time.sleep(5)  # Sim work
    print("Background retrain complete!")
    return "Retrained"