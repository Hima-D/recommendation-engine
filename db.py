from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import pandas as pd
import os
import numpy as np

# --- Environment Config ---
DB_URL = os.getenv('DB_URL', 'sqlite:///ratings.db')
echo_mode = os.getenv('SQL_ECHO', 'False').lower() == 'true'

# --- SQLAlchemy Setup ---
Base = declarative_base()
engine = create_engine(DB_URL, echo=echo_mode)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Database Models ---

class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_user_item', 'user_id', 'item_id'),
    )
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item_id = Column(Integer, index=True)
    rating = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_type = Column(String(20), default='casual')
    session_id = Column(String(50))

class ItemMetadata(Base):
    __tablename__ = "item_metadata"
    __table_args__ = (Index('idx_item_id', 'item_id'),)
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, unique=True, index=True)
    item_name = Column(String(255))
    category = Column(String(100))
    price = Column(Float)
    description = Column(String(500))
    in_stock = Column(Boolean, default=True)
    features = Column(String(1000))  # Store numpy array as JSON-like string

# --- Create Tables ---
def init_db():
    """Drop and recreate tables to ensure schema consistency."""
    Base.metadata.drop_all(bind=engine)  # Drop existing tables
    Base.metadata.create_all(bind=engine)  # Create tables with current schema
    print("Database tables initialized.")

# Initialize database schema
init_db()

# --- Session Dependency (for FastAPI) ---
def get_db_session():
    """Yield DB session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- DataFrame to DB (Upsert) ---
def df_to_db(df, session, table_name: str = 'item_metadata') -> None:
    """
    Upsert DataFrame rows into the specified table with session handling.
    Supports 'ratings' and 'item_metadata'.
    """
    try:
        if table_name == 'ratings':
            model = Rating
            unique_keys = ['user_id', 'item_id']
        elif table_name == 'item_metadata':
            model = ItemMetadata
            unique_keys = ['item_id']
            # Convert features to string for storage
            if 'features' in df.columns:
                df['features'] = df['features'].apply(lambda x: str(list(x)) if isinstance(x, np.ndarray) else str(x))
        else:
            raise ValueError(f"Unsupported table: {table_name}")

        for _, row in df.iterrows():
            query = session.query(model)
            for key in unique_keys:
                if key in row:
                    query = query.filter(getattr(model, key) == row[key])
            existing = query.first()

            if existing:
                # Update existing record
                for col in row.index:
                    if hasattr(existing, col) and col != 'id':
                        setattr(existing, col, row[col])
            else:
                # Insert new record
                new_data = {col: row[col] for col in row.index if hasattr(model, col)}
                instance = model(**new_data)
                session.add(instance)

        session.commit()
        print(f"Upserted {len(df)} rows to '{table_name}'.")
    except IntegrityError as e:
        session.rollback()
        raise RuntimeError(f"Integrity error on upsert: {e}")
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"DB operation failed: {e}")

# --- Batch Insert (Fast) ---
def df_to_db_batch(df: pd.DataFrame, session, table_name: str = 'ratings', if_exists: str = 'append') -> None:
    """
    Efficient batch insert into the specified table.
    Uses pandas' to_sql with SQLAlchemy.
    """
    try:
        if table_name == 'item_metadata' and 'features' in df.columns:
            df['features'] = df['features'].apply(lambda x: str(list(x)) if isinstance(x, np.ndarray) else str(x))

        df.to_sql(table_name, session.bind, if_exists=if_exists, index=False, method='multi')
        print(f"Batched {len(df)} rows to '{table_name}'.")
    except Exception as e:
        raise RuntimeError(f"Batch insert failed: {e}")

# --- Load Table to DataFrame ---
def db_to_df(session, table_name: str = 'ratings') -> pd.DataFrame:
    """
    Load entire table into a DataFrame.
    Optionally converts fields (e.g., JSON to np.array).
    Also performs cleanup for old ratings.
    """
    try:
        if table_name == 'ratings':
            stmt = session.query(Rating).statement
        elif table_name == 'item_metadata':
            stmt = session.query(ItemMetadata).statement
        else:
            raise ValueError(f"Unsupported table: {table_name}")

        df = pd.read_sql(stmt, session.bind)

        # Deserialize feature vectors from string back to np.array
        if table_name == 'item_metadata' and 'features' in df.columns:
            df['features'] = df['features'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else (x if isinstance(x, np.ndarray) else np.zeros(16))
            )

        # Optional cleanup for ratings
        if table_name == 'ratings':
            cutoff = datetime.utcnow() - pd.Timedelta(days=365)
            df = df[df['timestamp'] >= cutoff]
            # Prune old data from DB too
            session.query(Rating).filter(Rating.timestamp < cutoff).delete()
            session.commit()

        return df
    except Exception as e:
        raise RuntimeError(f"Load to DF failed: {e}")

# --- Manual Cleanup for Old Ratings ---
def cleanup_old_data(session, days: int = 365) -> int:
    """
    Delete old ratings from DB.
    Returns the number of deleted rows.
    """
    try:
        cutoff = datetime.utcnow() - pd.Timedelta(days=days)
        deleted = session.query(Rating).filter(Rating.timestamp < cutoff).delete()
        session.commit()
        print(f"Cleaned up {deleted} old ratings.")
        return deleted
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Cleanup failed: {e}")