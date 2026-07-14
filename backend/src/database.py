from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from src.config import settings

# For SQLite, allow multi-threading (since FastAPI is async)
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    source = Column(String, nullable=True) # Paste, URL, PDF, TXT
    content = Column(Text, nullable=False)
    
    # Categorization
    category = Column(String, nullable=False)
    category_confidence = Column(Float, nullable=False)
    category_distribution = Column(Text, nullable=False) # JSON string of category distributions
    
    # Bias Detection
    bias = Column(String, nullable=False)
    bias_confidence = Column(Float, nullable=False)
    bias_distribution = Column(Text, nullable=False) # JSON string of bias distributions
    bias_score = Column(Float, nullable=False)
    bias_interpretation = Column(String, nullable=False)
    neutral_percent = Column(Float, nullable=False)
    biased_percent = Column(Float, nullable=False)
    
    # Explainability
    important_words = Column(Text, nullable=False) # JSON string of words and weights
    explanation = Column(Text, nullable=False)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
