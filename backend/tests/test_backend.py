import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app
from src.database import Base, get_db, PredictionRecord
from src.utils.extractor import clean_text

# Setup shared memory database connection for tests to persist tables
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
# Keep connection open for the entire test lifecycle
connection = engine.connect()
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(autouse=True, scope="module")
def run_once_for_module():
    # Create tables once for the module using the persistent connection
    Base.metadata.create_all(bind=connection)
    yield
    Base.metadata.drop_all(bind=connection)
    connection.close()

@pytest.fixture(autouse=True)
def run_around_tests():
    # Clean/Reset tables or wrap in transaction if needed, but for simple tests
    # just yield to let tests run in the shared database
    yield

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_clean_text():
    raw_html = "<html><body><h1>Hello World</h1><p>Check out https://google.com for more info.</p></body></html>"
    cleaned = clean_text(raw_html)
    # Tags removed, url removed, whitespace normalized
    assert "Hello World" in cleaned
    assert "https://google.com" not in cleaned
    assert "<html>" not in cleaned

def test_analyze_post_empty():
    response = client.post("/analyze", json={})
    assert response.status_code == 400

def test_history_empty():
    response = client.get("/history")
    assert response.status_code == 200
    assert len(response.json()) == 0
