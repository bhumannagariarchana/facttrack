# FactTrack — AI News Bias & Categorizer System

FactTrack is a complete, production-ready, full-stack AI platform designed to analyze news articles, predict their category domain, detect political/editorial bias, and explain the predictions dynamically using NLP attributions.

The platform leverages fine-tuned **DeBERTa-v3** transformer model for bias detection and sequence classifications.

---

## Key Features

1. **News Categorization**: Predicts article domains across 7 categories:
   - Politics, Business, Sports, Technology, Entertainment, Health, Science
2. **Bias Detection**: Classifies political bias into **Left**, **Center**, or **Right** with confidence scores using fine-tuned **DeBERTa-v3**.
3. **Continuous Bias Score**: Calculates a continuous metrics from `0.00` (neutral) to `1.00` (partisan).
4. **Explainable AI (XAI)**: Displays token-level attributions and highlights words in real-time, explaining the model's classifications.
5. **Multi-source Inputs**: Supports copy-pasting text, entering URLs (auto scraped), or uploading PDF and TXT files.
6. **Retraining & TensorBoard Support**: Integrates full background fine-tuning pipeline for both category and bias classifiers with TensorBoard logger logging validation curves.

---

## Project Structure

```text
facttrack/
├── backend/
│   ├── src/
│   │   ├── api/          # Pydantic schemas
│   │   ├── ml/           # Model definitions & Training pipelines
│   │   │   ├── classifier.py
│   │   │   ├── explain.py
│   │   │   ├── train_category.py
│   │   │   └── train_bias.py
│   │   ├── utils/        # Parsers and scraping
│   │   ├── config.py     # Hyperparameters & settings
│   │   ├── database.py   # SQLite connection
│   │   └── main.py       # FastAPI application
│   ├── tests/            # Test suite
│   ├── requirements.txt  # Dependencies
│   └── Dockerfile        # Backend docker container
├── frontend/
│   ├── src/              # React dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── nginx.conf        # Production routing proxy
│   └── Dockerfile        # Frontend Nginx container
├── docker-compose.yml
└── README.md
```

---

## Getting Started (Local Development)

### 1. Backend Setup

Prerequisites: Python 3.10+

```bash
cd backend
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start FastAPI development server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
The FastAPI swagger docs will be available at `http://localhost:8000/docs`.

### 2. Frontend Setup

Prerequisites: Node 18+

```bash
cd ../frontend
# Install npm dependencies
npm install

# Start React + Vite development server (with reverse proxy to port 8000)
npm run dev
```
The application will launch on `http://localhost:3000`.

---

## Running with Docker Compose

To deploy the entire production stack (FastAPI Backend + React Nginx Frontend + persistent SQLite database + Saved models folders):

```bash
docker-compose up --build
```
- Access the dashboard at `http://localhost:3000`.
- API endpoints map directly through the frontend proxy at `http://localhost:3000/api`.

---

## Model Training & Fine-Tuning

To initiate fine-tuning for either model, you can run the pipeline CLI scripts manually or trigger them via the dashboard settings or FastAPI endpoints.

### Manually running training scripts:
```bash
# Fine-tune the Categorizer
python backend/src/ml/train_category.py

# Fine-tune the DeBERTa-v3 Bias Detector
python backend/src/ml/train_bias.py
```

### Triggering via APIs:
- **Retrain News Category**: `POST http://localhost:8000/train-category`
- **Retrain Bias Detector**: `POST http://localhost:8000/train-bias`
- **Evaluation report**: `POST http://localhost:8000/evaluate`

*Note: Training logs are saved to `./logs/` and can be visualized in TensorBoard using `tensorboard --logdir logs/`.*

---

## Unit Testing

To execute the test suite (health checks, text cleaning, database records):

```bash
cd backend
PYTHONPATH=. pytest tests/
```
