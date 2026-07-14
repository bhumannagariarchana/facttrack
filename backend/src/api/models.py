from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="The plain text of the article to analyze.")
    url: Optional[str] = Field(default=None, description="The URL of the article to scrape and analyze.")

class BiasDistribution(BaseModel):
    Left: float
    Center: float
    Right: float

class NeutralVsBiased(BaseModel):
    Neutral: float
    Biased: float

class CategoryDistribution(BaseModel):
    Politics: float
    Business: float
    Sports: float
    Technology: float
    Entertainment: float
    Health: float
    Science: float

class WordImportance(BaseModel):
    word: str
    score: float

class AnalyzeResponse(BaseModel):
    title: Optional[str] = None
    # Category
    category: str
    category_confidence: float
    category_distribution: Dict[str, float]
    
    # Bias
    bias: str
    bias_confidence: float
    bias_distribution: Dict[str, float]
    bias_score: float
    bias_interpretation: str
    neutral_vs_biased: Dict[str, float]
    
    # Explainability
    important_words: List[WordImportance]
    explanation: str

class TrainResponse(BaseModel):
    status: str
    message: str
    task_id: str

class EvaluationReport(BaseModel):
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]

class EvaluateResponse(BaseModel):
    category_metrics: EvaluationReport
    bias_metrics: EvaluationReport
