import os
import sys
import time
import json
import logging
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings
from src.utils.extractor import clean_text

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_category_dataset():
    """
    Downloads and pre-processes news category dataset.
    We try loading HuffPost news category dataset (which has all 7 classes)
    or fall back to AG News mapped classes.
    """
    logger.info("Loading news category dataset...")
    
    # Target categories
    categories = ["Politics", "Business", "Sports", "Technology", "Entertainment", "Health", "Science"]
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}
    
    try:
        # Load HF News Category Dataset (HuffPost)
        raw_dataset = load_dataset("heegyu/news-category-dataset", split="train")
        df = raw_dataset.to_pandas()
        
        # Keep only relevant columns and clean category names
        df = df.rename(columns={"headline": "title", "short_description": "description"})
        
        # Map HuffPost categories to our target 7 categories
        # HuffPost has: POLITICS, BUSINESS, SPORTS, TECH, ENTERTAINMENT, HEALTHY LIVING, SCIENCE, etc.
        category_map = {
            "POLITICS": "Politics",
            "BUSINESS": "Business",
            "SPORTS": "Sports",
            "TECH": "Technology",
            "ENTERTAINMENT": "Entertainment",
            "HEALTHY LIVING": "Health",
            "SCIENCE": "Science",
            "WORLD NEWS": "Politics",
            "PARENTING": "Health",
            "WELLNESS": "Health"
        }
        
        df["target_category"] = df["category"].map(category_map)
        df = df.dropna(subset=["target_category"])
        
        # Combine title and description to form article text representation
        df["text"] = df["title"] + " " + df["description"]
        df["label"] = df["target_category"].map(cat_to_id)
        
        df = df[["text", "label"]].dropna()
        
    except Exception as e:
        logger.warning(f"Could not load custom HuffPost dataset ({str(e)}). Falling back to AG News mapping...")
        # Fallback to AG News
        raw_dataset = load_dataset("ag_news", split="train")
        df = raw_dataset.to_pandas()
        
        # AG News labels: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
        # Map to: 0 -> Politics, 1 -> Sports, 2 -> Business, 3 -> Technology
        # And we'll generate synthetic placeholders for Entertainment, Health, Science if needed,
        # but for a quick train pipeline, we'll just run on the mapped 4 classes.
        ag_map = {0: 0, 1: 2, 2: 1, 3: 3} # World->Politics (0), Sports->Sports (2), Business->Business (1), Sci/Tech->Technology (3)
        df["label"] = df["label"].map(ag_map)
        
        # Create small synthetic items for remaining classes to ensure 7-class outputs compile properly
        synthetic_data = [
            {"text": "The new marvel movie was highly entertaining and filled with action.", "label": 4}, # Entertainment
            {"text": "Doctors advise eating more green vegetables and working out daily.", "label": 5}, # Health
            {"text": "Astronomers discover a new planet in a distant galaxy using Hubble.", "label": 6}  # Science
        ] * 20 # duplicate to avoid empty label classes
        
        import pandas as pd
        df_synthetic = pd.DataFrame(synthetic_data)
        df = pd.concat([df, df_synthetic], ignore_index=True)
        
    # Preprocessing
    logger.info("Cleaning and formatting dataset...")
    df = df.drop_duplicates(subset=["text"])
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]
    
    # Downsample for faster execution on CPU if needed (e.g. limit to 2000 items for demo)
    max_samples = 2000
    if len(df) > max_samples:
        # Balanced sampling if possible
        df = df.groupby('label').apply(lambda x: x.sample(min(len(x), max_samples // len(categories)))).reset_index(drop=True)
        
    # Split: 80% Train, 10% Val, 10% Test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[:int(n * 0.8)]
    val_df = df.iloc[int(n * 0.8):int(n * 0.9)]
    test_df = df.iloc[int(n * 0.9):]
    
    logger.info(f"Dataset split size - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return (
        Dataset.from_pandas(train_df), 
        Dataset.from_pandas(val_df), 
        Dataset.from_pandas(test_df), 
        categories
    )

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    f1_weighted = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def main():
    set_seed(42)
    os.makedirs(settings.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(settings.DATASETS_DIR, exist_ok=True)
    
    train_dataset, val_dataset, test_dataset, categories = prepare_category_dataset()
    
    model_name = "distilbert-base-uncased" # Fast model for quick fine-tuning
    logger.info(f"Initializing tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=settings.MAX_SEQUENCE_LENGTH)
        
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Initialize model with correct number of labels
    num_labels = len(categories)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label={i: cat for i, cat in enumerate(categories)},
        label2id={cat: i for i, cat in enumerate(categories)}
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./temp_category_checkpoints",
        num_train_epochs=settings.EPOCHS,
        learning_rate=settings.LEARNING_RATE,
        per_device_train_batch_size=settings.BATCH_SIZE,
        per_device_eval_batch_size=settings.BATCH_SIZE,
        weight_decay=settings.WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs/category_tensorboard",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=["tensorboard"],
        fp16=torch.cuda.is_available() # Enable FP16 if GPU is available
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    logger.info("Starting fine-tuning...")
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    logger.info(f"Training completed in {duration:.2f} seconds.")
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    test_results = trainer.predict(tokenized_test)
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids
    
    # Classification Report & Confusion Matrix
    report = classification_report(labels, preds, target_names=categories, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()
    
    metrics = {
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    # Save metrics report
    os.makedirs(settings.CATEGORY_MODEL_PATH, exist_ok=True)
    with open(os.path.join(settings.CATEGORY_MODEL_PATH, "evaluation_report.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save training hyperparameters config
    hyperparams = {
        "epochs": settings.EPOCHS,
        "learning_rate": settings.LEARNING_RATE,
        "batch_size": settings.BATCH_SIZE,
        "weight_decay": settings.WEIGHT_DECAY,
        "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
        "duration_seconds": duration,
        "timestamp": time.time()
    }
    with open(os.path.join(settings.CATEGORY_MODEL_PATH, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)
        
    # Save best model
    logger.info(f"Saving best news category model to {settings.CATEGORY_MODEL_PATH}")
    trainer.save_model(settings.CATEGORY_MODEL_PATH)
    tokenizer.save_pretrained(settings.CATEGORY_MODEL_PATH)
    logger.info("Category model training pipeline completed successfully!")

if __name__ == "__main__":
    main()
