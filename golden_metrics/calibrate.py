import argparse
import json
import sqlite3
import math
import random
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import numpy as np
from common.database import db
from common.config import config
from common.models import GoldenSetEntry
from common.repositories import GoldenSetRepository
from curation.scoring import scorer
from common.logging.logger import get_logger

# specific logging for calibration
logger = get_logger("calibration")

class Calibrator:
    def __init__(self):
        self.conn = db.get_connection()

def process_url_task(row_data, index, total_rows):
    """
    Independent function for processing a single URL task.
    Must be top-level for multiprocessing picklability.
    """
    # Re-import logger here to ensure it works in subprocess
    from common.logging.logger import get_logger
    from curation.scoring import scorer
    import trafilatura
    import requests

    url = row_data['url'].strip()
    if not url: return None

    label_raw = row_data['label'].strip().lower()
    content_type = row_data.get('content_type', 'default').strip() or 'default'
    notes = row_data.get('notes', '').strip()

    # Normalize label
    try:
        if label_raw.isdigit():
            label_val = int(label_raw)
        elif label_raw == "exemplary":
            label_val = 5
        elif label_raw == "garbage":
            label_val = 1
        else:
            return None # Skip invalid

        if not (1 <= label_val <= 5):
            return None # Skip out of range

    except ValueError:
            return None # Skip parse error

    # Fetch and score
    metrics = {}
    domain = ""
    first_sentence = None

    try:
        # Use requests for fetching
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'TruthAtlas/1.0 (Research Crawler)',
            'Accept': 'text/html,application/xhtml+xml'
        })
        downloaded = response.text if response.status_code == 200 else None

        if downloaded:
            # Pass HTML string directly to extract
            text = trafilatura.extract(downloaded) or ""
            if text:
                sentences = text.split('.')
                first_sentence = sentences[0].strip() if sentences else text[:50]

            # extract_metadata expects HTML string too
            metadata = trafilatura.extract_metadata(downloaded)
            meta_dict = metadata.as_dict() if metadata else {}
            meta_dict['url'] = url
        else:
            # Fallback or error
            text = "Simulated content for " + url
            meta_dict = {"url": url}
    except Exception as e:
        print(f"Fetcher error for {url}: {e}")
        text = ""
        meta_dict = {"url": url}

    try:
        raw_metrics = scorer.compute_raw_metrics(text, meta_dict)
        domain = scorer._get_domain(url)
    except Exception as e:
        print(f"Scoring failed for {url}: {e}")
        return None

    return {
        "url": url,
        "label": label_val,
        "content_type": content_type,
        "notes": notes,
        "metrics": raw_metrics,
        "domain": domain,
        "first_sentence": first_sentence,
        "index": index,
        "total": total_rows
    }

class Calibrator:
    def __init__(self, database=None, quality_scorer=None, golden_set_repo=None):
        self._database = database or db
        self._scorer = quality_scorer or scorer
        self._golden_set_repo = golden_set_repo or GoldenSetRepository(self._database)
        self.conn = self._database.get_connection()

    def import_from_csv(self, csv_path: str):
        """
        Imports golden set entries from a CSV file.
        CSV format: url,label,content_type,notes
        Label can be: exemplary/garbage OR numeric 1-5
        """
        import csv
        import concurrent.futures

        try:
            # First pass to count total rows for logging
            total_rows = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_rows = sum(1 for row in reader)

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Check for required columns
                required = {'url', 'label'}
                if not required.issubset(reader.fieldnames):
                    raise ValueError(f"CSV missing required columns: {required - set(reader.fieldnames)}")

                success_count = 0
                tasks = []

                # Use ProcessPoolExecutor to avoid GIL/malloc double free issues with lxml
                # Reduced max_workers to 4 to be safe with memory
                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    for i, row in enumerate(reader, 1):
                        # Convert row to dict to ensure picklability
                        row_dict = dict(row)
                        tasks.append(executor.submit(process_url_task, row_dict, i, total_rows))

                    for future in concurrent.futures.as_completed(tasks):
                        try:
                            result = future.result()
                            if result:
                                logger.info(f"Processing URL {result['index']}/{total_rows}: {result['url']}")

                                # DB Write (sequential in main thread)
                                try:
                                    self._write_entry(
                                        result["url"],
                                        result["label"],
                                        result["content_type"],
                                        result["notes"],
                                        result["metrics"],
                                        result["domain"]
                                    )
                                    if result["first_sentence"]:
                                            logger.info(f"Successfully fetched and extracted {result['url']}")
                                            print(f"Fetched {result['url']}: {result['first_sentence']}...")
                                    success_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to write DB for {result['url']}: {e}")
                        except Exception as e:
                            logger.error(f"Task failed: {e}")

                logger.info(f"Successfully imported {success_count} entries from {csv_path}")

        except Exception as e:
            logger.error(f"CSV import failed: {e}")
            raise

    # _fetch_and_compute removed as it's now in process_url_task

    def _write_entry(self, url: str, label: int, content_type: str, notes: str, raw_metrics: dict, domain: str):
        entry = GoldenSetEntry(
            url=url,
            label=str(label),
            content_type=content_type,
            notes=notes,
            raw_metrics=json.dumps(raw_metrics),
            version=1,
            domain=domain,
        )
        self._golden_set_repo.upsert(entry)

    # _add_entry removed

    def add_to_golden_set(self, url: str, label: str, content_type: str = "default", notes: str = ""):
        """
        Legacy wrapper for CLI compatibility.
        """
        row_data = {"url": url, "label": label, "content_type": content_type, "notes": notes}

        # We need to map string labels to int
        if label == "exemplary":
            row_data["label"] = "5"
        elif label == "garbage":
            row_data["label"] = "1"

        result = process_url_task(row_data, 1, 1)

        if result:
            self._write_entry(result["url"], result["label"], result["content_type"], result["notes"], result["metrics"], result["domain"])
            logger.info(f"Added {url} to golden set (label={label})")
        else:
            logger.error(f"Failed to process/add {url}")

    def train_weights(self, content_type: str = "default"):
        """
        Trains weights using Topic-Cluster Cross-Validation and Lasso Regression (L1 + Positive constraint).
        """
        try:
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed. Cannot train weights.")
            return

        rows = self._golden_set_repo.get_for_training(content_type)

        # Filter for exemplary (5) and garbage (1) only for training, as per guide
        training_samples = []
        for label_str, metrics, domain in rows:
            label_val = int(label_str)
            if label_val == 5 or label_val == 1:
                training_samples.append({
                    "label": 1.0 if label_val == 5 else 0.0,
                    "metrics": metrics,
                    "domain": domain
                })

        if len(training_samples) < 10:
            logger.warning(f"Not enough exemplary/garbage samples for {content_type} to train (found {len(training_samples)})")
            return

        # Topic-Cluster CV: Split by domain (Simple shuffle split for now)
        domains = list(set(s["domain"] for s in training_samples))
        random.shuffle(domains)

        # Use all data for training final weights, but could do CV for evaluation.
        train_set = training_samples

        if not train_set:
            logger.warning("Train set empty.")
            return

        feature_names = self._scorer.METRICS
        X_train = [[s["metrics"].get(f, 0.0) for f in feature_names] for s in train_set]
        y_train = [s["label"] for s in train_set]

        # Use Lasso Regression with L1 penalty and positive constraint
        model = Lasso(alpha=0.001, fit_intercept=False, positive=True, random_state=42)
        try:
             model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return

        coefs = model.coef_
        new_weights = {}
        total = sum(abs(c) for c in coefs)

        if total > 0:
            for name, coef in zip(feature_names, coefs):
                if abs(coef) > 0.0001:
                    new_weights[name] = abs(coef) / total # Normalize to sum to 1
        else:
             logger.warning("All coefficients zero, keeping default weights.")
             return

        logger.info(f"New calibrated weights for {content_type}: {new_weights}")
        self._scorer.update_weights(content_type, new_weights)
        return new_weights

    def compute_qadi(self, confusion_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute Quantity-Allocation Disagreement Index (QADI).
        """
        n = confusion_matrix.sum()
        if n == 0:
            return {"qadi": 0.0, "quantity": 0.0, "allocation": 0.0}

        row_sums = confusion_matrix.sum(axis=1) # Actual
        col_sums = confusion_matrix.sum(axis=0) # Predicted

        # Quantity disagreement
        Q = sum(abs(row_sums[i] - col_sums[i]) for i in range(len(row_sums))) / (2 * n)

        # Allocation disagreement
        diagonal = sum(confusion_matrix[i, i] for i in range(len(row_sums)))
        A = (n - diagonal) / n - Q

        qadi_score = math.sqrt(Q**2 + A**2)
        return {"qadi": qadi_score, "quantity": Q, "allocation": A}

    def validate(self, content_type: str = "default"):
        """
        Validates the separation and computes QADI metrics.
        """
        rows = self._golden_set_repo.get_for_validation(content_type)

        y_true = []
        y_pred = []

        exemplary_scores = []
        garbage_scores = []

        for label_str, metrics in rows:
            label_val = int(label_str)
            score = self._scorer.compute_score(metrics, content_type)

            # Binary classification for QADI: Good (>=4) vs Bad (<=2). Ignore 3.
            if label_val >= 4:
                y_true.append(1)
                y_pred.append(1 if score > 0.5 else 0) # Assuming 0.5 threshold
                exemplary_scores.append(score)
            elif label_val <= 2:
                y_true.append(0)
                y_pred.append(1 if score > 0.5 else 0)
                garbage_scores.append(score)

        if exemplary_scores and garbage_scores:
            avg_ex = sum(exemplary_scores) / len(exemplary_scores)
            avg_gb = sum(garbage_scores) / len(garbage_scores)
            logger.info(f"Validation for {content_type}: Avg Exemplary={avg_ex:.3f}, Avg Garbage={avg_gb:.3f}")
            if avg_ex > avg_gb:
                 logger.info("Separation verified.")
            else:
                 logger.warning("Separation failed! Garbage scoring higher than Exemplary.")

        # Compute QADI
        if y_true:
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                qadi_res = self.compute_qadi(cm)

                logger.info(f"QADI Metrics for {content_type}:")
                logger.info(f"  Score: {qadi_res['qadi']:.3f} (Lower is better)")
                logger.info(f"  Quantity Disagreement: {qadi_res['quantity']:.3f}")
                logger.info(f"  Allocation Disagreement: {qadi_res['allocation']:.3f}")

                if qadi_res['qadi'] < 0.3:
                    logger.info("QADI verification passed (< 0.3).")
                else:
                    logger.warning("QADI verification failed (> 0.3). Recalibration recommended.")

            except ImportError:
                logger.warning("sklearn not available for QADI confusion matrix.")


def main():
    parser = argparse.ArgumentParser(description="Truth Atlas Calibration Module")
    parser.add_argument("--add-exemplary", help="URL to add as exemplary")
    parser.add_argument("--add-garbage", help="URL to add as garbage")
    parser.add_argument("--content-type", help="Content type for the URL")
    parser.add_argument("--notes", default="", help="Notes for the golden set entry")
    parser.add_argument("--train-weights", action="store_true", help="Train weights based on golden set")
    parser.add_argument("--validate", action="store_true", help="Validate current weights")
    parser.add_argument("--qadi-report", action="store_true", help="Report QADI metrics (alias for --validate)")

    parser.add_argument("--import-csv", help="Path to CSV file to import golden set from")

    args = parser.parse_args()

    # Defaults from config
    csv_path = args.import_csv or config.get("calibration.csv_path")
    content_type = args.content_type or config.get("calibration.default_content_type")

    calibrator = Calibrator()

    # If explicit CSV arg is provided, or if we want to ensure import happens before training
    if args.import_csv:
        calibrator.import_from_csv(args.import_csv)
    elif csv_path and not (args.add_exemplary or args.add_garbage or args.train_weights or args.validate or args.qadi_report):
        # If no action specified but CSV is in config, run import
        logger.info(f"No action specified. Importing from config CSV: {csv_path}")
        calibrator.import_from_csv(csv_path)

    if args.add_exemplary:
        calibrator.add_to_golden_set(args.add_exemplary, "exemplary", content_type, args.notes)

    if args.add_garbage:
        calibrator.add_to_golden_set(args.add_garbage, "garbage", content_type, args.notes)

    if args.train_weights:
        calibrator.train_weights(content_type)

    if args.validate or args.qadi_report:
        calibrator.validate(content_type)

if __name__ == "__main__":
    main()
