# DOTA 2 Match Prediction Using Big Data Analytics
**Team 12**

**Anh Pham**  
**Jinzhe Bai**  
**Utkarsh Roy**

**Course:** MET CS777 - Big Data Analytics  
**Boston University Metropolitan College**

---

## Project Overview

This project implements three machine learning models to predict Dota 2 professional match outcomes using big data analytics on Google Cloud Platform. We analyze 27.86 GB of professional esports data across 123,692 matches (2020-2024) using Apache Spark and PySpark.

### Research Questions:
1. Can draft-only features predict match outcomes with ≥65% accuracy?
2. How much does early game data (first 10 minutes) improve prediction accuracy?
3. Can we build an effective hero recommendation system based on synergy and counter-pick analysis?

### Key Results:
- **Draft-Only Prediction:** 60.4% accuracy, 0.643 AUC (Sample 1)
- **Early Game Prediction:** 65.1% accuracy, 0.716 AUC (+14.24% improvement over draft-only baseline)
- **Hero Recommendation:** Successfully generates top-3 hero suggestions based on 125 heroes and 7,635 synergy pairs
- **Best Model:** Logistic Regression for both draft and early game prediction

---

## Repository Structure

```
cs777-termproject-team12/
├── code/
│   ├── METCS777-term-project-code-sample-1-Team12.py    # Draft-based prediction
│   ├── METCS777-term-project-code-sample-2-Team12.py    # Early game enhanced prediction
│   └── METCS777-term-project-code-sample-3-Team12.py    # Hero recommendation system
└── data/sampled/
    ├── all_word_counts.csv                              # Word count analysis
    ├── chat.csv                                         # In-game chat data
    ├── cosmetics.csv                                    # Cosmetic items data
    ├── draft_timings.csv                                # Draft timing information
    ├── main_metadata.csv                                # Match outcomes and metadata
    ├── objectives.csv                                   # Game objectives
    ├── picks_bans.csv                                   # Hero draft selections
    ├── players.csv                                      # Player performance stats
    ├── radiant_exp_adv.csv                              # Experience advantages
    ├── radiant_gold_adv.csv                             # Gold advantages
    ├── teamfights.csv                                   # Teamfight statistics
    └── teams.csv                                        # Team information
```

---

## Dataset

**Source:** Dota 2 Professional Matches (Kaggle)  
**Size:** 27.86 GB (full dataset), sampled data provided in repository  
**Matches:** 123,692 professional games (2020-2024)  
**Files:** 12 interconnected CSV files

### Key Files

| File | Description | Key Columns |
|------|-------------|-------------|
| main_metadata.csv | Match outcomes and metadata | match_id, radiant_win, duration, patch |
| picks_bans.csv | Hero draft selections | match_id, hero_id, team, is_pick, order |
| radiant_gold_adv.csv | Gold advantages over time | match_id, minute, gold |
| radiant_exp_adv.csv | Experience advantages | match_id, minute, exp |
| objectives.csv | Game objectives | match_id, type, time, team |
| teamfights.csv | Teamfight statistics | match_id, start, deaths, gold_delta |
| draft_timings.csv | Pick/ban timing data | match_id, pick, total_time_taken, extra_time |

**Full Dataset:** [Kaggle - Dota 2 Matches Dataset](https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches)

---

## Quick Start

### Prerequisites
- Apache Spark 3.x with PySpark
- Python 3.7+
- Google Cloud Platform account
- Libraries: pyspark, json

### Setup

```bash
# Clone repository
git clone https://github.com/mewwmewie/cs777-termproject-team12.git
cd cs777-termproject-team12
```

### Google Cloud Platform Setup

```bash
# Create Cloud Storage buckets
gsutil mb gs://bucket0820/
gsutil mb gs://cs777-termpaper/
gsutil mb gs://dota2-dataproc-bucket-utkarsh/

# Upload data to Cloud Storage
gsutil cp -r data/sampled/* gs://bucket0820/dota_data/

# Upload code files
gsutil cp code/* gs://bucket0820/scripts/

# Create Dataproc cluster
gcloud dataproc clusters create dota2-prediction-cluster \
  --region us-east1 \
  --master-machine-type n1-standard-4 \
  --master-boot-disk-size 50 \
  --num-workers 2 \
  --worker-machine-type n1-standard-4 \
  --worker-boot-disk-size 50 \
  --image-version 2.0-debian10
```

---

## Running the Code

### Code Sample 1: Draft-Based Prediction

Predicts match outcomes using hero picks/bans, synergy, pick order, and timing.

**Configuration:**
- Year Range: 2020-2024
- Features: Hero picks/bans (one-hot encoded), synergy matrix, pick order, draft timing, patch version
- Models: Logistic Regression, Random Forest, Gradient Boosting
- Target Metrics: Accuracy ≥65%, AUC-ROC ≥0.70

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://bucket0820/scripts/METCS777-term-project-code-sample-1-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1
```

**Expected Outputs:**
```
Model Performance:
- Logistic Regression: 60.4% accuracy, 0.643 AUC
- Random Forest: 58.0% accuracy, 0.610 AUC
- Gradient Boosting: 58.5% accuracy, 0.621 AUC

Files Generated:
- model_comparison.csv
- feature_importance_random_forest.csv
- feature_importance_gradient_boosting.csv
- logistic_regression_coefficients.csv
- logistic_regression_intercept.csv
- predictions.csv
- best_model_[model_name]/
```

**Top Features (Gradient Boosting):**
1. synergy_diff: 7.36%
2. patch: 3.55%
3. radiant_synergy: 2.37%
4. dire_first_pick: 2.29%
5. dire_synergy: 1.97%

---

### Code Sample 2: Early Game Enhanced Prediction

Improves predictions by adding first 10 minutes of gameplay data.

**Configuration:**
- Early Game Window: 10 minutes
- Additional Features: Gold/XP advantages, trends, volatility, objectives, teamfights
- Train/Test Split: 80/20
- Models: Logistic Regression, Random Forest, Gradient Boosting

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://cs777-termpaper/scripts/METCS777-term-project-code-sample-2-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1 \
  -- gs://cs777-termpaper/merged gs://cs777-termpaper/output/sample2
```

**Expected Outputs:**
```
Data Loading: 72.43 seconds
- Total rows loaded: 11,068,988

Feature Engineering: 19.43 seconds
- Features created: 21
- Final dataset: 123,692 matches
- Train: 98,953 samples, Test: 24,739 samples

Model Results:
- Logistic Regression (BEST): 65.07% accuracy, 0.7155 AUC, 33.28 sec training
- Random Forest: 64.99% accuracy, 0.7149 AUC, 79.53 sec training
- Gradient Boosting: 64.81% accuracy, 0.7108 AUC, 306.72 sec training

Draft-Only Baseline: 50.83% accuracy, 0.503 AUC
Improvement: +14.24% accuracy gain, +0.2125 AUC gain

Files Generated:
- early_game_results.json/
- early_game_report/
- predictions/
```

**Top 10 Features (Random Forest):**
1. combined_advantage: 24.05%
2. gold_trend_10min: 15.55%
3. gold_at_10min: 14.18%
4. is_radiant_ahead: 10.46%
5. exp_trend_10min: 8.38%
6. gold_min_10min: 4.35%
7. gold_max_10min: 3.50%
8. exp_max_10min: 3.44%
9. exp_at_10min: 3.20%
10. gold_volatility_10min: 2.93%

**Sample Prediction Output:**
```
Example 1 (Match ID: 5234567890):
  Input at 10 minutes:
    - Gold lead: Radiant +2,450
    - Experience lead: Radiant +1,850
    - Teamfights: 1
    - Towers destroyed: 1
  
  Prediction:
    - Draft-only estimate: ~50.8% (baseline)
    - Updated prediction: Radiant 76.2%
    - Confidence: strong
  
  Recommendation: Radiant should maintain aggressive tempo and secure objectives.
                  Dire needs successful ganks to stabilize.
  Actual result: Radiant won
```

---

### Code Sample 3: Hero Recommendation System

Recommends optimal hero picks based on team synergy and enemy counters.

**Configuration:**
- Scoring Formula: 0.4 × Synergy + 0.4 × Counter + 0.2 × Meta
- Minimum Games for Synergy: Statistical threshold
- Output: Top-3 hero recommendations

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://dota2-dataproc-bucket-utkarsh/scripts/METCS777-term-project-code-sample-3-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1
```

**Expected Outputs:**
```
Data Loading: 13.537 seconds
- Rows Loaded: 2,820,604

Meta Statistics: 2.545 seconds
- Meta Heroes: 125

Synergy Matrix: 0.129 seconds
- Synergy Pairs: 7,635

Counter Matrix: 0.081 seconds
- Counter Pairs: 15,282

Recommendation: 1,084.568 seconds

Total Pipeline Time: 1,124.366 seconds

Files Generated:
- Results/output_[timestamp].txt (uploaded to GCS bucket)
```

**Sample Output:**
```
TOP 3 RECOMMENDATIONS
Radiant Picks: [74, 12]
Dire Picks: [102, 45]

Hero 13 (Score: 502.4942)
  • Synergy: 808.25
  • Counter: 447.75
  • Meta: 0.4708

Hero 129 (Score: 453.7899)
  • Synergy: 727.5
  • Counter: 406.75
  • Meta: 0.4497

Hero 106 (Score: 452.3939)
  • Synergy: 694.25
  • Counter: 436.5
  • Meta: 0.4697
```

---

## Results Summary

### Model Performance Comparison

#### Sample 1: Draft-Based Prediction

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|-------|----------|---------|-----------|--------|----------|
| Logistic Regression | 60.42% | 0.643 | 60.40% | 60.42% | 0.604 |
| Random Forest | 58.00% | 0.610 | 58.01% | 58.00% | 0.578 |
| Gradient Boosting | 58.45% | 0.621 | 58.43% | 58.45% | 0.584 |

**Best Model:** Logistic Regression (60.4% accuracy, 0.643 AUC)

#### Sample 2: Early Game Enhanced Prediction

| Model | Accuracy | AUC-ROC | F1 Score | Training Time |
|-------|----------|---------|----------|---------------|
| Logistic Regression | **65.07%** | **0.7155** | 0.6507 | 33.28 sec |
| Random Forest | 64.99% | 0.7149 | 0.6499 | 79.53 sec |
| Gradient Boosting | 64.81% | 0.7108 | 0.6479 | 306.72 sec |

**Best Model:** Logistic Regression (65.1% accuracy, 0.716 AUC)

**Draft-Only Baseline:** 50.83% accuracy, 0.503 AUC  
**Improvement:** +14.24% accuracy, +0.2125 AUC

#### Sample 3: Hero Recommendation System

| Metric | Value |
|--------|-------|
| Heroes Analyzed | 125 |
| Synergy Pairs | 7,635 |
| Counter Pairs | 15,282 |
| Recommendation Time | 1,084.57 sec |
| Total Pipeline Time | 1,124.37 sec |

---

## Key Findings

### 1. Draft Analysis (Sample 1)
- **Synergy Matters Most:** synergy_diff is the top predictor (7.36% importance)
- **Patch Effects:** Game version significantly impacts outcomes (3.55% importance)
- **Team Composition:** Radiant/Dire synergy scores crucial for prediction
- **Logistic Regression Best:** 60.4% accuracy with fastest training time

### 2. Early Game Impact (Sample 2)
- **10-Minute Rule:** First 10 minutes determine 65%+ of match outcomes
- **Gold is King:** Gold advantages account for 54% of top feature importance
  - combined_advantage: 24.05%
  - gold_trend_10min: 15.55%
  - gold_at_10min: 14.18%
- **Momentum Matters:** Trend features more predictive than absolute values
- **Significant Improvement:** +14.24% accuracy gain over draft-only baseline
- **Draft Alone Insufficient:** 50.8% baseline nearly random (coin flip)

### 3. Hero Recommendations (Sample 3)
- **Balanced Approach:** Synergy (40%) and counters (40%) equally weighted
- **Meta Context:** Current hero strength contributes 20% to recommendation
- **Large-Scale Analysis:** Processes 125 heroes and 7,635 synergy combinations
- **Computational Intensity:** ~18 minutes for comprehensive recommendation
- **Actionable Output:** Provides clear top-3 picks with score breakdowns

---

## Technical Architecture

### Sample 1: Draft Prediction Pipeline

```
Multi-Year Data Loading (2016-2024)
    ↓
Feature Engineering
    ├── Hero Picks/Bans (one-hot encoding)
    ├── Synergy Matrix (2-hero combinations)
    ├── Pick Order Features
    ├── Draft Timing Features
    └── Patch Version
    ↓
ML Pipeline
    ├── Vector Assembly
    ├── Standard Scaling
    └── Train/Test Split (80/20)
    ↓
Model Training
    ├── Logistic Regression (maxIter=100, regParam=0.01)
    ├── Random Forest (trees=100, depth=10)
    └── Gradient Boosting (maxIter=100, depth=5)
    ↓
Evaluation & Export
```

### Sample 2: Early Game Prediction Pipeline

```
Data Loading (GCS)
    ├── picks_bans.csv
    ├── main_metadata.csv
    ├── radiant_gold_adv.csv
    ├── radiant_exp_adv.csv
    ├── objectives.csv (optional)
    └── teamfights.csv (optional)
    ↓
Feature Engineering (10-minute window)
    ├── Gold Features (at, max, min, avg, trend, volatility)
    ├── Experience Features (at, max, min, avg, trend, volatility)
    ├── Draft Features (radiant_picks, dire_picks)
    ├── Objectives (tower_kills, early_objective_count)
    ├── Teamfights (count)
    └── Derived Features (is_radiant_ahead, combined_advantage)
    ↓
ML Pipeline
    ├── Vector Assembly (21 features)
    ├── Standard Scaling
    └── Train/Test Split (80/20)
    ↓
Model Training
    ├── Logistic Regression
    ├── Random Forest
    └── Gradient Boosting
    ↓
Baseline Comparison (Draft-Only LR)
    ↓
Sample Predictions with Recommendations
    ↓
Results Export (JSON, Text, CSV)
```

### Sample 3: Hero Recommendation Pipeline

```
Data Loading
    ├── picks_bans.csv
    └── main_metadata.csv
    ↓
Statistical Analysis
    ├── Meta Statistics (win rates, pick rates)
    ├── Synergy Matrix (2-hero win rates)
    └── Counter Matrix (hero vs hero matchups)
    ↓
Scoring Algorithm
    Final Score = 0.4 × Synergy + 0.4 × Counter + 0.2 × Meta
    ↓
Recommendation Generation
    ├── Filter used heroes
    ├── Score all candidates
    └── Return top-3
    ↓
Output to GCS
```

---

## Spark Configuration

### Sample 1 (Draft Prediction)
```python
SparkSession.builder \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "32") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### Sample 2 (Early Game Prediction)
```python
SparkSession.builder \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

### Sample 3 (Hero Recommendation)
```python
SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200")
```

---

## GCP Dataproc Cluster Specifications

- **Master Node:** 1 × n1-standard-4 (4 vCPU, 15 GB RAM)
- **Worker Nodes:** 2 × n1-standard-4 (4 vCPU, 15 GB RAM)
- **Total Resources:** 12 vCPU, 45 GB RAM
- **Image Version:** 2.0-debian10
- **Spark Version:** 3.1.3
- **Storage:** Google Cloud Storage
- **Estimated Cost:** ~$0.17/hour

---

## Performance Metrics

### Sample 1: Draft Prediction (Multi-Year Data)
- **Data Loading:** Variable (depends on years 2016-2024)
- **Feature Engineering:** ~60-120 seconds
- **Model Training:** 
  - Logistic Regression: ~30-40 seconds
  - Random Forest: ~80-100 seconds
  - Gradient Boosting: ~300-350 seconds

### Sample 2: Early Game Prediction (123,692 matches)
- **Data Loading:** 72.43 seconds (11M+ rows)
- **Feature Engineering:** 19.43 seconds
- **Model Training:**
  - Logistic Regression: 33.28 seconds
  - Random Forest: 79.53 seconds
  - Gradient Boosting: 306.72 seconds
- **Total Execution:** 577.23 seconds (~9.6 minutes)

### Sample 3: Hero Recommendation (2.8M records)
- **Data Loading:** 13.54 seconds
- **Meta Statistics:** 2.55 seconds
- **Synergy Matrix:** 0.13 seconds
- **Counter Matrix:** 0.08 seconds
- **Recommendation:** 1,084.57 seconds (~18 minutes)
- **Total Pipeline:** 1,124.37 seconds (~18.7 minutes)

---

## Code Features

### Sample 1 Highlights
- Multi-year data loading (2016-2024)
- Dynamic hero ID detection
- Hero synergy calculation with minimum games threshold
- Pick order and timing features
- Comprehensive feature importance analysis
- Model comparison and export
- GCS-compatible output

### Sample 2 Highlights
- Command-line argument parsing for input/output paths
- Robust error handling for missing files
- Window functions for correct trend calculation
- Draft-only baseline for comparison
- Sample predictions with strategic recommendations
- JSON and CSV output formats
- Performance metrics tracking

### Sample 3 Highlights
- Efficient dictionary-based scoring
- Broadcast variables for distributed computation
- UTF-8 output with GCS upload
- Detailed execution time tracking
- Comprehensive scoring breakdown
- Top-3 recommendation generation

---

## Troubleshooting

### Out of Memory Errors
Increase executor memory:
```bash
--properties spark.executor.memory=8g,spark.driver.memory=8g
```

### Missing Data Files
Sample 2 handles missing teamfights/objectives gracefully:
```python
try:
    teamfights = spark.read.option("mode", "PERMISSIVE").json(...)
except:
    print("[WARNING] Teamfights data not available - skipping")
    teamfights_available = False
```

### Trend Calculation Issues
Ensure proper Window function usage:
```python
window_spec = Window.partitionBy("match_id").orderBy("minute")
gold_start = df.filter(F.col("row_num_asc") == 1)
gold_end = df.filter(F.col("row_num_desc") == 1)
```

### GCS Access Denied
Ensure service account has:
- Storage Admin
- Dataproc Worker
- Dataproc Service Agent

---

## Sample Insights

### Draft Phase
- **Synergy > Individual Heroes:** Team composition more important than star players
- **Patch Dependency:** Meta shifts significantly between patches
- **First Pick Advantage:** Minimal impact (~2% importance)
- **Draft Timing:** Does not strongly correlate with outcomes

### Early Game
- **Critical Window:** Minutes 5-10 most predictive
- **Gold Momentum:** Trend more important than snapshot value
- **Comeback Difficult:** Teams behind at 10 min rarely recover
- **Objective Priority:** Towers > Kills for early game advantage

### Hero Recommendations
- **Synergy Wins Games:** Hero combinations crucial
- **Counter-Picking Works:** Matching up against opponent matters
- **Meta Shifts:** Win rates change significantly over time
- **Complexity Trade-off:** 18-minute calculation for optimal picks

---

## Technologies

- **Big Data:** Apache Spark 3.1.3 (PySpark)
- **Cloud Platform:** Google Cloud Platform (Dataproc)
- **Storage:** Google Cloud Storage
- **ML Framework:** Spark MLlib
- **Languages:** Python 3.7+
- **Libraries:** PySpark, NumPy, JSON

---

## Cleanup

```bash
# Delete Dataproc cluster
gcloud dataproc clusters delete dota2-prediction-cluster --region=us-east1

# Delete Cloud Storage buckets (optional)
gsutil -m rm -r gs://bucket0820
gsutil -m rm -r gs://cs777-termpaper
gsutil -m rm -r gs://dota2-dataproc-bucket-utkarsh
```

---
## Team Contributions

**Jinzhe Bai:**
- Sample 1: Draft-based prediction implementation
- Multi-year data loading pipeline (2016-2024)
- Hero synergy matrix calculation (7,626 pairs)
- Logistic Regression, Random Forest, and Gradient Boosting models
- GCP Dataproc cluster setup and configuration
- Feature importance analysis and model comparison
- One-hot encoding for 248 hero pick/ban features

**Anh Pham:**
- Sample 2: Early game prediction implementation
- Time-series feature engineering (gold/XP advantages at 10 minutes)
- Window function optimization for trend calculations
- Edge case handling for matches <10 minutes duration
- Teamfight win ratio processing (~8 fights per match average)
- Sample prediction generation with strategic recommendations
- Draft-only baseline comparison (+14.24% improvement validation)
- Feature joining pipeline (early game + draft features)

**Utkarsh Roy:**
- Sample 3: Hero recommendation system
- Scoring algorithm development (40% synergy + 40% counter + 20% meta)
- Counter-pick matrix calculation (15,282 combinations)
- Hero statistics analysis (124 unique heroes)
- Top-3 recommendation ranking algorithm
- Performance optimization for 7,635 hero pair computations
- GCS integration and UTF-8 output handling
- Hero pick rate visualization prototyping

**Shared Responsibilities:**
- Weekly progress meetings and coordination
- Data pipeline integration across all three samples
- Performance benchmarking on full dataset (123,692 matches)
- Code review and optimization
- Final results validation and analysis

---

## Citations

1. **Dataset:** Anzelmo, D. (2024). *Dota 2 Matches Dataset*. Kaggle. https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches
2. **Apache Spark:** Apache Software Foundation. (2024). *Apache Spark Documentation*. https://spark.apache.org/docs/latest/
3. **GCP Dataproc:** Google Cloud. (2024). *Dataproc Documentation*. https://cloud.google.com/dataproc/docs
4. **PySpark MLlib:** Apache Software Foundation. (2024). *MLlib: Machine Learning Library*. https://spark.apache.org/mllib/

---

## License

This project is for educational purposes as part of MET CS777 coursework at Boston University.
