Here's the updated README with the simplified Repository Structure based on your screenshot:

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
- **Draft-Only Prediction:** 67.4% accuracy, 0.74 AUC (exceeds 65% target)
- **Early Game Prediction:** 71.6% accuracy, 0.72 AUC (+14.2% improvement over draft-only)
- **Hero Recommendation:** Successfully generates top-3 hero suggestions with 64% validation rate
- **Best Model:** Random Forest for draft prediction, Logistic Regression for early game prediction

---

## Repository Structure

```
cs777-termproject-team12/
├── code/
│   ├── METCS777-term-project-code-sample-1-Team12.py    # Code Sample 1: Draft-based prediction
│   ├── METCS777-term-project-code-sample-2-Team12.py    # Code Sample 2: Early game enhanced prediction
│   └── METCS777-term-project-code-sample-3-Team12.py    # Code Sample 3: Hero recommendation system
└── data/sampled/
    ├── all_word_counts.csv                              # Word count analysis
    ├── chat.csv                                         # In-game chat data
    ├── cosmetics.csv                                    # Cosmetic items data
    ├── draft_timings.csv                                # Draft timing information
    ├── main_metadata.csv                                # Match outcomes and metadata
    ├── objectives.csv                                   # Game objectives (towers, roshans, etc.)
    ├── picks_bans.csv                                   # Hero draft selections
    ├── players.csv                                      # Player performance stats
    ├── radiant_exp_adv.csv                              # Experience advantages over time
    ├── radiant_gold_adv.csv                             # Gold advantages over time
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
| picks_bans.csv | Hero draft selections | match_id, hero_id, team, is_pick |
| players.csv | Player performance stats | match_id, kills, deaths, gold_per_min |
| radiant_gold_adv.csv | Gold advantages over time | match_id, minute, gold |
| radiant_exp_adv.csv | Experience advantages | match_id, minute, exp |
| objectives.csv | Game objectives | match_id, type, time, team |
| teamfights.csv | Teamfight statistics | match_id, start, deaths |
| draft_timings.csv | Pick/ban timing data | match_id, pick, total_time_taken |
| teams.csv | Team information | team_id, name, tag |
| chat.csv | In-game chat logs | match_id, time, player_slot, key |
| cosmetics.csv | Cosmetic items | match_id, item_id, hero_id |
| all_word_counts.csv | Chat word frequency | word, count |

**Full Dataset:** [Kaggle - Dota 2 Matches Dataset](https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches)

---

## Quick Start

### Prerequisites
- Apache Spark 3.x with PySpark
- Python 3.7+
- Google Cloud Platform account
- Libraries: numpy, json

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/cs777-termproject-team12.git
cd cs777-termproject-team12
```

### Google Cloud Platform Setup

```bash
# Create Cloud Storage buckets
gsutil mb gs://dota2-project-input
gsutil mb gs://dota2-project-output

# Upload data to Cloud Storage
gsutil cp -r data/sampled/* gs://dota2-project-input/

# Upload code files
gsutil cp code/* gs://dota2-project-code/

# Create Dataproc cluster
gcloud dataproc clusters create dota2-prediction-cluster \
  --region us-east1 \
  --master-machine-type n1-standard-4 \
  --master-boot-disk-size 50 \
  --num-workers 2 \
  --worker-machine-type n1-standard-4 \
  --worker-boot-disk-size 50 \
  --image-version 2.1-debian11
```

---

## Running the Code

### Code Sample 1: Draft-Based Prediction

Predicts match outcomes using only hero picks and bans before the game starts.

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://dota2-project-code/METCS777-term-project-code-sample-1-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1 \
  -- gs://dota2-project-input/ gs://dota2-project-output/sample1/
```

**Expected Outputs:**
- Draft-only accuracy: 67.4%
- AUC-ROC: 0.74
- Training time: ~80 seconds
- Files: model_comparison.csv, feature_importance_random_forest.csv, predictions.csv

**Key Features Used:**
- Hero pick indicators (248 heroes × 2 teams = 496 features)
- Hero synergy scores (team composition)
- Pick order patterns
- Pick timing statistics

---

### Code Sample 2: Early Game Enhanced Prediction

Improves predictions by adding first 10 minutes of gameplay data.

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://dota2-project-code/METCS777-term-project-code-sample-2-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1 \
  -- gs://dota2-project-input/ gs://dota2-project-output/sample2/
```

**Expected Outputs:**
- Early game accuracy: 71.6%
- AUC-ROC: 0.72
- Improvement over draft: +14.2%
- Training time: ~100 seconds
- Files: early_game_results.json, early_game_report/, predictions/

**Additional Features:**
- Gold advantage at 10 minutes (strongest predictor)
- Experience advantage trends
- Teamfight outcomes
- Tower kills and objectives
- Combined momentum indicators

**Sample Prediction Output:**
```
Example 1 (Match ID: 6789012345):
  Input at 10 minutes:
    - Gold lead: Radiant +2,450
    - Experience lead: Radiant +1,850
    - Teamfights: 1
    - Towers destroyed: 1
  
  Prediction:
    - Draft-only estimate: ~52% (baseline)
    - Updated prediction: Radiant 76.2%
    - Confidence: strong
  
  Recommendation: Radiant should maintain aggressive tempo and secure objectives.
                  Dire needs successful ganks to stabilize.
  Actual result: Radiant won
```

---

### Code Sample 3: Hero Recommendation System

Recommends optimal hero picks based on team synergy and enemy counters.

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://dota2-project-code/METCS777-term-project-code-sample-3-Team12.py \
  --cluster=dota2-prediction-cluster \
  --region=us-east1 \
  -- gs://dota2-project-input/ gs://dota2-project-output/sample3/
```

**Expected Outputs:**
- Top 3 hero recommendations with reasoning
- Synergy scores for 7,635 hero pairs
- Counter-pick matrix for 15,282 combinations
- Execution time: ~1,100 seconds
- Validation rate: 64%

**Algorithm:**
```
Final Score = 0.4 × Synergy + 0.4 × Counter + 0.2 × Meta
```

**Sample Output:**
```
TOP 3 RECOMMENDATIONS
Radiant Picks: [74, 12]
Dire Picks: [102, 45]

Hero 13 (Score: 502.49)
  • Synergy: 808.25
  • Counter: 447.75
  • Meta: 0.4708

Hero 129 (Score: 453.79)
  • Synergy: 727.50
  • Counter: 406.75
  • Meta: 0.4497
```

---

## Results Summary

### Model Performance Comparison

| Model | Accuracy | AUC-ROC | Training Time | Key Insight |
|-------|----------|---------|---------------|-------------|
| **Draft-Only (LR)** | 63.8% | 0.71 | 33 sec | Baseline performance |
| **Draft-Only (RF)** | 66.1% | 0.73 | 80 sec | Best draft-only |
| **Draft-Only (GBT)** | **67.4%** | **0.74** | 307 sec | Exceeds 65% target ✓ |
| **Early Game (LR)** | **71.6%** | **0.72** | 28 sec | Best overall ✓ |
| **Early Game (RF)** | 65.1% | 0.72 | 70 sec | Solid performance |
| **Early Game (GBT)** | 64.9% | 0.71 | 280 sec | Slower convergence |

### Feature Importance (Top 10)

**From Random Forest Analysis:**
1. **combined_advantage** (24.05%) - Normalized gold + experience lead
2. **gold_trend_10min** (15.55%) - Gold momentum direction
3. **gold_at_10min** (14.18%) - Current gold advantage
4. **is_radiant_ahead** (10.46%) - Binary lead indicator
5. **exp_trend_10min** (8.38%) - Experience momentum
6. **gold_min_10min** (4.35%) - Minimum gold lead
7. **gold_max_10min** (3.50%) - Maximum gold lead
8. **exp_max_10min** (3.44%) - Maximum experience lead
9. **exp_at_10min** (3.20%) - Current experience advantage
10. **gold_volatility_10min** (2.93%) - Lead stability

### Key Findings

1. **Early Game Impact:**
   - First 10 minutes determine 72% of match outcomes
   - Gold advantage is the strongest predictor
   - +14.2% accuracy improvement over draft-only

2. **Draft Analysis:**
   - Hero synergy matters more than individual hero strength
   - Pick order and timing provide marginal improvements
   - 67.4% accuracy achievable with draft alone

3. **Hero Recommendations:**
   - 64% of recommended heroes appear in winning team compositions
   - Synergy and counter-pick equally important (40% each)
   - Meta trends contribute 20% to recommendation score

---

## Technical Architecture

### Spark Configuration
```python
SparkSession.builder \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g")
```

### GCP Dataproc Cluster Specifications
- **Master Node:** 1 × n1-standard-4 (4 vCPU, 15 GB RAM)
- **Worker Nodes:** 2 × n1-standard-4 (4 vCPU, 15 GB RAM)
- **Total Resources:** 12 vCPU, 45 GB RAM
- **Storage:** Google Cloud Storage
- **Cost:** ~$0.17/hour

### Data Processing Pipeline
```
Raw Data (27.86 GB)
    ↓
Feature Engineering
    ├── Draft Features (248 one-hot encoded heroes)
    ├── Synergy Matrix (7,635 hero pairs)
    ├── Gold/XP Trends (time-series aggregation)
    └── Objectives (event-based features)
    ↓
ML Pipeline
    ├── Vector Assembly
    ├── Standard Scaling
    └── Train/Test Split (80/20)
    ↓
Model Training
    ├── Logistic Regression
    ├── Random Forest (100 trees, depth=10)
    └── Gradient Boosting (100 iterations)
    ↓
Evaluation & Predictions
```

---

## Code Description

### Sample 1: Draft-Based Prediction
**File:** `METCS777-term-project-code-sample-1-Team12.py`

**Features Created:**
- One-hot encoding for 124 heroes × 2 teams
- Hero synergy matrix (pairwise win rates)
- Pick order features (first pick, last pick)
- Draft timing statistics
- Patch version indicators

**Models:**
- Logistic Regression (baseline)
- Random Forest (depth=10, trees=100)
- Gradient Boosting (iterations=100)

**Output:** Model comparison metrics, feature importance, best model predictions

---

### Sample 2: Early Game Enhanced Prediction
**File:** `METCS777-term-project-code-sample-2-Team12.py`

**Additional Features:**
- **Gold Advantages:** gold_at_10min, gold_max_10min, gold_trend_10min, gold_volatility_10min
- **Experience Advantages:** exp_at_10min, exp_trend_10min, exp_volatility_10min
- **Objectives:** tower_kills_10min, first_blood, early_objective_count
- **Teamfights:** teamfights_count_10min (if data available)
- **Momentum:** is_radiant_ahead, combined_advantage, comeback_potential

**Improvements:**
- Window functions for proper trend calculation
- Robust error handling for missing data
- Draft-only baseline for comparison
- Sample predictions with recommendations

**Output:** JSON metrics, text report, prediction CSV with probabilities

---

### Sample 3: Hero Recommendation System
**File:** `METCS777-term-project-code-sample-3-Team12.py`

**Components:**
1. **Meta Analysis:** Calculate win rates and pick rates for all heroes
2. **Synergy Matrix:** Win rates for all 2-hero combinations
3. **Counter Matrix:** Win rates when specific heroes face each other
4. **Scoring Algorithm:** Weighted combination of synergy (40%), counter (40%), and meta (20%)

**Output:** Top-3 hero recommendations with detailed scoring breakdown

---

## Performance Metrics

### Execution Times (Full Dataset on GCP)

| Component | Time | Throughput |
|-----------|------|------------|
| Data Loading | 72 sec | 387 MB/s |
| Feature Engineering | 19 sec | 1,465 MB/s |
| Draft Model Training | 80 sec | - |
| Early Game Training | 100 sec | - |
| Hero Recommendation | 1,100 sec | - |
| **Total Pipeline** | ~570 sec | - |

### Model Training Times

| Model | Draft-Only | Early Game | Improvement |
|-------|------------|------------|-------------|
| Logistic Regression | 33 sec | 28 sec | +15% faster |
| Random Forest | 80 sec | 70 sec | +13% faster |
| Gradient Boosting | 307 sec | 280 sec | +9% faster |

---

## Cost Analysis

### GCP Dataproc Costs

| Task | Runtime | Cost |
|------|---------|------|
| Sample 1 (Draft) | 10 min | $0.28 |
| Sample 2 (Early Game) | 12 min | $0.34 |
| Sample 3 (Recommendation) | 20 min | $0.57 |
| **Total Project** | 42 min | **$1.19** |

**Cost Optimization Tips:**
- Use preemptible workers (70% cost reduction)
- Auto-scale cluster based on workload
- Delete cluster immediately after completion

---

## Troubleshooting

### Out of Memory Errors
Increase executor memory:
```bash
--properties spark.executor.memory=8g,spark.driver.memory=8g
```

### "RDD is empty" Error
This occurs with insufficient sample data. Solutions:
- Use full dataset (123,692 matches)
- Create consistent sample across all files by match_id
- Minimum recommended: 1,000 matches

### Slow Performance
Enable adaptive query execution:
```bash
--properties spark.sql.adaptive.enabled=true
```

### GCS Access Denied
Ensure service account has proper permissions:
- Storage Admin
- Dataproc Worker

---

## Sample Insights

### Match Outcome Patterns
- **Radiant Advantage:** Inherent 3% higher win rate due to map asymmetry
- **Gold Lead Impact:** +1,000 gold at 10 min → +15% win probability
- **Comeback Potential:** High volatility (±500 gold swings) indicates competitive games

### Hero Meta Trends
- **Most Picked Heroes:** Hero 66, 121, 38 (core carries)
- **Highest Win Rates:** Heroes with strong team synergy
- **Ban Priority:** Heroes 112, 137, 53 most frequently banned

---

## Technologies

- **Big Data:** Apache Spark 3.x (PySpark)
- **Cloud Platform:** Google Cloud Platform (Dataproc)
- **Storage:** Google Cloud Storage
- **ML Framework:** Spark MLlib (DataFrame API)
- **Languages:** Python 3.7+
- **Libraries:** NumPy, JSON

---

## Cleanup

```bash
# Delete Dataproc cluster
gcloud dataproc clusters delete dota2-prediction-cluster --region=us-east1

# Delete Cloud Storage buckets
gsutil -m rm -r gs://dota2-project-input
gsutil -m rm -r gs://dota2-project-output
gsutil -m rm -r gs://dota2-project-code
```

---

## Team Contributions

**Anh Pham:**
- GCP infrastructure setup and configuration
- Data ingestion pipeline (12 CSV files)
- Draft feature engineering (Code Sample 1)
- Logistic Regression models
- Report: Introduction and Background sections

**Jinzhe Bai:**
- Time-series feature engineering (gold/XP advantages)
- Teamfight and objectives processing
- Random Forest models (Code Sample 2)
- Meta evolution analysis
- Report: Methodology and Data Description sections

**Utkarsh Roy:**
- Gradient Boosting models and hyperparameter tuning
- Hero recommendation system (Code Sample 3)
- Visualizations and performance dashboards
- Model evaluation framework
- Report: Results, Discussion, and Conclusions sections

**Shared:**
- Weekly progress meetings and code reviews
- Collaborative insight generation
- Final report assembly

---

## Citations

1. **Dataset:** Anzelmo, D. (2024). *Dota 2 Matches Dataset*. Kaggle. https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches
2. **Apache Spark:** Apache Software Foundation. (2024). *Apache Spark Documentation*. https://spark.apache.org/docs/latest/
3. **GCP Dataproc:** Google Cloud. (2024). *Dataproc Documentation*. https://cloud.google.com/dataproc/docs
4. **PySpark MLlib:** Apache Software Foundation. (2024). *MLlib: Machine Learning Library*. https://spark.apache.org/mllib/

---

## License

This project is for educational purposes as part of MET CS777 coursework at Boston University.

---

## Contact

For questions or feedback:
- **Team 12** - MET CS777 Big Data Analytics
- **Institution:** Boston University Metropolitan College
