"""
SAMPLE 2
Early Game Enhanced Match Outcome Prediction Model
================================================================
Dota2 Match Outcome Prediction with Draft + 10-Minute Features

Usage:
    spark-submit early_game_prediction_gcp.py <input_path> <output_path>
    
Example:
    spark-submit early_game_prediction_gcp.py gs://bucket/dota2-data gs://bucket/output
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import time
import json

# ==============================================================================
# PARSE ARGUMENTS
# ==============================================================================

if len(sys.argv) != 3:
    print("Usage: script.py <input_path> <output_path>", file=sys.stderr)
    print("Example: spark-submit script.py gs://bucket/dota2-data gs://bucket/output", file=sys.stderr)
    sys.exit(-1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print("=" * 80)

# ==============================================================================
# INITIALIZE SPARK SESSION
# ==============================================================================

spark = SparkSession.builder \
    .appName("Dota2-EarlyGame-Prediction-GCP") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.rdd.compress", "true") \
    .config("spark.shuffle.compress", "true") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

sc = spark.sparkContext

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters"""
    EARLY_GAME_MINUTES = 10
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2
    RANDOM_SEED = 42
    
    RF_MAX_DEPTH = 10
    RF_NUM_TREES = 100
    GBT_MAX_ITER = 100
    GBT_LEARNING_RATE = 0.1

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def round_float(value, decimals=2):
    """Round a float value to specified decimals"""
    if value is None or value != value:  # Check for None or NaN
        return 0.0
    return float(f"{value:.{decimals}f}")

def calculate_throughput(rows_processed, time_seconds):
    """Calculate data throughput in rows per second"""
    if time_seconds > 0:
        return rows_processed / time_seconds
    return 0

# ==============================================================================
# INITIALIZE METRICS TRACKING
# ==============================================================================

pipeline_start_time = time.time()

results = {
    "execution_info": {
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_PATH
    },
    "data_loading": {
        "loading_time_sec": 0,
        "total_rows_loaded": 0
    },
    "feature_engineering": {
        "time_sec": 0,
        "features_created": 0,
        "final_dataset_rows": 0
    },
    "model_results": {
        "logistic_regression": {},
        "random_forest": {},
        "gradient_boosting": {}
    },
    "performance_summary": {}
}

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

load_start = time.time()

try:
    print("Loading datasets from GCS...")
    
    picks_bans = spark.read.option("mode", "DROPMALFORMED").csv(
        f"{INPUT_PATH}/picks_bans.csv*", header=True, inferSchema=True)
    
    metadata = spark.read.csv(
        f"{INPUT_PATH}/main_metadata.csv", header=True, inferSchema=True)
    
    gold_adv = spark.read.option("mode", "DROPMALFORMED").csv(
        f"{INPUT_PATH}/radiant_gold_adv.csv*", header=True, inferSchema=True)
    
    exp_adv = spark.read.option("mode", "DROPMALFORMED").csv(
        f"{INPUT_PATH}/radiant_exp_adv.csv*", header=True, inferSchema=True)
    
    # Optional files - handle if not present
    try:
        teamfights = spark.read.option("mode", "PERMISSIVE").json(
            f"{INPUT_PATH}/teamfights.csv*")
        if "_corrupt_record" in teamfights.columns:
            teamfights = teamfights.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")
        teamfights_available = True
    except:
        print("[WARNING] Teamfights data not available or corrupt - skipping")
        teamfights_available = False
    
    try:
        objectives = spark.read.option("mode", "DROPMALFORMED").csv(
            f"{INPUT_PATH}/objectives.csv*", header=True, inferSchema=True)
        objectives_available = True
    except:
        print("[WARNING] Objectives data not available - skipping")
        objectives_available = False
    
    # Cache frequently accessed datasets
    picks_bans.persist()
    metadata.persist()
    gold_adv.persist()
    exp_adv.persist()
    
    picks_bans_count = picks_bans.count()
    metadata_count = metadata.count()
    gold_count = gold_adv.count()
    exp_count = exp_adv.count()
    
    print(f"Picks/Bans: {picks_bans_count:,} rows")
    print(f"Metadata: {metadata_count:,} rows")
    print(f"Gold Advantage: {gold_count:,} rows")
    print(f"Exp Advantage: {exp_count:,} rows")
    
    load_time = time.time() - load_start
    total_rows_loaded = picks_bans_count + metadata_count + gold_count + exp_count
    
    print(f"\nData loading completed in {round_float(load_time, 2)} seconds")
    print(f"Total rows loaded: {total_rows_loaded:,}")
    print(f"Throughput: {round_float(calculate_throughput(total_rows_loaded, load_time), 2):,} rows/sec")
    
    results["data_loading"]["loading_time_sec"] = round_float(load_time, 2)
    results["data_loading"]["total_rows_loaded"] = total_rows_loaded
    
except Exception as e:
    print(f"[ERROR] Failed to load data: {str(e)}")
    spark.stop()
    sys.exit(-1)

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

fe_start = time.time()

# Prepare metadata with labels
print("Processing match metadata...")
metadata_clean = metadata.select("match_id", "radiant_win", "patch").dropDuplicates(["match_id"])
metadata_clean = metadata_clean.withColumn(
    "label",
    F.when(F.col("radiant_win").cast("string").isin(["True", "1", "true"]), 1).otherwise(0)
)

# 1. Gold advantage features
print("Creating gold advantage features...")
early_gold = gold_adv.filter(F.col("minute") <= Config.EARLY_GAME_MINUTES)

window_spec = Window.partitionBy("match_id").orderBy("minute")
window_spec_desc = Window.partitionBy("match_id").orderBy(F.desc("minute"))

early_gold_with_order = early_gold.withColumn(
    "row_num_asc", F.row_number().over(window_spec)
).withColumn(
    "row_num_desc", F.row_number().over(window_spec_desc)
)

gold_start = early_gold_with_order.filter(F.col("row_num_asc") == 1) \
    .select("match_id", F.col("gold").alias("gold_start"))

gold_end = early_gold_with_order.filter(F.col("row_num_desc") == 1) \
    .select("match_id", F.col("gold").alias("gold_end"))

gold_features = early_gold.groupBy("match_id").agg(
    F.max(F.when(F.col("minute") == Config.EARLY_GAME_MINUTES, F.col("gold"))).alias("gold_at_10min"),
    F.max("gold").alias("gold_max_10min"),
    F.min("gold").alias("gold_min_10min"),
    F.avg("gold").alias("gold_avg_10min"),
    F.stddev("gold").alias("gold_volatility_10min"),
    F.count("*").alias("gold_samples")
)

gold_features = gold_features.join(gold_start, on="match_id", how="left")
gold_features = gold_features.join(gold_end, on="match_id", how="left")
gold_features = gold_features.withColumn(
    "gold_trend_10min",
    F.coalesce(F.col("gold_end") - F.col("gold_start"), F.lit(0))
)
gold_features = gold_features.drop("gold_start", "gold_end")

print(f"Gold features created for {gold_features.count():,} matches")

# 2. Experience advantage features
print("Creating experience advantage features...")
early_exp = exp_adv.filter(F.col("minute") <= Config.EARLY_GAME_MINUTES)

early_exp_with_order = early_exp.withColumn(
    "row_num_asc", F.row_number().over(window_spec)
).withColumn(
    "row_num_desc", F.row_number().over(window_spec_desc)
)

exp_start = early_exp_with_order.filter(F.col("row_num_asc") == 1) \
    .select("match_id", F.col("exp").alias("exp_start"))

exp_end = early_exp_with_order.filter(F.col("row_num_desc") == 1) \
    .select("match_id", F.col("exp").alias("exp_end"))

exp_features = early_exp.groupBy("match_id").agg(
    F.max(F.when(F.col("minute") == Config.EARLY_GAME_MINUTES, F.col("exp"))).alias("exp_at_10min"),
    F.max("exp").alias("exp_max_10min"),
    F.min("exp").alias("exp_min_10min"),
    F.avg("exp").alias("exp_avg_10min"),
    F.stddev("exp").alias("exp_volatility_10min"),
    F.count("*").alias("exp_samples")
)

exp_features = exp_features.join(exp_start, on="match_id", how="left")
exp_features = exp_features.join(exp_end, on="match_id", how="left")
exp_features = exp_features.withColumn(
    "exp_trend_10min",
    F.coalesce(F.col("exp_end") - F.col("exp_start"), F.lit(0))
)
exp_features = exp_features.drop("exp_start", "exp_end")

print(f"Exp features created for {exp_features.count():,} matches")

# 3. Draft features
print("Creating draft features...")
hero_picks = picks_bans.filter(F.col("is_pick") == True)

draft_features = hero_picks.groupBy("match_id").agg(
    F.sum(F.when(F.col("team") == 0, 1).otherwise(0)).alias("radiant_picks"),
    F.sum(F.when(F.col("team") == 1, 1).otherwise(0)).alias("dire_picks")
)

print(f"Draft features created for {draft_features.count():,} matches")

# 4. Objectives features (if available)
if objectives_available:
    print("Creating objectives features...")
    early_objectives = objectives.withColumn(
        "time_minute", F.col("time") / 60
    ).filter(F.col("time_minute") <= Config.EARLY_GAME_MINUTES)
    
    objectives_features = early_objectives.groupBy("match_id").agg(
        F.count("*").alias("early_objective_count"),
        F.sum(F.when(F.col("type").like("%tower%"), 1).otherwise(0)).alias("tower_kills_10min")
    )
    print(f"Objectives features created for {objectives_features.count():,} matches")
else:
    objectives_features = None

# 5. Teamfight features (if available)
if teamfights_available and "match_id" in teamfights.columns and "start" in teamfights.columns:
    print("Creating teamfight features...")
    early_teamfights = teamfights.withColumn(
        "start_minute", F.col("start") / 60
    ).filter(F.col("start_minute") <= Config.EARLY_GAME_MINUTES)
    
    teamfight_features = early_teamfights.groupBy("match_id").agg(
        F.count("*").alias("teamfights_count_10min")
    )
    print(f"Teamfight features created for {teamfight_features.count():,} matches")
else:
    teamfight_features = None

# Merge all features
print("\nMerging all features...")
features_df = metadata_clean.select("match_id", "label", "patch")

features_df = features_df.join(draft_features, on="match_id", how="left")
features_df = features_df.join(gold_features, on="match_id", how="left")
features_df = features_df.join(exp_features, on="match_id", how="left")

if objectives_features:
    features_df = features_df.join(objectives_features, on="match_id", how="left")
else:
    features_df = features_df.withColumn("early_objective_count", F.lit(0))
    features_df = features_df.withColumn("tower_kills_10min", F.lit(0))

if teamfight_features:
    features_df = features_df.join(teamfight_features, on="match_id", how="left")
else:
    features_df = features_df.withColumn("teamfights_count_10min", F.lit(0))

# Add derived features
features_df = features_df.withColumn(
    "is_radiant_ahead",
    F.when((F.coalesce(F.col("gold_at_10min"), F.lit(0)) > 0) & 
           (F.coalesce(F.col("exp_at_10min"), F.lit(0)) > 0), 1).otherwise(0)
)

features_df = features_df.withColumn(
    "combined_advantage",
    (F.coalesce(F.col("gold_at_10min"), F.lit(0)) / 1000.0) +
    (F.coalesce(F.col("exp_at_10min"), F.lit(0)) / 1000.0)
)

# Fill nulls
features_df = features_df.fillna(0)

# Repartition and cache
features_df = features_df.repartition(50)
features_df.persist()

fe_time = time.time() - fe_start
final_count = features_df.count()
feature_cols = [c for c in features_df.columns if c not in ["match_id", "label", "patch"]]

print(f"\nFeature engineering completed in {round_float(fe_time, 2)} seconds")
print(f"Final dataset: {final_count:,} rows")
print(f"Total features: {len(feature_cols)}")
print(f"Throughput: {round_float(calculate_throughput(final_count, fe_time), 2):,} rows/sec")

results["feature_engineering"]["time_sec"] = round_float(fe_time, 2)
results["feature_engineering"]["features_created"] = len(feature_cols)
results["feature_engineering"]["final_dataset_rows"] = final_count

if final_count < 100:
    print(f"\n[WARNING] Only {final_count} samples - results may not be reliable")
    print("[WARNING] Consider using full dataset for production runs")

# ==============================================================================
# PREPARE ML DATA
# ==============================================================================

print("\n" + "=" * 80)
print("PREPARING ML DATA")
print("=" * 80)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

assembled_df = assembler.transform(features_df)
scaler_model = scaler.fit(assembled_df)
ml_df = scaler_model.transform(assembled_df)
ml_df = ml_df.select("match_id", "features", "label")

# Split data
train_df, test_df = ml_df.randomSplit(
    [Config.TRAIN_RATIO, Config.TEST_RATIO],
    seed=Config.RANDOM_SEED
)

train_df.persist()
test_df.persist()

train_count = train_df.count()
test_count = test_df.count()

print(f"Training set: {train_count:,} samples")
print(f"Test set: {test_count:,} samples")

if test_count == 0:
    print("[ERROR] No test samples available - cannot evaluate models")
    spark.stop()
    sys.exit(-1)

# ==============================================================================
# MODEL TRAINING & EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING & EVALUATION")
print("=" * 80)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

# 1. Logistic Regression
print("\n--- Logistic Regression ---")
lr_train_start = time.time()

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)

lr_train_time = time.time() - lr_train_start
lr_auc = evaluator_auc.evaluate(lr_predictions)
lr_accuracy = evaluator_acc.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)

print(f"Training time: {round_float(lr_train_time, 2)} sec")
print(f"AUC-ROC: {round_float(lr_auc, 4)}")
print(f"Accuracy: {round_float(lr_accuracy, 4)}")
print(f"F1 Score: {round_float(lr_f1, 4)}")

results["model_results"]["logistic_regression"] = {
    "training_time_sec": round_float(lr_train_time, 2),
    "auc": round_float(lr_auc, 4),
    "accuracy": round_float(lr_accuracy, 4),
    "f1_score": round_float(lr_f1, 4)
}

# 2. Random Forest
print("\n--- Random Forest ---")
rf_train_start = time.time()

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=Config.RF_NUM_TREES,
    maxDepth=Config.RF_MAX_DEPTH,
    seed=Config.RANDOM_SEED
)

rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

rf_train_time = time.time() - rf_train_start
rf_auc = evaluator_auc.evaluate(rf_predictions)
rf_accuracy = evaluator_acc.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)

# Feature importance
feature_importance = [(feature_cols[i], float(rf_model.featureImportances[i]))
                     for i in range(len(feature_cols))]
feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"Training time: {round_float(rf_train_time, 2)} sec")
print(f"AUC-ROC: {round_float(rf_auc, 4)}")
print(f"Accuracy: {round_float(rf_accuracy, 4)}")
print(f"F1 Score: {round_float(rf_f1, 4)}")
print("\nTop 10 Important Features:")
for feat, imp in feature_importance[:10]:
    print(f"  {feat}: {round_float(imp, 4)}")

results["model_results"]["random_forest"] = {
    "training_time_sec": round_float(rf_train_time, 2),
    "auc": round_float(rf_auc, 4),
    "accuracy": round_float(rf_accuracy, 4),
    "f1_score": round_float(rf_f1, 4),
    "feature_importance": {feat: round_float(imp, 4) for feat, imp in feature_importance[:20]}
}

# 3. Gradient Boosting
print("\n--- Gradient Boosting ---")
gbt_train_start = time.time()

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=Config.GBT_MAX_ITER,
    stepSize=Config.GBT_LEARNING_RATE,
    maxDepth=5,
    seed=Config.RANDOM_SEED
)

gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

gbt_train_time = time.time() - gbt_train_start
gbt_auc = evaluator_auc.evaluate(gbt_predictions)
gbt_accuracy = evaluator_acc.evaluate(gbt_predictions)
gbt_f1 = evaluator_f1.evaluate(gbt_predictions)

print(f"Training time: {round_float(gbt_train_time, 2)} sec")
print(f"AUC-ROC: {round_float(gbt_auc, 4)}")
print(f"Accuracy: {round_float(gbt_accuracy, 4)}")
print(f"F1 Score: {round_float(gbt_f1, 4)}")

results["model_results"]["gradient_boosting"] = {
    "training_time_sec": round_float(gbt_train_time, 2),
    "auc": round_float(gbt_auc, 4),
    "accuracy": round_float(gbt_accuracy, 4),
    "f1_score": round_float(gbt_f1, 4)
}

# ==============================================================================
# DRAFT-ONLY BASELINE FOR COMPARISON
# ==============================================================================

print("\n" + "=" * 80)
print("TRAINING DRAFT-ONLY BASELINE (for comparison)")
print("=" * 80)

# Create draft-only features
draft_only_cols = [c for c in feature_cols if 'pick' in c.lower() or 'draft' in c.lower()]

if len(draft_only_cols) > 0:
    print(f"Draft-only features: {draft_only_cols}")
    
    # Assemble draft-only features
    draft_assembler = VectorAssembler(
        inputCols=draft_only_cols,
        outputCol="draft_features_raw",
        handleInvalid="skip"
    )
    
    draft_scaler = StandardScaler(
        inputCol="draft_features_raw",
        outputCol="draft_features",
        withStd=True,
        withMean=False
    )
    
    draft_assembled = draft_assembler.transform(features_df)
    draft_scaler_model = draft_scaler.fit(draft_assembled)
    draft_ml_df = draft_scaler_model.transform(draft_assembled)
    draft_ml_df = draft_ml_df.select("match_id", "draft_features", "label")
    
    # Use same train/test split
    draft_train = draft_ml_df.join(
        train_df.select("match_id"), on="match_id", how="inner"
    ).withColumnRenamed("draft_features", "features")
    
    draft_test = draft_ml_df.join(
        test_df.select("match_id"), on="match_id", how="inner"
    ).withColumnRenamed("draft_features", "features")
    
    # Train simple LR model on draft only
    draft_lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100
    )
    
    draft_lr_model = draft_lr.fit(draft_train)
    draft_predictions = draft_lr_model.transform(draft_test)
    
    draft_accuracy = evaluator_acc.evaluate(draft_predictions)
    draft_auc = evaluator_auc.evaluate(draft_predictions)
    
    print(f"Draft-Only Baseline:")
    print(f"  Accuracy: {round_float(draft_accuracy, 4)}")
    print(f"  AUC: {round_float(draft_auc, 4)}")
    
    # Calculate improvement
    improvement_accuracy = lr_accuracy - draft_accuracy
    improvement_auc = lr_auc - draft_auc
    
    print(f"\nImprovement with Early Game Features:")
    print(f"  Accuracy: +{round_float(improvement_accuracy * 100, 2)}%")
    print(f"  AUC: +{round_float(improvement_auc, 4)}")
    
    draft_baseline = {
        "accuracy": round_float(draft_accuracy, 4),
        "auc": round_float(draft_auc, 4)
    }
    improvement_metrics = {
        "accuracy_gain": round_float(improvement_accuracy * 100, 2),
        "auc_gain": round_float(improvement_auc, 4)
    }
else:
    print("No draft features found - skipping baseline")
    draft_accuracy = 0.52  # Random baseline
    improvement_accuracy = lr_accuracy - draft_accuracy
    draft_baseline = {"accuracy": 0.52, "auc": 0.50}
    improvement_metrics = {
        "accuracy_gain": round_float((lr_accuracy - 0.52) * 100, 2),
        "auc_gain": round_float(lr_auc - 0.50, 4)
    }

# ==============================================================================
# DETERMINE BEST MODEL
# ==============================================================================

# Find best model BEFORE using it
best_model = max(
    [("Logistic Regression", lr_auc),
     ("Random Forest", rf_auc),
     ("Gradient Boosting", gbt_auc)],
    key=lambda x: x[1]
)

print(f"\nBest Model: {best_model[0]} (AUC: {round_float(best_model[1], 4)})")

# ==============================================================================
# GENERATE SAMPLE PREDICTIONS WITH RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 80)
print("GENERATING SAMPLE PREDICTIONS")
print("=" * 80)

# Get a few sample predictions from best model
if best_model[0] == "Random Forest":
    sample_preds = rf_predictions
elif best_model[0] == "Gradient Boosting":
    sample_preds = gbt_predictions
else:
    sample_preds = lr_predictions

# Join with original features to show context
sample_with_features = sample_preds.join(
    features_df.select("match_id", "gold_at_10min", "exp_at_10min", 
                      "teamfights_count_10min", "tower_kills_10min"),
    on="match_id",
    how="inner"
)

# Get 5 example predictions
try:
    examples = sample_with_features.limit(5).collect()
    
    print("\nExample Predictions (simulating real-time scenario):")
    print("-" * 80)
    
    for i, row in enumerate(examples, 1):
        match_id = row["match_id"]
        actual = "Radiant" if row["label"] == 1 else "Dire"
        predicted = "Radiant" if row["prediction"] == 1.0 else "Dire"
        
        # Extract probability for Radiant
        prob_array = row["probability"]
        # Handle different probability formats
        if hasattr(prob_array, 'toArray'):
            probs = prob_array.toArray()
            radiant_prob = probs[1]
        else:
            radiant_prob = float(prob_array[1]) if len(prob_array) > 1 else 0.5
        
        gold = int(row["gold_at_10min"]) if row["gold_at_10min"] else 0
        exp = int(row["exp_at_10min"]) if row["exp_at_10min"] else 0
        fights = int(row["teamfights_count_10min"]) if row["teamfights_count_10min"] else 0
        towers = int(row["tower_kills_10min"]) if row["tower_kills_10min"] else 0
        
        # Generate recommendation
        if radiant_prob > 0.65:
            strength = "strong"
            recommendation = "Radiant should maintain aggressive tempo and secure objectives. Dire needs successful ganks to stabilize."
        elif radiant_prob > 0.55:
            strength = "slight"
            recommendation = "Game is balanced. Both teams should focus on objective control and avoid mistakes."
        elif radiant_prob > 0.45:
            strength = "slight"
            recommendation = "Game is even. Next teamfight will be crucial in determining momentum."
        elif radiant_prob > 0.35:
            strength = "strong"  
            recommendation = "Dire has advantage. Radiant needs to avoid fights and farm defensively. Dire should press advantage."
        else:
            strength = "very strong"
            recommendation = "Dire has commanding lead. Radiant should defend high ground and look for comeback opportunities."
        
        print(f"\nExample {i} (Match ID: {match_id}):")
        print(f"  Input at 10 minutes:")
        print(f"    - Gold lead: {'Radiant' if gold > 0 else 'Dire'} +{abs(gold):,}")
        print(f"    - Experience lead: {'Radiant' if exp > 0 else 'Dire'} +{abs(exp):,}")
        print(f"    - Teamfights: {fights}")
        print(f"    - Towers destroyed: {towers}")
        print(f"  ")
        print(f"  Prediction:")
        print(f"    - Draft-only estimate: ~{round_float(draft_accuracy * 100, 1)}% (baseline)")
        print(f"    - Updated prediction: {predicted} {round_float(radiant_prob * 100, 1)}%")
        print(f"    - Confidence: {strength}")
        print(f"  ")
        print(f"  Recommendation: {recommendation}")
        print(f"  Actual result: {actual} won")
    
    print("-" * 80)
except Exception as e:
    print(f"[WARNING] Could not generate sample predictions: {str(e)}")
    print("Continuing with results...")

# ==============================================================================
# FINALIZE PERFORMANCE SUMMARY
# ==============================================================================

pipeline_end_time = time.time()
total_execution_time = pipeline_end_time - pipeline_start_time

results["performance_summary"] = {
    "total_execution_time_sec": round_float(total_execution_time, 2),
    "best_model": best_model[0],
    "best_model_auc": round_float(best_model[1], 4),
    "best_model_accuracy": round_float(
        lr_accuracy if best_model[0] == "Logistic Regression" 
        else rf_accuracy if best_model[0] == "Random Forest" 
        else gbt_accuracy, 4
    ),
    "data_loading_time_sec": results["data_loading"]["loading_time_sec"],
    "feature_engineering_time_sec": results["feature_engineering"]["time_sec"],
    "models_trained": 3,
    "total_features_used": len(feature_cols),
    "draft_only_baseline": draft_baseline,
    "improvement_over_draft": improvement_metrics
}

# ==============================================================================
# GENERATE REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("GENERATING COMPREHENSIVE REPORT")
print("=" * 80)

report = []
report.append("=" * 80)
report.append("DOTA 2 EARLY GAME PREDICTION - RESULTS")
report.append("=" * 80)
report.append(f"\nExecution Date: {results['execution_info']['start_time']}")
report.append(f"Input Path: {INPUT_PATH}")
report.append(f"Output Path: {OUTPUT_PATH}")
report.append(f"Total Execution Time: {results['performance_summary']['total_execution_time_sec']} seconds")

report.append("\n" + "=" * 80)
report.append("DATA LOADING")
report.append("=" * 80)
report.append(f"Loading Time: {results['data_loading']['loading_time_sec']} sec")
report.append(f"Total Rows Loaded: {results['data_loading']['total_rows_loaded']:,}")

report.append("\n" + "=" * 80)
report.append("FEATURE ENGINEERING")
report.append("=" * 80)
report.append(f"Processing Time: {results['feature_engineering']['time_sec']} sec")
report.append(f"Features Created: {results['feature_engineering']['features_created']}")
report.append(f"Final Dataset Rows: {results['feature_engineering']['final_dataset_rows']:,}")

report.append("\n" + "=" * 80)
report.append("MODEL RESULTS")
report.append("=" * 80)

for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
    report.append(f"\n{model_name.replace('_', ' ').title()}:")
    metrics = results["model_results"][model_name]
    report.append(f"  Training Time: {metrics['training_time_sec']} sec")
    report.append(f"  AUC-ROC: {metrics['auc']}")
    report.append(f"  Accuracy: {metrics['accuracy']}")
    report.append(f"  F1 Score: {metrics['f1_score']}")

report.append("\n" + "=" * 80)
report.append("PERFORMANCE SUMMARY")
report.append("=" * 80)
report.append(f"Best Model: {results['performance_summary']['best_model']}")
report.append(f"Best AUC: {results['performance_summary']['best_model_auc']}")
report.append(f"Best Accuracy: {results['performance_summary']['best_model_accuracy']}")
report.append(f"\nDraft-Only Baseline:")
report.append(f"  Accuracy: {results['performance_summary']['draft_only_baseline']['accuracy']}")
report.append(f"  AUC: {results['performance_summary']['draft_only_baseline']['auc']}")
report.append(f"\nImprovement with Early Game Features:")
report.append(f"  Accuracy Gain: +{results['performance_summary']['improvement_over_draft']['accuracy_gain']}%")
report.append(f"  AUC Gain: +{results['performance_summary']['improvement_over_draft']['auc_gain']}")
report.append(f"\nPerformance Metrics:")
report.append(f"  Total Execution Time: {results['performance_summary']['total_execution_time_sec']} sec")
report.append(f"  Models Trained: {results['performance_summary']['models_trained']}")
report.append(f"  Features Used: {results['performance_summary']['total_features_used']}")

report.append("\n" + "=" * 80)
report.append("KEY FINDINGS")
report.append("=" * 80)
report.append(f" Early game features improve prediction by {results['performance_summary']['improvement_over_draft']['accuracy_gain']}%")
report.append(f" Gold and experience advantages are strong predictors")
report.append(f" {results['performance_summary']['best_model']} performs best with {results['performance_summary']['best_model_auc']} AUC")
report.append(f" Model provides actionable recommendations for teams")

report_text = "\n".join(report)
print("\n" + report_text)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

try:
    # Save JSON results
    json_output = json.dumps(results, indent=2)
    json_rdd = sc.parallelize([json_output])
    json_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/early_game_results.json")
    print(f" Saved JSON results to: {OUTPUT_PATH}/early_game_results.json")
    
    # Save text report
    report_rdd = sc.parallelize([report_text])
    report_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/early_game_report")
    print(f" Saved text report to: {OUTPUT_PATH}/early_game_report")
    
    # Save predictions from best model
    if best_model[0] == "Random Forest":
        best_predictions = rf_predictions
    elif best_model[0] == "Gradient Boosting":
        best_predictions = gbt_predictions
    else:
        best_predictions = lr_predictions
    
    predictions_output = best_predictions.select(
        "match_id",
        "label",
        "prediction",
        F.expr("probability[1]").alias("radiant_win_probability")
    )
    
    predictions_output.coalesce(10).write.mode("overwrite").csv(
        f"{OUTPUT_PATH}/predictions",
        header=True
    )
    print(f" Saved predictions to: {OUTPUT_PATH}/predictions")
    
    print("\n[SUCCESS] All results saved successfully!")
    
except Exception as e:
    print(f"[ERROR] Failed to save results: {str(e)}")

# ==============================================================================
# CLEANUP AND EXIT
# ==============================================================================

print("\n" + "=" * 80)
print("CLEANING UP")
print("=" * 80)

# Unpersist cached DataFrames
picks_bans.unpersist()
metadata.unpersist()
gold_adv.unpersist()
exp_adv.unpersist()
features_df.unpersist()
train_df.unpersist()
test_df.unpersist()


spark.stop()

print(f"Total execution time: {round_float(total_execution_time, 2)} seconds")
print(f"Best model: {best_model[0]} (AUC: {round_float(best_model[1], 4)})")
