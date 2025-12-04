#!/usr/bin/env python3
"""
Enhanced Dota 2 Match Outcome Prediction Model - Google Cloud Version
Features:
- Hero picks/bans (one-hot encoding)
- Hero synergy matrix (2-hero combination win rates)
- Pick order features
- Pick timing features
- Patch version features

Models: Logistic Regression, Random Forest, Gradient Boosting
Target: Accuracy ≥65%, AUC-ROC ≥0.70
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import os
import time
from itertools import combinations


# ============================================================
# Configuration - MODIFY THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================
class Config:
    # Data paths - Google Cloud Storage bucket
    DATA_DIR = "gs://bucket0820/dota_data"
    OUTPUT_DIR = "gs://bucket0820/output"

    # Year range
    START_YEAR = 2016
    END_YEAR = 2024

    # Model parameters
    RANDOM_SEED = 42
    TEST_SPLIT = 0.2

    # Target metrics
    TARGET_ACCURACY = 0.65
    TARGET_AUC = 0.70

    # Synergy calculation
    MIN_GAMES_FOR_SYNERGY = 10


# ============================================================
# Spark Session - Optimized for Google Cloud Storage
# ============================================================
def create_spark_session():
    return SparkSession.builder \
        .appName("Dota2 Enhanced Prediction") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()


# ============================================================
# Data Loading Functions (Multi-Year)
# ============================================================
def get_year_files(base_dir, file_pattern):
    """Find files matching pattern for each year"""
    files = []
    for year in range(Config.START_YEAR, Config.END_YEAR + 1):
        filename = file_pattern.format(year=year)
        filepath = base_dir + "/" + filename  # Use / for GCS paths
        files.append((year, filepath))
    return files


def load_picks_bans_multi_year(spark):
    """Load picks_bans data from multiple years"""
    print("[INFO] Loading picks_bans data...")

    year_files = get_year_files(Config.DATA_DIR, "picks_bans_{year}.csv")

    # Common columns we need
    required_cols = ["is_pick", "hero_id", "team", "order", "match_id", "leagueid"]

    dfs = []
    for year, filepath in year_files:
        try:
            df = spark.read.csv(filepath, header=True, inferSchema=True)

            # Select only required columns that exist
            existing_cols = [c for c in required_cols if c in df.columns]
            df = df.select(*existing_cols)

            # Add missing columns with null
            for col in required_cols:
                if col not in df.columns:
                    df = df.withColumn(col, F.lit(None))

            df = df.select(*required_cols)
            df = df.withColumn("year", F.lit(year))

            row_count = df.count()
            dfs.append(df)
            print(f"  - Loaded {year}: {row_count} rows")
        except Exception as e:
            print(f"  - Skipping {year}: {e}")

    if not dfs:
        raise FileNotFoundError("No picks_bans data loaded")

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.union(df)

    print(f"[INFO] Total picks_bans rows: {combined.count()}")
    return combined


def load_metadata_multi_year(spark):
    """Load metadata from multiple years"""
    print("[INFO] Loading metadata...")

    year_files = get_year_files(Config.DATA_DIR, "main_metadata_{year}.csv")

    required_cols = ["match_id", "radiant_win"]
    optional_cols = ["patch", "leagueid", "start_time"]

    dfs = []
    for year, filepath in year_files:
        try:
            df = spark.read.csv(filepath, header=True, inferSchema=True)

            select_cols = [c for c in required_cols if c in df.columns]
            for col in optional_cols:
                if col in df.columns:
                    select_cols.append(col)

            df = df.select(*select_cols)
            df = df.withColumn("year", F.lit(year))

            row_count = df.count()
            dfs.append(df)
            print(f"  - Loaded {year}: {row_count} rows")
        except Exception as e:
            print(f"  - Skipping {year}: {e}")

    if not dfs:
        raise FileNotFoundError("No metadata loaded")

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.unionByName(df, allowMissingColumns=True)

    print(f"[INFO] Total metadata rows: {combined.count()}")
    return combined


def load_draft_timings_multi_year(spark):
    """Load draft timings from multiple years"""
    print("[INFO] Loading draft timings...")

    year_files = get_year_files(Config.DATA_DIR, "draft_timings_{year}.csv")

    required_cols = ["match_id", "pick", "total_time_taken", "extra_time"]

    dfs = []
    for year, filepath in year_files:
        try:
            df = spark.read.csv(filepath, header=True, inferSchema=True)

            existing_cols = [c for c in required_cols if c in df.columns]

            if "match_id" not in existing_cols:
                print(f"  - Skipping {year}: no match_id column")
                continue

            df = df.select(*existing_cols)

            if "pick" not in df.columns:
                df = df.withColumn("pick", F.lit(True))
            if "total_time_taken" not in df.columns:
                df = df.withColumn("total_time_taken", F.lit(0))
            if "extra_time" not in df.columns:
                df = df.withColumn("extra_time", F.lit(0))

            df = df.select(*required_cols)
            df = df.withColumn("year", F.lit(year))

            row_count = df.count()
            dfs.append(df)
            print(f"  - Loaded {year}: {row_count} rows")
        except Exception as e:
            print(f"  - Skipping {year}: {e}")

    if not dfs:
        print("  - No draft timings data available")
        return None

    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.union(df)

    print(f"[INFO] Total draft_timings rows: {combined.count()}")
    return combined


# ============================================================
# Feature Engineering
# ============================================================
def create_hero_pick_features(picks_bans_df, spark):
    """Create one-hot encoded hero pick/ban features"""
    print("[INFO] Creating hero pick/ban features...")

    # Cast hero_id to integer
    picks_bans_df = picks_bans_df.withColumn("hero_id", F.col("hero_id").cast("int"))

    # Cache for multiple uses
    picks_bans_df.cache()

    hero_ids = picks_bans_df.select("hero_id").distinct().collect()
    hero_ids = sorted([int(row.hero_id) for row in hero_ids if row.hero_id is not None])
    print(f"  - Found {len(hero_ids)} unique heroes")

    # Radiant picks
    radiant_picks = picks_bans_df.filter(
        (F.col("is_pick") == True) & (F.col("team") == 0)
    ).groupBy("match_id").pivot("hero_id", hero_ids).agg(F.lit(1)).fillna(0)

    for col_name in radiant_picks.columns:
        if col_name != "match_id":
            clean_name = f"radiant_pick_{col_name}".replace(".", "_")
            radiant_picks = radiant_picks.withColumnRenamed(col_name, clean_name)

    # Dire picks
    dire_picks = picks_bans_df.filter(
        (F.col("is_pick") == True) & (F.col("team") == 1)
    ).groupBy("match_id").pivot("hero_id", hero_ids).agg(F.lit(1)).fillna(0)

    for col_name in dire_picks.columns:
        if col_name != "match_id":
            clean_name = f"dire_pick_{col_name}".replace(".", "_")
            dire_picks = dire_picks.withColumnRenamed(col_name, clean_name)

    # Radiant bans
    radiant_bans = picks_bans_df.filter(
        (F.col("is_pick") == False) & (F.col("team") == 0)
    ).groupBy("match_id").pivot("hero_id", hero_ids).agg(F.lit(1)).fillna(0)

    for col_name in radiant_bans.columns:
        if col_name != "match_id":
            clean_name = f"radiant_ban_{col_name}".replace(".", "_")
            radiant_bans = radiant_bans.withColumnRenamed(col_name, clean_name)

    # Dire bans
    dire_bans = picks_bans_df.filter(
        (F.col("is_pick") == False) & (F.col("team") == 1)
    ).groupBy("match_id").pivot("hero_id", hero_ids).agg(F.lit(1)).fillna(0)

    for col_name in dire_bans.columns:
        if col_name != "match_id":
            clean_name = f"dire_ban_{col_name}".replace(".", "_")
            dire_bans = dire_bans.withColumnRenamed(col_name, clean_name)

    # Join all features
    features_df = radiant_picks
    for df in [dire_picks, radiant_bans, dire_bans]:
        features_df = features_df.join(df, on="match_id", how="outer")

    features_df = features_df.fillna(0)
    print(f"  - Created {len(features_df.columns) - 1} hero pick/ban features")

    picks_bans_df.unpersist()

    return features_df, hero_ids


def create_pick_order_features(picks_bans_df):
    """Create pick order features"""
    print("[INFO] Creating pick order features...")

    picks_only = picks_bans_df.filter(F.col("is_pick") == True)

    window_spec = Window.partitionBy("match_id", "team").orderBy("order")
    picks_with_order = picks_only.withColumn("pick_order", F.row_number().over(window_spec))

    pick_order_stats = picks_with_order.groupBy("match_id", "team").agg(
        F.avg("pick_order").alias("avg_pick_order"),
        F.first("hero_id").alias("first_pick_hero"),
        F.last("hero_id").alias("last_pick_hero")
    )

    radiant_order = pick_order_stats.filter(F.col("team") == 0).select(
        "match_id",
        F.col("avg_pick_order").alias("radiant_avg_pick_order"),
        F.col("first_pick_hero").alias("radiant_first_pick"),
        F.col("last_pick_hero").alias("radiant_last_pick")
    )

    dire_order = pick_order_stats.filter(F.col("team") == 1).select(
        "match_id",
        F.col("avg_pick_order").alias("dire_avg_pick_order"),
        F.col("first_pick_hero").alias("dire_first_pick"),
        F.col("last_pick_hero").alias("dire_last_pick")
    )

    order_features = radiant_order.join(dire_order, on="match_id", how="outer")
    order_features = order_features.fillna(0)

    print(f"  - Created pick order features")
    return order_features


def create_timing_features(draft_timings_df):
    """Create timing-based features"""
    print("[INFO] Creating timing features...")

    if draft_timings_df is None:
        print("  - No timing data available")
        return None

    picks_only = draft_timings_df.filter(F.col("pick") == True)

    timing_features = picks_only.groupBy("match_id").agg(
        F.avg("total_time_taken").alias("avg_pick_time"),
        F.max("total_time_taken").alias("max_pick_time"),
        F.min("total_time_taken").alias("min_pick_time"),
        F.stddev("total_time_taken").alias("std_pick_time"),
        F.sum("extra_time").alias("total_extra_time")
    )

    timing_features = timing_features.fillna(0)
    print(f"  - Created timing features for {timing_features.count()} matches")

    return timing_features


def calculate_hero_synergy(picks_bans_df, metadata_df):
    """Calculate hero synergy matrix"""
    print("[INFO] Calculating hero synergy matrix...")

    picks_with_result = picks_bans_df.filter(
        F.col("is_pick") == True
    ).join(
        metadata_df.select("match_id", "radiant_win"),
        on="match_id",
        how="inner"
    )

    picks_with_result = picks_with_result.withColumn(
        "won",
        F.when(
            ((F.col("team") == 0) & (F.col("radiant_win") == True)) |
            ((F.col("team") == 1) & (F.col("radiant_win") == False)),
            1
        ).otherwise(0)
    )

    # Get hero pairs per match/team
    heroes_per_team = picks_with_result.groupBy("match_id", "team", "won").agg(
        F.collect_list("hero_id").alias("heroes")
    )

    # Calculate synergy for pairs
    @F.udf("array<struct<hero1:int,hero2:int>>")
    def get_pairs(heroes):
        if heroes is None or len(heroes) < 2:
            return []
        heroes = sorted([int(h) for h in heroes if h is not None])
        return [(heroes[i], heroes[j]) for i in range(len(heroes)) for j in range(i + 1, len(heroes))]

    hero_pairs = heroes_per_team.withColumn("pairs", get_pairs(F.col("heroes")))
    hero_pairs = hero_pairs.withColumn("pair", F.explode("pairs"))
    hero_pairs = hero_pairs.select(
        F.col("pair.hero1").alias("hero1"),
        F.col("pair.hero2").alias("hero2"),
        "won"
    )

    synergy_stats = hero_pairs.groupBy("hero1", "hero2").agg(
        F.count("*").alias("games"),
        F.sum("won").alias("wins")
    ).filter(F.col("games") >= Config.MIN_GAMES_FOR_SYNERGY)

    synergy_stats = synergy_stats.withColumn(
        "win_rate",
        F.col("wins") / F.col("games")
    )

    synergy_count = synergy_stats.count()
    print(f"  - Calculated synergy for {synergy_count} hero pairs")

    return synergy_stats


def apply_synergy_features(features_df, synergy_stats, picks_bans_df, spark):
    """Apply synergy scores to matches"""
    print("[INFO] Applying synergy features...")

    if synergy_stats is None or synergy_stats.count() == 0:
        features_df = features_df.withColumn("radiant_synergy", F.lit(0.5))
        features_df = features_df.withColumn("dire_synergy", F.lit(0.5))
        features_df = features_df.withColumn("synergy_diff", F.lit(0.0))
        return features_df

    # Create synergy lookup
    synergy_dict = {}
    for row in synergy_stats.collect():
        key = (int(row.hero1), int(row.hero2))
        synergy_dict[key] = float(row.win_rate)
        synergy_dict[(int(row.hero2), int(row.hero1))] = float(row.win_rate)

    synergy_broadcast = spark.sparkContext.broadcast(synergy_dict)

    # Get heroes per team
    radiant_heroes = picks_bans_df.filter(
        (F.col("is_pick") == True) & (F.col("team") == 0)
    ).groupBy("match_id").agg(
        F.collect_list("hero_id").alias("radiant_heroes")
    )

    dire_heroes = picks_bans_df.filter(
        (F.col("is_pick") == True) & (F.col("team") == 1)
    ).groupBy("match_id").agg(
        F.collect_list("hero_id").alias("dire_heroes")
    )

    @F.udf("double")
    def calc_team_synergy(heroes):
        if heroes is None or len(heroes) < 2:
            return 0.5
        heroes = [int(h) for h in heroes if h is not None]
        syn_dict = synergy_broadcast.value
        total_syn = 0.0
        count = 0
        for i in range(len(heroes)):
            for j in range(i + 1, len(heroes)):
                key = (heroes[i], heroes[j])
                if key in syn_dict:
                    total_syn += syn_dict[key]
                    count += 1
        return total_syn / count if count > 0 else 0.5

    radiant_syn = radiant_heroes.withColumn(
        "radiant_synergy", calc_team_synergy(F.col("radiant_heroes"))
    ).select("match_id", "radiant_synergy")

    dire_syn = dire_heroes.withColumn(
        "dire_synergy", calc_team_synergy(F.col("dire_heroes"))
    ).select("match_id", "dire_synergy")

    features_df = features_df.join(radiant_syn, on="match_id", how="left")
    features_df = features_df.join(dire_syn, on="match_id", how="left")

    features_df = features_df.fillna(0.5, subset=["radiant_synergy", "dire_synergy"])
    features_df = features_df.withColumn(
        "synergy_diff",
        F.col("radiant_synergy") - F.col("dire_synergy")
    )

    print(f"  - Added synergy features")
    return features_df


def add_patch_features(features_df, metadata_df):
    """Add patch/version features"""
    print("[INFO] Adding patch features...")

    if "patch" in metadata_df.columns:
        patch_data = metadata_df.select("match_id", "patch")
        features_df = features_df.join(patch_data, on="match_id", how="left")
        features_df = features_df.fillna(0, subset=["patch"])
        print(f"  - Added patch feature")
    elif "year" in metadata_df.columns:
        year_data = metadata_df.select("match_id", F.col("year").alias("patch_year"))
        features_df = features_df.join(year_data, on="match_id", how="left")
        features_df = features_df.fillna(2020, subset=["patch_year"])
        print(f"  - Added year as patch proxy")
    else:
        print("  - No patch data available")

    return features_df


# ============================================================
# Model Training
# ============================================================
def prepare_training_data(features_df, metadata_df):
    """Prepare final training dataset"""
    print("[INFO] Preparing training data...")

    # Clean column names
    for col_name in features_df.columns:
        if "." in col_name:
            clean_name = col_name.replace(".", "_")
            features_df = features_df.withColumnRenamed(col_name, clean_name)

    labels = metadata_df.select(
        "match_id",
        F.when(F.col("radiant_win") == True, 1.0).otherwise(0.0).alias("label")
    )

    data = features_df.join(labels, on="match_id", how="inner")

    feature_cols = [c for c in data.columns if c not in ["match_id", "label"]]

    data = data.fillna(0)

    for col_name in feature_cols:
        data = data.withColumn(col_name, F.col(col_name).cast("double"))

    # Cache the data for multiple model training
    data.cache()

    sample_count = data.count()
    print(f"  - Total samples: {sample_count}")
    print(f"  - Total features: {len(feature_cols)}")

    label_dist = data.groupBy("label").count().collect()
    for row in label_dist:
        print(f"  - Label {int(row.label)}: {row['count']} samples")

    return data, feature_cols


def assemble_features(data, feature_cols):
    """Assemble features into vector"""
    print("[INFO] Assembling feature vector...")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    data = assembler.transform(data)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    scaler_model = scaler.fit(data)
    data = scaler_model.transform(data)

    return data, scaler_model


def train_models(train_data, test_data):
    """Train all three models"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    models = {}
    results = {}

    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    start_time = time.time()

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01
    )
    lr_model = lr.fit(train_data)

    lr_time = time.time() - start_time
    print(f"  - Training time: {lr_time:.2f}s")

    models["logistic_regression"] = lr_model

    # 2. Random Forest
    print("\n[2/3] Training Random Forest...")
    start_time = time.time()

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=Config.RANDOM_SEED
    )
    rf_model = rf.fit(train_data)

    rf_time = time.time() - start_time
    print(f"  - Training time: {rf_time:.2f}s")

    models["random_forest"] = rf_model

    # 3. Gradient Boosting
    print("\n[3/3] Training Gradient Boosting...")
    start_time = time.time()

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        maxDepth=5,
        stepSize=0.1,
        seed=Config.RANDOM_SEED
    )
    gbt_model = gbt.fit(train_data)

    gbt_time = time.time() - start_time
    print(f"  - Training time: {gbt_time:.2f}s")

    models["gradient_boosting"] = gbt_model

    return models


def evaluate_models(models, test_data, feature_cols):
    """Evaluate all models"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = {}

    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    for name, model in models.items():
        print(f"\n--- {name.upper()} ---")

        predictions = model.transform(test_data)

        auc_roc = binary_evaluator.evaluate(predictions)
        accuracy = accuracy_evaluator.evaluate(predictions)

        # Precision, Recall, F1
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        )
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        )
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )

        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # Confusion matrix
        pred_and_labels = predictions.select("prediction", "label").rdd.map(
            lambda row: (float(row.prediction), float(row.label))
        )
        metrics = MulticlassMetrics(pred_and_labels)
        cm = metrics.confusionMatrix().toArray()
        print(f"  Confusion Matrix:")
        print(f"    TN={int(cm[0][0])}, FP={int(cm[0][1])}")
        print(f"    FN={int(cm[1][0])}, TP={int(cm[1][1])}")

        # Check targets
        accuracy_met = "✓" if accuracy >= Config.TARGET_ACCURACY else "✗"
        auc_met = "✓" if auc_roc >= Config.TARGET_AUC else "✗"
        print(f"  Target Accuracy (≥{Config.TARGET_ACCURACY}): {accuracy_met}")
        print(f"  Target AUC-ROC (≥{Config.TARGET_AUC}): {auc_met}")

        results[name] = {
            "accuracy": accuracy,
            "auc_roc": auc_roc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions
        }

        # Feature importance for tree models
        if hasattr(model, "featureImportances"):
            print_feature_importance(model, feature_cols, name)

    return results


def print_feature_importance(model, feature_cols, model_name):
    """Print top 20 most important features"""
    print(f"\n  Top 20 Feature Importance ({model_name}):")

    importances = model.featureImportances.toArray()
    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(feature_importance[:20]):
        print(f"    {i + 1}. {feature}: {importance:.6f}")


# ============================================================
# Export Results
# ============================================================
def export_results(results, models, feature_cols, spark):
    """Export results to CSV files"""
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    # GCS doesn't need makedirs - directories are virtual

    # 1. Model comparison
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            "model": name,
            "accuracy": result["accuracy"],
            "auc_roc": result["auc_roc"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"]
        })

    comparison_df = spark.createDataFrame(comparison_data)
    comparison_path = Config.OUTPUT_DIR + "/model_comparison.csv"
    comparison_df.coalesce(1).write.mode("overwrite").option("header", True).csv(comparison_path)
    print(f"  - Saved model comparison to {comparison_path}")

    # 2. Feature importance for tree models
    for name in ["random_forest", "gradient_boosting"]:
        if name in models and hasattr(models[name], "featureImportances"):
            importances = models[name].featureImportances.toArray()
            fi_data = [{"feature": f, "importance": float(i)} for f, i in zip(feature_cols, importances)]
            fi_df = spark.createDataFrame(fi_data)
            fi_df = fi_df.orderBy(F.desc("importance"))
            fi_path = Config.OUTPUT_DIR + f"/feature_importance_{name}.csv"
            fi_df.coalesce(1).write.mode("overwrite").option("header", True).csv(fi_path)
            print(f"  - Saved {name} feature importance to {fi_path}")

    # 3. Logistic Regression coefficients
    if "logistic_regression" in models:
        lr_model = models["logistic_regression"]
        coefficients = lr_model.coefficients.toArray()
        coef_data = [{"feature": f, "coefficient": float(c), "abs_coefficient": abs(float(c))}
                     for f, c in zip(feature_cols, coefficients)]
        coef_df = spark.createDataFrame(coef_data)
        coef_df = coef_df.orderBy(F.desc("abs_coefficient"))
        coef_path = Config.OUTPUT_DIR + "/logistic_regression_coefficients.csv"
        coef_df.coalesce(1).write.mode("overwrite").option("header", True).csv(coef_path)
        print(f"  - Saved LR coefficients to {coef_path}")

        # Intercept - save as DataFrame for GCS compatibility
        intercept_data = [{"intercept": float(lr_model.intercept)}]
        intercept_df = spark.createDataFrame(intercept_data)
        intercept_path = Config.OUTPUT_DIR + "/logistic_regression_intercept.csv"
        intercept_df.coalesce(1).write.mode("overwrite").option("header", True).csv(intercept_path)
        print(f"  - Saved LR intercept to {intercept_path}")

    # 4. Find best model and save predictions
    best_model_name = max(results, key=lambda x: results[x]["auc_roc"])
    print(f"\n  Best model by AUC-ROC: {best_model_name}")

    best_predictions = results[best_model_name]["predictions"]

    try:
        pred_export = best_predictions.select(
            "match_id",
            "label",
            "prediction",
            "rawPrediction"
        )
        pred_path = Config.OUTPUT_DIR + "/predictions.csv"
        pred_export.coalesce(1).write.mode("overwrite").option("header", True).csv(pred_path)
        print(f"  - Saved predictions to {pred_path}")
    except Exception as e:
        print(f"  - Could not save predictions: {e}")

    # 5. Save best model
    best_model = models[best_model_name]
    model_path = Config.OUTPUT_DIR + f"/best_model_{best_model_name}"
    best_model.write().overwrite().save(model_path)
    print(f"  - Saved best model to {model_path}")

    return best_model_name


# ============================================================
# Main Function
# ============================================================
def main():
    print("=" * 60)
    print("DOTA 2 ENHANCED MATCH OUTCOME PREDICTION")
    print("=" * 60)
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Year range: {Config.START_YEAR} - {Config.END_YEAR}")
    print()

    try:
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        # Load data
        picks_bans_df = load_picks_bans_multi_year(spark)
        metadata_df = load_metadata_multi_year(spark)
        draft_timings_df = load_draft_timings_multi_year(spark)

        # Feature engineering
        features_df, hero_ids = create_hero_pick_features(picks_bans_df, spark)

        pick_order_df = create_pick_order_features(picks_bans_df)
        features_df = features_df.join(pick_order_df, on="match_id", how="left")

        timing_features_df = create_timing_features(draft_timings_df)
        if timing_features_df is not None:
            features_df = features_df.join(timing_features_df, on="match_id", how="left")

        synergy_stats = calculate_hero_synergy(picks_bans_df, metadata_df)
        features_df = apply_synergy_features(features_df, synergy_stats, picks_bans_df, spark)

        features_df = add_patch_features(features_df, metadata_df)

        # Prepare training data
        data, feature_cols = prepare_training_data(features_df, metadata_df)

        # Assemble features
        data, scaler_model = assemble_features(data, feature_cols)

        # Split data
        train_data, test_data = data.randomSplit(
            [1 - Config.TEST_SPLIT, Config.TEST_SPLIT],
            seed=Config.RANDOM_SEED
        )

        print(f"\n[INFO] Train/Test split:")
        print(f"  - Training samples: {train_data.count()}")
        print(f"  - Test samples: {test_data.count()}")

        # Train models
        models = train_models(train_data, test_data)

        # Evaluate models
        results = evaluate_models(models, test_data, feature_cols)

        # Export results
        best_model = export_results(results, models, feature_cols, spark)

        print("\n" + "=" * 60)
        print("COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Best model: {best_model}")
        print(f"Results saved to: {Config.OUTPUT_DIR}")

        spark.stop()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()