# =====================================================================
# DOTA 2 HERO RECOMMENDATION SYSTEM — FINAL OPTIMIZED VERSION
# Course: CS777 - Big Data Analytics Term Project

# =====================================================================

from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, count, when
from pyspark.sql.types import FloatType
import time
import datetime


# -------------------------------------------------------------
def create_spark():
    return (
        SparkSession.builder
        .appName("Dota2-Hero-Recommendation")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


# -------------------------------------------------------------
def load_data(spark, base_path):

    picks = spark.read.csv(base_path + "/picks_bans.csv", header=True, inferSchema=True)
    meta  = spark.read.csv(base_path + "/main_metadata.csv", header=True, inferSchema=True)

    # Build radiant_win, dire_win
    meta = meta.withColumn(
        "radiant_win", when(col("radiant_score") > col("dire_score"), 1).otherwise(0)
    ).withColumn(
        "dire_win", when(col("dire_score") > col("radiant_score"), 1).otherwise(0)
    )

    # Join picks with match outcomes
    df = picks.join(meta.select("match_id", "radiant_win", "dire_win"), on="match_id")

    # Assign hero-level win indicator
    df = df.withColumn(
        "hero_win",
        when((col("team") == 0) & (col("radiant_win") == 1), 1)
        .when((col("team") == 1) & (col("dire_win") == 1), 1)
        .otherwise(0)
    )

    return df


# -------------------------------------------------------------
def compute_meta(df):
    total_matches = df.select("match_id").distinct().count()

    meta = (
        df.groupBy("hero_id")
        .agg(
            count("*").alias("pick_count"),
            _sum("hero_win").alias("wins")
        )
        .withColumn("win_rate", (col("wins") / col("pick_count")).cast(FloatType()))
        .withColumn("pick_rate", (col("pick_count") / total_matches).cast(FloatType()))
    )
    return meta


# -------------------------------------------------------------
def compute_synergy(df):
    rad = df.filter(col("team") == 0)

    sy = (
        rad.alias("a")
        .join(
            rad.alias("b"),
            (col("a.match_id") == col("b.match_id")) &
            (col("a.hero_id") < col("b.hero_id"))
        )
        .groupBy(
            col("a.hero_id").alias("h1"),
            col("b.hero_id").alias("h2")
        )
        .agg(_sum("a.hero_win").alias("wins"))
    )
    return sy


# -------------------------------------------------------------
def compute_counters(df):
    rad = df.filter(col("team") == 0)
    dire = df.filter(col("team") == 1)

    ctr = (
        rad.alias("r")
        .join(dire.alias("d"), col("r.match_id") == col("d.match_id"))
        .groupBy(
            col("d.hero_id").alias("counter_hero"),
            col("r.hero_id").alias("target_hero")
        )
        .agg(_sum("d.hero_win").alias("wins"))
    )
    return ctr


# -------------------------------------------------------------
# Optimized scoring using Python dictionaries
# -------------------------------------------------------------
def score_hero(hero, radiant, dire, synergy_dict, counter_dict, meta_dict):

    # META SCORE
    if hero in meta_dict:
        win_rate = meta_dict[hero]['win_rate']
        pick_rate = meta_dict[hero]['pick_rate']
        meta_score = 0.7 * win_rate + 0.3 * pick_rate
    else:
        meta_score = 0.5

    # SYNERGY LOOKUP
    syn_sum = sum(synergy_dict.get((hero, rp), 0) for rp in radiant)
    synergy_score = (syn_sum + 1) / (len(radiant) + 2)

    # COUNTER LOOKUP
    ctr_sum = sum(counter_dict.get((hero, dp), 0) for dp in dire)
    counter_score = (ctr_sum + 1) / (len(dire) + 2)

    # FINAL
    final_score = (0.4 * synergy_score) + (0.4 * counter_score) + (0.2 * meta_score)

    return final_score, synergy_score, counter_score, meta_score


# -------------------------------------------------------------
def recommend_top3(radiant, dire, synergy_dict, counter_dict, meta_dict):

    all_heroes = list(meta_dict.keys())
    used = set(radiant + dire)

    candidates = [h for h in all_heroes if h not in used]

    results = []
    for h in candidates:
        f, syn, ctr, m = score_hero(h, radiant, dire, synergy_dict, counter_dict, meta_dict)
        results.append((h, f, syn, ctr, m))

    return sorted(results, key=lambda x: x[1], reverse=True)[:3]


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():

    total_start = time.time()
    spark = create_spark()

    base_path = "gs://dota2-dataproc-bucket-utkarsh"   # DATAPROC PATH
    output_bucket = "dota2-dataproc-bucket-utkarsh"

    output = []
    output.append("\n==============================================================================")
    output.append("DOTA 2 HERO RECOMMENDATION SYSTEM — RESULTS")
    output.append("==============================================================================\n")
    output.append(f"Execution Date: {datetime.datetime.now()}\n")

    # -------------------------------------------------------------
    # LOAD DATA
    load_start = time.time()
    df = load_data(spark, base_path)
    row_count = df.count()
    load_end = time.time()

    output.append("DATA LOADING")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Rows Loaded: {row_count}")
    output.append(f"[EXECUTION TIME] load_data: {round(load_end - load_start, 3)} sec\n")

    # -------------------------------------------------------------
    # META
    meta_start = time.time()
    meta = compute_meta(df)
    meta_end = time.time()

    meta_dict = {
        int(r["hero_id"]): {
            "win_rate": float(r["win_rate"]) if r["win_rate"] is not None else 0.5,
            "pick_rate": float(r["pick_rate"]) if r["pick_rate"] is not None else 0.05
        }
        for r in meta.collect()
    }

    output.append("META STATISTICS")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Meta Heroes: {len(meta_dict)}")
    output.append(f"[EXECUTION TIME] compute_meta: {round(meta_end - meta_start, 3)} sec\n")

    # -------------------------------------------------------------
    # SYNERGY
    sy_start = time.time()
    synergy = compute_synergy(df)
    sy_end = time.time()

    synergy_dict = {}
    for r in synergy.collect():
        h1, h2, w = int(r["h1"]), int(r["h2"]), float(r["wins"])
        synergy_dict[(h1, h2)] = w
        synergy_dict[(h2, h1)] = w

    output.append("SYNERGY MATRIX")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Synergy Pairs: {len(synergy_dict)}")
    output.append(f"[EXECUTION TIME] compute_synergy: {round(sy_end - sy_start, 3)} sec\n")

    # -------------------------------------------------------------
    # COUNTERS
    ctr_start = time.time()
    counters = compute_counters(df)
    ctr_end = time.time()

    counter_dict = {
        (int(r["counter_hero"]), int(r["target_hero"])): float(r["wins"])
        for r in counters.collect()
    }

    output.append("COUNTER MATRIX")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Counter Pairs: {len(counter_dict)}")
    output.append(f"[EXECUTION TIME] compute_counters: {round(ctr_end - ctr_start, 3)} sec\n")

    # -------------------------------------------------------------
    # RECOMMENDATION
    rad = [74, 12]
    dire = [102, 45]

    rec_start = time.time()
    top3 = recommend_top3(rad, dire, synergy_dict, counter_dict, meta_dict)
    rec_end = time.time()

    output.append("TOP 3 RECOMMENDATIONS")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Radiant Picks: {rad}")
    output.append(f"Dire Picks: {dire}\n")

    for hero, score, syn, ctr, m in top3:
        output.append(f"Hero {hero} (Score: {round(score, 4)})")
        output.append(f"  • Synergy: {round(syn, 4)}")
        output.append(f"  • Counter: {round(ctr, 4)}")
        output.append(f"  • Meta:    {round(m, 4)}\n")

    output.append(f"[EXECUTION TIME] recommend_top3: {round(rec_end - rec_start, 3)} sec\n")

    # -------------------------------------------------------------
    # PERFORMANCE SUMMARY
    total_end = time.time()

    output.append("PERFORMANCE SUMMARY")
    output.append("------------------------------------------------------------------------------")
    output.append(f"Heroes Analyzed: {len(meta_dict)}")
    output.append(f"Synergy Pairs:   {len(synergy_dict)}")
    output.append(f"Counter Pairs:   {len(counter_dict)}")
    output.append(f"[EXECUTION TIME] Total Pipeline Time: {round(total_end - total_start, 3)} sec\n")

    # =====================================================================
    # WRITE FINAL OUTPUT (UTF-8) + UPLOAD TO GCS
    # =====================================================================
    final_txt = "\n".join(output)
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Save locally as UTF-8
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_txt)

    # Upload to GCS bucket /Results/
    client = storage.Client()
    bucket = client.bucket(output_bucket)
    blob = bucket.blob(f"Results/{filename}")
    blob.upload_from_filename(filename)

    print("\nFINAL MERGED OUTPUT SAVED TO:")
    print(f"gs://{output_bucket}/Results/{filename}\n")


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
