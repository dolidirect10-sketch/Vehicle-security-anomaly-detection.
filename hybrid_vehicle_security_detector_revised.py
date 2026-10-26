#!/usr/bin/env python3
"""
Hybrid Vehicle Security Detector — revised version (reduced login false alarms)
Designed for Pydroid 3 / Android and offline use after dependencies are installed.

Expected files in the same folder as this script:
  telemetry.csv
  logins.csv

Outputs:
  telemetry_alerts_revised.csv
  login_alerts_revised.csv
  detector_summary_revised.txt

Important:
- This script does NOT use the is_attack label to decide whether to alert.
- Labels are used only for evaluation, if the column exists.
- This is a synthetic-data detector prototype, not proof of real-world security.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path(__file__).resolve().parent
TELEMETRY_FILE = BASE_DIR / "telemetry.csv"
LOGIN_FILE = BASE_DIR / "logins.csv"

TELEMETRY_OUTPUT = BASE_DIR / "telemetry_alerts_revised.csv"
LOGIN_OUTPUT = BASE_DIR / "login_alerts_revised.csv"
SUMMARY_OUTPUT = BASE_DIR / "detector_summary_revised.txt"

RANDOM_STATE = 42


def as_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def isolation_forest_scores(features, contamination):
    """Return anomaly scores and flags, safely handling small/empty datasets."""
    clean = features.replace([np.inf, -np.inf], np.nan)
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.fillna(clean.median(numeric_only=True)).fillna(0.0)

    if len(clean) < 10 or clean.shape[1] == 0:
        return np.zeros(len(clean)), np.zeros(len(clean), dtype=bool)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(clean)
    scores = -model.decision_function(clean)
    flags = model.predict(clean) == -1
    return scores, flags


def evaluate_if_available(frame, label, alert_column, title, report_lines):
    if label not in frame.columns:
        report_lines.append(f"{title}: no '{label}' column; metrics not calculated.")
        return

    truth_text = frame[label].astype(str).str.strip().str.lower()
    truth = truth_text.isin(["1", "true", "yes", "attack", "anomaly"])
    predicted = frame[alert_column].astype(bool)

    report_lines.append("")
    report_lines.append(f"===== {title} EVALUATION (labels used only after detection) =====")
    report_lines.append(f"Rows evaluated: {len(frame)}")
    report_lines.append(f"Ground-truth attacks: {int(truth.sum())}")
    report_lines.append(f"Alerts: {int(predicted.sum())}")
    report_lines.append(f"Attacks detected: {int((truth & predicted).sum())}")
    report_lines.append(f"Attacks missed: {int((truth & ~predicted).sum())}")
    report_lines.append(f"False alarms on normal rows: {int((~truth & predicted).sum())}")
    report_lines.append("Confusion matrix [normal, attack] x [normal, alert]:")
    report_lines.append(str(confusion_matrix(truth, predicted, labels=[False, True])))
    report_lines.append(
        classification_report(
            truth,
            predicted,
            labels=[False, True],
            target_names=["normal", "attack"],
            zero_division=0,
        )
    )


def run_telemetry(report_lines):
    df = pd.read_csv(TELEMETRY_FILE)
    required = [
        "vehicle_id", "timestamp", "latitude", "longitude",
        "speed_kmh", "gps_accuracy_m", "signal_strength_dbm",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"telemetry.csv is missing required columns: {missing}")

    # Keep original row order for output, while calculating features in time order.
    df["_original_order"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = as_numeric(
        df,
        ["latitude", "longitude", "speed_kmh", "gps_accuracy_m", "signal_strength_dbm"],
    )
    df = df.sort_values(["vehicle_id", "timestamp"], kind="stable").reset_index(drop=True)

    grouped = df.groupby("vehicle_id", sort=False)
    previous_lat = grouped["latitude"].shift(1)
    previous_lon = grouped["longitude"].shift(1)
    previous_time = grouped["timestamp"].shift(1)

    elapsed_hours = (df["timestamp"] - previous_time).dt.total_seconds() / 3600.0
    valid_pair = (
        previous_time.notna()
        & df["timestamp"].notna()
        & elapsed_hours.gt(0)
        & df["latitude"].notna()
        & df["longitude"].notna()
        & previous_lat.notna()
        & previous_lon.notna()
    )

    # Haversine distance in km; coordinate values are not fed directly to the model.
    lat1 = np.radians(previous_lat.astype(float))
    lon1 = np.radians(previous_lon.astype(float))
    lat2 = np.radians(df["latitude"].astype(float))
    lon2 = np.radians(df["longitude"].astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine_a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    haversine_a = haversine_a.clip(0.0, 1.0)
    distance_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(haversine_a))

    df["derived_speed_kmh"] = np.nan
    df.loc[valid_pair, "derived_speed_kmh"] = (
        distance_km.loc[valid_pair] / elapsed_hours.loc[valid_pair]
    )
    df["speed_delta_kmh"] = np.nan
    valid_speed = valid_pair & df["speed_kmh"].notna() & df["derived_speed_kmh"].notna()
    df.loc[valid_speed, "speed_delta_kmh"] = (
        df.loc[valid_speed, "speed_kmh"] - df.loc[valid_speed, "derived_speed_kmh"]
    ).abs()

    df["speed_change_kmh"] = grouped["speed_kmh"].diff().abs()
    df["rolling_speed_mean"] = (
        df.groupby("vehicle_id", sort=False)["speed_kmh"]
        .transform(lambda values: values.rolling(window=5, min_periods=2).mean())
    )
    df["speed_from_rolling_mean"] = (df["speed_kmh"] - df["rolling_speed_mean"]).abs()
    df["signal_change_dbm"] = (
        df.groupby("vehicle_id", sort=False)["signal_strength_dbm"].diff().abs()
    )

    # Important correction: do not use absolute latitude/longitude as anomaly features.
    model_features = [
        "speed_kmh", "gps_accuracy_m", "signal_strength_dbm",
        "derived_speed_kmh", "speed_delta_kmh", "speed_change_kmh",
        "speed_from_rolling_mean", "signal_change_dbm",
    ]
    scores, model_flags = isolation_forest_scores(df[model_features], contamination=0.01)
    df["model_anomaly_score"] = scores
    df["model_anomaly"] = model_flags

    # Rules. A speed mismatch is checked only when there is a valid GPS pair.
    reasons = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        if pd.notna(row["gps_accuracy_m"]) and row["gps_accuracy_m"] >= 100:
            reasons[i].append("poor_gps_accuracy")
        if pd.notna(row["signal_strength_dbm"]) and row["signal_strength_dbm"] <= -105:
            reasons[i].append("weak_signal")
        if pd.notna(row["speed_kmh"]) and row["speed_kmh"] > 160:
            reasons[i].append("extreme_reported_speed")
        if pd.notna(row["speed_delta_kmh"]) and row["speed_delta_kmh"] > 45:
            reasons[i].append("reported_vs_gps_speed_mismatch")

    df["rule_alert"] = [bool(items) for items in reasons]
    df["model_only_alert"] = df["model_anomaly"] & ~df["rule_alert"]
    for i, flag in enumerate(model_flags):
        if flag:
            reasons[i].append("isolation_forest_anomaly")

    df["alert"] = df["rule_alert"] | df["model_anomaly"]
    df["alert_reason"] = [";".join(items) if items else "none" for items in reasons]
    df["severity"] = "NONE"
    df.loc[df["alert"], "severity"] = "LOW"
    high_mask = (
        df["alert_reason"].str.contains("extreme_reported_speed|reported_vs_gps_speed_mismatch", regex=True)
    )
    df.loc[df["alert"] & ~high_mask, "severity"] = "MEDIUM"
    df.loc[df["alert"] & high_mask, "severity"] = "HIGH"

    evaluate_if_available(df, "is_attack", "alert", "TELEMETRY", report_lines)
    report_lines.append(f"Telemetry rows processed: {len(df)}")
    report_lines.append(f"Telemetry rule alerts: {int(df['rule_alert'].sum())}")
    report_lines.append(f"Telemetry model anomalies: {int(df['model_anomaly'].sum())}")
    report_lines.append(f"Telemetry model-only alerts: {int(df['model_only_alert'].sum())}")
    df.sort_values("_original_order").drop(columns=["_original_order"]).to_csv(
        TELEMETRY_OUTPUT, index=False
    )


def run_logins(report_lines):
    df = pd.read_csv(LOGIN_FILE)
    required = ["vehicle_id", "timestamp", "login_status", "device_type", "source_ip"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"logins.csv is missing required columns: {missing}")

    df["_original_order"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ["login_status", "device_type", "source_ip"]:
        df[column] = df[column].fillna("").astype(str).str.strip().str.lower()

    df = df.sort_values(["vehicle_id", "timestamp"], kind="stable").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["night_login"] = (df["hour"] < 5) | (df["hour"] >= 23)
    df["failed_login"] = df["login_status"].isin(["failed", "failure", "denied", "unsuccessful"])
    df["successful_login"] = df["login_status"].isin(["success", "successful", "ok", "passed"])
    df["unknown_device"] = df["device_type"].isin(
        ["unknown_device", "unknown", "unrecognized", "unrecognised", ""]
    )

    grouped = df.groupby("vehicle_id", sort=False)
    df["new_ip"] = grouped["source_ip"].transform(
        lambda values: ~values.eq("") & ~values.duplicated()
    )
    df["new_device"] = grouped["device_type"].transform(
        lambda values: ~values.eq("") & ~values.duplicated()
    )
    df["prior_login_count"] = grouped.cumcount()

    recent_failures = np.zeros(len(df), dtype=int)
    for _, indexes in df.groupby("vehicle_id", sort=False).groups.items():
        positions = list(indexes)
        failure_times = []
        for pos in positions:
            current_time = df.at[pos, "timestamp"]
            if pd.notna(current_time):
                cutoff = current_time - pd.Timedelta(minutes=60)
                failure_times = [t for t in failure_times if t > cutoff]
                recent_failures[pos] = len(failure_times)
            if bool(df.at[pos, "failed_login"]) and pd.notna(current_time):
                failure_times.append(current_time)
    df["recent_failures_before_event"] = recent_failures

    model_features = pd.DataFrame({
        "hour": df["hour"],
        "night_login": df["night_login"].astype(int),
        "failed_login": df["failed_login"].astype(int),
        "successful_login": df["successful_login"].astype(int),
        "unknown_device": df["unknown_device"].astype(int),
        # Novelty features are kept for review, but are not fed to the model.
    })
    scores, model_flags = isolation_forest_scores(model_features, contamination=0.02)
    df["model_anomaly_score"] = scores
    df["model_anomaly"] = model_flags

    # Login alerts: only strong indicators trigger an alert. New IPs and night-time
    # logins are common in this synthetic data, so they are recorded as review signals
    # rather than treated as attacks by themselves.
    reasons = [[] for _ in range(len(df))]
    review_reasons = [[] for _ in range(len(df))]
    for i, row in df.iterrows():
        # Strong signal in this simulator: an unrecognized device. This is still an
        # indicator for investigation, not proof of a real account takeover.
        if bool(row["unknown_device"]):
            reasons[i].append("unknown_device")
            if bool(row["successful_login"]):
                reasons[i].append("successful_login_from_unknown_device")

        if bool(row["failed_login"]) and int(row["recent_failures_before_event"]) >= 2:
            reasons[i].append("repeated_failed_logins_within_60m")

        # Keep weaker signals for audit/review, but do not alert on them alone.
        if bool(row["new_ip"]):
            review_reasons[i].append("new_ip_observed")
        if bool(row["new_device"]):
            review_reasons[i].append("new_device_observed")
        if bool(row["night_login"]):
            review_reasons[i].append("night_time_login")
        if (
            int(row["prior_login_count"]) >= 3
            and bool(row["successful_login"])
            and bool(row["new_ip"])
            and bool(row["new_device"])
        ):
            review_reasons[i].append("successful_login_with_new_ip_and_device_review")
        if bool(row["night_login"]) and bool(row["new_ip"]) and bool(row["successful_login"]):
            review_reasons[i].append("night_login_with_new_ip_review")

    df["rule_alert"] = [bool(items) for items in reasons]

    # Model anomalies remain visible in the CSV, but a model score plus generic
    # novelty/time signals does not create a login alert. This avoids promoting
    # routine synthetic IP churn into incidents.
    df["model_only_alert"] = False
    df["alert"] = [bool(items) for items in reasons]
    df["alert_reason"] = [";".join(dict.fromkeys(items)) if items else "none" for items in reasons]
    df["review_reason"] = [";".join(dict.fromkeys(items)) if items else "none" for items in review_reasons]
    df["severity"] = "NONE"
    df.loc[df["alert"], "severity"] = "HIGH"
    df.loc[df["alert_reason"].str.contains("repeated_failed_logins", regex=False), "severity"] = "HIGH"

    evaluate_if_available(df, "is_attack", "alert", "LOGIN", report_lines)
    report_lines.append(f"Login rows processed: {len(df)}")
    report_lines.append(f"Login rule alerts: {int(df['rule_alert'].sum())}")
    report_lines.append(f"Login model anomalies: {int(df['model_anomaly'].sum())}")
    report_lines.append(f"Login model-only alerts: {int(df['model_only_alert'].sum())}")
    report_lines.append(f"Login review-only observations (not alerts): {int((df['review_reason'] != 'none').sum())}")
    df.sort_values("_original_order").drop(columns=["_original_order"]).to_csv(
        LOGIN_OUTPUT, index=False
    )


def main():
    report_lines = [
        "HYBRID VEHICLE SECURITY DETECTOR — REVISED VERSION",
        f"Working folder: {BASE_DIR}",
        "Rule checks + Isolation Forest anomaly scoring",
        "Synthetic/demo metrics are not proof of real-world performance.",
        "",
    ]

    if not TELEMETRY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TELEMETRY_FILE.name}. Put it in the same folder as this script."
        )
    if not LOGIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing {LOGIN_FILE.name}. Put it in the same folder as this script."
        )

    run_telemetry(report_lines)
    run_logins(report_lines)

    report_lines.extend([
        "",
        "OUTPUT FILES",
        f"Telemetry: {TELEMETRY_OUTPUT.name}",
        f"Logins: {LOGIN_OUTPUT.name}",
        f"Summary: {SUMMARY_OUTPUT.name}",
        "",
        "Review alert_reason for why each row was flagged.",
        "Login new-IP/night-time signals are recorded in review_reason, not alerted alone.",
        "Compare alerts with ground-truth labels only during evaluation.",
    ])
    SUMMARY_OUTPUT.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print("\nFinished. Revised output files were saved beside this script.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Check that telemetry.csv and logins.csv are in the same folder.")
        print("If needed in Pydroid 3, install: pandas numpy scikit-learn")
        raise
