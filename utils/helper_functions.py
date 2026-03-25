"""
Helper functions for loading and processing data for correlation analysis.
"""

import pandas as pd
from pathlib import Path


def load_listener_responses(violation_threshold: int = 3, csv_path: str = None) -> pd.DataFrame:
    """
    Load listener responses from CSV and filter by violation threshold.
    
    Args:
        violation_threshold: Maximum allowed total_violations per row. 
                           If None, uses the global VIOLATION_THRESHOLD.
                           Rows with total_violations >= this value are dropped.
        csv_path: Optional path to a listener responses CSV. If None, uses bake_off default.
    
    Returns:
        DataFrame containing filtered listener responses.
    """
    
    # Determine CSV path
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "third_party" / "bake_off" / "raw_listener_responses_w_violations.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Warning: Listener responses file not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    
    # Only filter if 'violation_total' column exists
    if "violation_total" in df.columns:
        initial_rows = len(df)
        df = df[df["violation_total"] <= violation_threshold]
        filtered_rows = len(df)
        dropped_rows = initial_rows - filtered_rows

        print(f"Loaded listener responses from {csv_path}")
        print(f"Initial rows: {initial_rows}")
        print(f"Dropped rows (violation_total <= {violation_threshold}): {dropped_rows}")
        print(f"Remaining rows: {filtered_rows}")
        if initial_rows:
            print(f"Dropped {dropped_rows/initial_rows*100:.2f}% of data")
    else:
        print(f"'violation_total' column not found in {csv_path}, returning unfiltered DataFrame.")
    
    return df


def load_metrics(metrics_csv_path: str = None) -> pd.DataFrame:
    """
    Load embedding MSE metrics from CSV.
    
    Args:
        metrics_csv_path: Path to the metrics CSV file. If None, uses default path.
    
    Returns:
        DataFrame containing metrics data.
    """
    if metrics_csv_path is None:
        metrics_csv_path = Path(__file__).parent.parent / "emb_mse_results" / "emb_mse_results.csv"
    else:
        metrics_csv_path = Path(metrics_csv_path)
    
    df = pd.read_csv(metrics_csv_path)
    print(f"Loaded metrics from {metrics_csv_path}")
    print(f"Metrics shape: {df.shape}")
    
    return df


def load_ot_results(ot_csv_filename: str = None) -> pd.DataFrame:
    """
    Load optimal transport results from CSV.
    
    Args:
        ot_csv_filename: Filename of the OT results CSV file in ot_results folder.
                        If None, uses 'combined_emd_MERT-v1-95M_l2.csv' as default.
                        Can be a full path or just a filename (will look in ot_results/).
    
    Returns:
        DataFrame containing OT results data.
    """

    # If just a filename (no path separator), look in ot_results folder
    ot_csv_path = Path(ot_csv_filename) 
    if not ot_csv_path.is_absolute() and '/' not in str(ot_csv_filename):
        ot_csv_path = Path(__file__).parent.parent / "ot_results" / ot_csv_filename
    
    if not ot_csv_path.exists():
        print(f"Warning: OT results file not found: {ot_csv_path}")
        return None
    
    df = pd.read_csv(ot_csv_path)
    print(f"Loaded OT results from {ot_csv_path}")
    print(f"OT results shape: {df.shape}")
    
    # Rename ot_cost to OT-Cost-MSE for consistency with other metrics
    df = df.rename(columns={"ot_cost": "OT-Cost-MSE"})
    
    # Rename other matmetrics columns to have -MSE suffix for consistency
    matmetrics_cols = {
        "avg_transport_distance": "Avg-Transport-Distance-MSE",
        "temporal_alignment_error": "Temporal-Alignment-Error-MSE",
        "transport_entropy": "Transport-Entropy-MSE",
        "transport_concentration_top10": "Transport-Concentration-Top10-MSE",
        "diagonal_mass_ratio": "Diagonal-Mass-Ratio-MSE",
        "transport_spread_std": "Transport-Spread-Std-MSE",
        "max_transport_distance": "Max-Transport-Distance-MSE",
        "effective_support_size": "Effective-Support-Size-MSE"
    }
    df = df.rename(columns=matmetrics_cols)
    
    return df


def load_ot_results_from_all_models(ot_csv_filename: str) -> pd.DataFrame:
    """
    Load OT results from individual model folders and combine them.
    
    The combined CSV files only contain htdemucs_ft data, so we need to load
    from individual model folders (IRM1, Open-UMix, REP1, SCNet-large, htdemucs_ft).
    
    Combined file patterns (method is in folder path, not filename):
    - All methods: combined_MERT-v1-95M_subsample_False_metric_XXX_norm_l2.csv
    
    Individual file patterns (method is in folder path, not filename):
    - All methods: MERT-v1-95M_subsample_False_metric_XXX_norm_l2_results.csv
    
    Args:
        ot_csv_filename: Path to the combined OT results CSV file.
        
    Returns:
        Combined DataFrame with OT results from all models.
    """
    ot_path = Path(ot_csv_filename)
    
    # Extract the method folder (emd2, sinkhorn, etc.) and filename pattern
    if ot_path.parts[0] == 'ot_results' and len(ot_path.parts) > 1:
        method_folder = ot_path.parts[1]
        filename = ot_path.name
        
        # Transform combined filename to individual filename pattern
        # Remove "combined_" prefix and add "_results" before .csv
        if filename.startswith("combined_"):
            # Remove "combined_" and replace .csv with _results.csv
            individual_filename = filename.replace("combined_", "").replace(".csv", "_results.csv")
        else:
            individual_filename = filename.replace(".csv", "_results.csv")
    else:
        print(f"Warning: Could not parse OT path structure: {ot_csv_filename}")
        return load_ot_results(ot_csv_filename=str(ot_csv_filename))
    
    # Define model folders
    model_folders = ["IRM1", "Open-UMix", "REP1", "SCNet-large", "htdemucs_ft"]
    ot_results_dir = Path(__file__).parent.parent / "ot_results" / method_folder
    
    combined_dfs = []
    
    for model_name in model_folders:
        model_dir = ot_results_dir / model_name
        model_file = model_dir / individual_filename
        
        if model_file.exists():
            print(f"  Loading {model_name}: {model_file.name}")
            df = pd.read_csv(model_file)
            combined_dfs.append(df)
        else:
            print(f"  Warning: File not found for {model_name}: {model_file.name}")
    
    if not combined_dfs:
        print(f"  No individual model files found, trying combined file...")
        return load_ot_results(ot_csv_filename=str(ot_csv_filename))
    
    # Concatenate all model results
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    print(f"  Combined {len(combined_dfs)} model files: {combined_df.shape}")
    
    # Rename ot_cost to OT-Cost-MSE for consistency
    combined_df = combined_df.rename(columns={"ot_cost": "OT-Cost-MSE"})
    
    # Rename other matmetrics columns to have -MSE suffix for consistency
    matmetrics_cols = {
        "avg_transport_distance": "Avg-Transport-Distance-MSE",
        "temporal_alignment_error": "Temporal-Alignment-Error-MSE",
        "transport_entropy": "Transport-Entropy-MSE",
        "transport_concentration_top10": "Transport-Concentration-Top10-MSE",
        "diagonal_mass_ratio": "Diagonal-Mass-Ratio-MSE",
        "transport_spread_std": "Transport-Spread-Std-MSE",
        "max_transport_distance": "Max-Transport-Distance-MSE",
        "effective_support_size": "Effective-Support-Size-MSE"
    }
    combined_df = combined_df.rename(columns=matmetrics_cols)
    
    return combined_df


def merge_metrics_and_ratings(
    metrics_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    ot_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Merge metrics and averaged ratings by track, stem type, and model name.
    Optionally merge OT results as well.
    
    Args:
        metrics_df: DataFrame containing metrics (with 'track', 'instrument_name', 'model_name' columns).
        ratings_df: DataFrame containing averaged ratings (with 'track', 'stem type', 'model_name' columns).
        ot_df: Optional DataFrame containing OT results (with 'track', 'instrument_name', 'model_name' columns).
    
    Returns:
        Merged DataFrame with metrics and averaged ratings (one row per [track, stem type, model_name] pair).
    """
    # Rename column for consistency
    metrics_df = metrics_df.copy()
    ratings_df = ratings_df.copy()
    
    metrics_df = metrics_df.rename(columns={"instrument_name": "stem type"})
    
    # Merge on track, stem type, and model name
    merged = pd.merge(
        ratings_df,
        metrics_df,
        on=["track", "stem type", "model_name"],
        how="inner"
    )
    
    print(f"\nMerged metrics and averaged ratings on [track, stem type, model_name]")
    print(f"Merged shape: {merged.shape}")
    
    # Merge OT results if provided
    if ot_df is not None:
        ot_df = ot_df.copy()
        ot_df = ot_df.rename(columns={"instrument_name": "stem type"})
        
        # Select metric columns from OT results (OT-Cost-MSE and any matmetrics)
        # Keep all columns that end with -MSE
        id_cols = ["track", "stem type", "model_name"]
        metric_cols = [col for col in ot_df.columns if col.endswith("-MSE")]
        ot_cols = id_cols + metric_cols
        
        # Only select columns that exist in the dataframe
        ot_cols = [col for col in ot_cols if col in ot_df.columns]
        ot_df = ot_df[ot_cols]
        
        merged = pd.merge(
            merged,
            ot_df,
            on=["track", "stem type", "model_name"],
            how="left"  # Use left join to keep all existing data even if OT results are missing
        )
        
        print(f"Merged OT results on [track, stem type, model_name]")
        print(f"Final merged shape: {merged.shape}")
    
    return merged
