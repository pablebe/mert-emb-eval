"""
Correlate metrics and ratings from listener responses.
"""

import os
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, kendalltau
from utils.helper_functions import (
    load_listener_responses,
    load_metrics,
    merge_metrics_and_ratings
)
#TODO: combine pcc and srcc table
# Dataset selection: set to 'bake_off' or 'gensvs'
DATASET = 'gensvs'

# Default gensvs models (flat per-model folders in audio/gensvs_eval_audio)
DEFAULT_MODELS_GENSVS = ['htdemucs', 'melroformer_bigvgan', 'melroformer_large', 'melroformer_small', 'sgmsvs']

DISC_MODELS_GENSVS = ['htdemucs', 'melroformer_large', 'melroformer_small']
GEN_MODELS_GENSVS = ['melroformer_bigvgan', 'sgmsvs']

# Preferred metric order for tables/exports (internal names)
DESIRED_METRIC_ORDER = [
    'MERT-v1-95M-MSE', 'FADMERT-v1-95M-MSE',
    'SDR-MSE', 'SI-SDR-MSE', 'SI-SAR-MSE', 'SI-SIR-MSE',
    'SPEC-MSE-MSE',
]

HEATMAP_METRIC_ORDER = DESIRED_METRIC_ORDER

METRIC_DISPLAY_NAMES = {
    'FADMERT-v1-95M-MSE': r'FAD$_{\text{Mert}}$',
    'MERT-v1-95M-MSE': r'MSE$_{\text{Mert}}$',
    'FADmusic2latent-MSE': r'FAD$_{\text{M2L}}$',
    'music2latent-MSE': r'MSE$_{\text{M2L}}$',
    'SI-SDR-MSE': 'SI-SDR',
    'SDR-MSE': 'SDR',
    'SI-SIR-MSE': 'SI-SIR',
    'SI-SAR-MSE': 'SI-SAR',
    'WAV-MSE-MSE': r'MSE$_{\text{wav}}$',
    'SPEC-MSE-MSE': r'MSE$_{\text{spec}}$'
}


def calculate_correlations_by_stem(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SRCC (Spearman rank correlation coefficient) and PCC (Pearson correlation coefficient)
    between ratings and each metric, grouped by stem type, plus overall correlations.
    
    Args:
        merged_df: Merged DataFrame with metrics and ratings.
    
    Returns:
        DataFrame with columns: stem_type, metric, srcc, pcc (includes 'overall' stem_type)
    """
    # Get metric columns (all numeric columns except rating, num_rating_sessions, num_tracks)
    metric_cols = [col for col in merged_df.columns 
                   if col not in ["track", "stem type", "model_name", "rating", 
                                 "num_rating_sessions", "num_tracks", "filepath"]]
    
    results = []
    
    # Calculate overall correlations (across all stem types) — skip for gensvs (vocals-only)
    if DATASET != 'gensvs':
        print(f"\nProcessing overall (all stem types combined)")
        for metric_col in metric_cols:
            # Remove NaN values for correlation calculation
            valid_idx = ~(merged_df[metric_col].isna() | merged_df["rating"].isna())
            valid_data = merged_df[valid_idx]
            
            if len(valid_data) < 2:
                print(f"  {metric_col}: Skipped (insufficient data)")
                continue
            
            # Calculate SRCC
            srcc, srcc_pval = spearmanr(valid_data["rating"], valid_data[metric_col])
            
            # Calculate PCC
            pcc, pcc_pval = pearsonr(valid_data["rating"], valid_data[metric_col])
            
            # Calculate Kendall's tau
            tau, tau_pval = kendalltau(valid_data["rating"], valid_data[metric_col])
            
            results.append({
                "stem_type": "overall",
                "metric": metric_col,
                "srcc": srcc,
                "srcc_pval": srcc_pval,
                "pcc": pcc,
                "pcc_pval": pcc_pval,
                "tau": tau,
                "tau_pval": tau_pval,
                "num_samples": len(valid_data)
            })
            
            print(f"  {metric_col}: SRCC={srcc:.4f}, PCC={pcc:.4f}, Tau={tau:.4f}, n={len(valid_data)}")
    
    # Group by stem type
    for stem_type, group in merged_df.groupby("stem type"):
        print(f"\nProcessing stem type: {stem_type}")
        
        # Calculate correlations for each metric
        for metric_col in metric_cols:
            # Remove NaN values for correlation calculation
            valid_idx = ~(group[metric_col].isna() | group["rating"].isna())
            valid_data = group[valid_idx]
            
            if len(valid_data) < 2:
                print(f"  {metric_col}: Skipped (insufficient data)")
                continue
            
            # Calculate SRCC
            srcc, srcc_pval = spearmanr(valid_data["rating"], valid_data[metric_col])
            
            # Calculate PCC
            pcc, pcc_pval = pearsonr(valid_data["rating"], valid_data[metric_col])
            
            # Calculate Kendall's tau
            tau, tau_pval = kendalltau(valid_data["rating"], valid_data[metric_col])
            
            results.append({
                "stem_type": stem_type,
                "metric": metric_col,
                "srcc": srcc,
                "srcc_pval": srcc_pval,
                "pcc": pcc,
                "pcc_pval": pcc_pval,
                "tau": tau,
                "tau_pval": tau_pval,
                "num_samples": len(valid_data)
            })
            
            print(f"  {metric_col}: SRCC={srcc:.4f}, PCC={pcc:.4f}, Tau={tau:.4f}, n={len(valid_data)}")
    
    corr_df = pd.DataFrame(results)
    return corr_df


def calculate_gensvs_split_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlations for GenSVS in three pooled groups:
    - overall (all stems, all models)
    - vocals_gen_models (vocals rows for generative models only)
    - vocals_disc_models (vocals rows for discriminative models only)
    """
    metric_cols = [col for col in merged_df.columns
                   if col not in ["track", "stem type", "model_name", "rating",
                                  "num_rating_sessions", "num_tracks", "filepath"]]

    subsets = [
        ("overall", merged_df),
        ("vocals_gen_models", merged_df[
            (merged_df["stem type"] == "vocals") &
            (merged_df["model_name"].isin(GEN_MODELS_GENSVS))
        ]),
        ("vocals_disc_models", merged_df[
            (merged_df["stem type"] == "vocals") &
            (merged_df["model_name"].isin(DISC_MODELS_GENSVS))
        ]),
    ]

    results = []
    for subset_name, subset_df in subsets:
        print(f"\nProcessing pooled subset: {subset_name} (n={len(subset_df)})")
        for metric_col in metric_cols:
            valid_idx = ~(subset_df[metric_col].isna() | subset_df["rating"].isna())
            valid_data = subset_df[valid_idx]
            if len(valid_data) < 2:
                continue

            srcc, srcc_pval = spearmanr(valid_data["rating"], valid_data[metric_col])
            pcc, pcc_pval = pearsonr(valid_data["rating"], valid_data[metric_col])
            tau, tau_pval = kendalltau(valid_data["rating"], valid_data[metric_col])

            results.append({
                "stem_type": subset_name,
                "metric": metric_col,
                "srcc": srcc,
                "srcc_pval": srcc_pval,
                "pcc": pcc,
                "pcc_pval": pcc_pval,
                "tau": tau,
                "tau_pval": tau_pval,
                "num_samples": len(valid_data),
            })
            print(f"  {metric_col}: SRCC={srcc:.4f}, PCC={pcc:.4f}, n={len(valid_data)}")

    return pd.DataFrame(results)


def export_gensvs_split_heatmap_table(
    correlations_df: pd.DataFrame,
    output_path: str = None,
    filter_metrics: list = None,
    metrics_display_names: dict = None,
) -> str:
    """
    Export a single heatmap-style table with SRCC/PCC pairs for:
    Overall, Vocals (gen. models), Vocals (disc. models).
    """
    if metrics_display_names is None:
        metrics_display_names = {}

    corr_abs = correlations_df.copy()
    corr_abs["srcc"] = corr_abs["srcc"].abs()
    corr_abs["pcc"] = corr_abs["pcc"].abs()

    if filter_metrics is not None:
        corr_abs = corr_abs[corr_abs["metric"].isin(filter_metrics)]

    metrics_present = [m for m in HEATMAP_METRIC_ORDER if m in corr_abs["metric"].unique()]
    other_metrics = [m for m in corr_abs["metric"].unique() if m not in metrics_present]
    metrics = metrics_present + other_metrics

    row_order = ["vocals_gen_models", "vocals_disc_models", "overall"]
    row_labels = {
        "overall": r"Overall",
        "vocals_gen_models": r"Vocals (gen. models)",
        "vocals_disc_models": r"Vocals (disc. models)",
    }

    latex_lines = []
    latex_lines.append(r"\begin{table*}[ht]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
    latex_lines.append(
        r"\caption{Overall SRCC and PCC between metrics and ratings for the GenSVS dataset. "
        r"Correlations are pooled across all models (Overall), generative models only "
        r"(vocals), and discriminative models only (vocals).}"
    )
    latex_lines.append(r"\label{tab:correlations_gensvs_gen_disc_heatmap}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{l" + "c" * (2 * len(metrics)) + "}")
    latex_lines.append(r"\toprule")

    header_top = "& " + " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{{metrics_display_names.get(metric, metric)}}}" for metric in metrics]
    ) + r" \\"
    latex_lines.append(header_top)

    cmidrules = []
    for i in range(len(metrics)):
        left = 2 + 2 * i
        right = left + 1
        cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")
    latex_lines.append(" ".join(cmidrules))

    latex_lines.append(
        "Stem & " + " & ".join(["SRCC & PCC"] * len(metrics)) + r" \\"
    )
    latex_lines.append(r"\midrule")

    for row_key in row_order:
        row_df = corr_abs[corr_abs["stem_type"] == row_key]
        row_cells = [row_labels[row_key]]
        for metric in metrics:
            metric_row = row_df[row_df["metric"] == metric]
            if len(metric_row) == 0:
                row_cells.extend(["--", "--"])
            else:
                srcc_val = float(metric_row.iloc[0]["srcc"])
                pcc_val = float(metric_row.iloc[0]["pcc"])
                row_cells.extend([
                    rf"\corr{{{srcc_val:.2f}}}",
                    rf"\corr{{{pcc_val:.2f}}}",
                ])
        latex_lines.append(" & ".join(row_cells) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")

    latex_table = "\n".join(latex_lines)
    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            f.write(latex_table)
        print(f"Saved GenSVS split heatmap table to {output_path}")

    return latex_table


def print_stem_type_statistics(df: pd.DataFrame) -> None:
    """
    Print statistics about the distribution of rows by stem type.
    
    Args:
        df: DataFrame containing listener responses.
    """
    stem_counts = df["stem type"].value_counts()
    total_rows = len(df)
    
    print("\n" + "="*50)
    print("Stem Type Distribution")
    print("="*50)
    for stem_type, count in stem_counts.items():
        percentage = count / total_rows * 100
        print(f"{stem_type:10s}: {count:5d} rows ({percentage:5.2f}%)")
    print("-"*50)
    print(f"{'Total':10s}: {total_rows:5d} rows (100.00%)")
    print("="*50)


def find_ot_result_files(pattern: str = "subsample_False") -> list:
    """
    Find all combined OT result CSV files matching a pattern.
    Only returns "combined_" files, which will then load from individual model folders.
    
    Args:
        pattern: String pattern to match in filenames.
    
    Returns:
        List of Path objects for matching files (relative to script location, including ot_results/).
    """
    ot_results_dir = Path(__file__).parent / "ot_results"
    
    if not ot_results_dir.exists():
        print(f"Warning: OT results directory not found: {ot_results_dir}")
        return []
    
    # Find all combined CSV files containing the pattern
    # We only want "combined_" files at the method level (emd2, sinkhorn)
    # not the individual model files in subdirectories
    matching_files = []
    for csv_file in ot_results_dir.rglob("*.csv"):
        if pattern in csv_file.name and "combined" in csv_file.name:
            # Get relative path from script location (includes "ot_results/")
            relative_path = csv_file.relative_to(Path(__file__).parent)
            matching_files.append(str(relative_path))
    
    return sorted(matching_files)

# Use `load_ot_results_from_all_models` from `utils.helper_functions` (imported above)


def correlate_ratings_and_metrics(
    csv_filename: str,
    ratings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """
    Process correlations for a single OT results file.
    
    Args:
        ot_csv_filename: Filename/path of OT results CSV.
        ratings_df: Already loaded and processed ratings DataFrame.
        metrics_df: Already loaded and deduplicated metrics DataFrame.
    """
    
    merged_df = merge_metrics_and_ratings(metrics_df, ratings_df, None)
    
    # Print stem type statistics
    print_stem_type_statistics(merged_df)
    
    # Calculate correlations by stem type
    print("\n" + "="*80)
    print("CALCULATING CORRELATIONS BY STEM TYPE")
    print("="*80)
    correlations_df = calculate_correlations_by_stem(merged_df)
    
    print("\n" + "="*80)
    print("CORRELATION RESULTS:")
    print("="*80)
    print(correlations_df.to_string(index=False))
    

    # Create output directory for this OT method (dataset-specific)
    output_dir = Path(__file__).parent / "correlation_across_songs_results_overall" / DATASET
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged dataframe to CSV
    merged_output_path = output_dir / f"merged_metrics_and_ratings.csv"
    merged_df.to_csv(merged_output_path, index=False)
    print(f"\nSaved merged data to {merged_output_path}")
    
    # Reorder correlations by stem_type and desired metric priority for consistent tables
    desired_metric_order = DESIRED_METRIC_ORDER
    order_present = [m for m in desired_metric_order if m in correlations_df['metric'].unique()]
    others = [m for m in correlations_df['metric'].unique() if m not in order_present]
    import pandas as _pd
    metric_cat = _pd.CategoricalDtype(categories=order_present + others, ordered=True)
    correlations_df['metric'] = correlations_df['metric'].astype(metric_cat)
    correlations_df = correlations_df.sort_values(['stem_type', 'metric'])

    # Save correlations to CSV
    corr_output_path = output_dir / f"correlations_by_stem.csv"
    correlations_df.to_csv(corr_output_path, index=False)
    print(f"Saved correlations to {corr_output_path}")

    # Export baseline metrics only (use preferred order defined above);
    baseline_metrics = DESIRED_METRIC_ORDER.copy()
    # For gensvs exclude SI-SIR and SI-SAR (single-stem vocals only)
    if DATASET == 'gensvs':
        baseline_metrics = [m for m in baseline_metrics if m not in ('SI-SIR-MSE', 'SI-SAR-MSE')]

    latex_baseline_output_path = output_dir / f"correlations_table_baseline.tex"
    # Provide a custom PCC caption for overall correlations when Bake-Off dataset is selected
    caption_by_type = None
    if DATASET == 'bake_off':
        caption_by_type = {
            'PCC': "Overall PCC between metrics and ratings for Bake-Off dataset. Metrics and ratings were pooled across models before calculating the correlation coefficients."
        }
    latex_table_baseline = export_correlations_to_latex(
        correlations_df,
        latex_baseline_output_path,
        filter_metrics=baseline_metrics,
        metrics_display_names=METRIC_DISPLAY_NAMES,
        caption_by_type=caption_by_type,
    )

    if DATASET == 'gensvs':
        split_correlations_df = calculate_gensvs_split_correlations(merged_df)
        split_corr_output_path = output_dir / "correlations_gensvs_gen_disc_split.csv"
        split_correlations_df.to_csv(split_corr_output_path, index=False)
        print(f"Saved gensvs split correlations to {split_corr_output_path}")

        split_metrics = [m for m in HEATMAP_METRIC_ORDER if m in split_correlations_df['metric'].unique()]
        split_latex_output_path = output_dir / "correlations_table_gensvs_gen_disc_heatmap.tex"
        export_gensvs_split_heatmap_table(
            split_correlations_df,
            output_path=split_latex_output_path,
            filter_metrics=split_metrics,
            metrics_display_names=METRIC_DISPLAY_NAMES,
        )
    print("\n" + "="*80)
    print(f"COMPLETED: {csv_filename}")
    print("="*80)


def export_correlations_to_latex(correlations_df: pd.DataFrame, output_path: str = None, filter_metrics: list = None, metrics_display_names: dict = None, caption_by_type: dict = None) -> str:
    """
    Export correlation results to three separate LaTeX tables (SRCC, PCC, and Tau) with absolute values 
    and bold highest values per row.
    
    Args:
        correlations_df: DataFrame with correlation results (stem_type, metric, srcc, pcc, tau columns).
        output_path: Path to save the LaTeX tables. If None, returns the LaTeX string only.
        filter_metrics: List of metrics to include. If None, includes all metrics.
        metrics_display_names: Optional dictionary mapping metric names to display names.
        caption_by_type: Optional dict mapping correlation types ('SRCC','PCC','Tau') to custom caption strings.
    Returns:
        LaTeX tables string.
    """
    # Take absolute values of correlations
    corr_abs = correlations_df.copy()
    corr_abs['abs_srcc'] = corr_abs['srcc'].abs()
    corr_abs['abs_pcc'] = corr_abs['pcc'].abs()
    corr_abs['abs_tau'] = corr_abs['tau'].abs()
    
    # Filter metrics if specified
    if filter_metrics is not None:
        corr_abs = corr_abs[corr_abs['metric'].isin(filter_metrics)]
    
    # Pivot the data for better table layout: stem types as rows, metrics as columns
    stem_types = list(corr_abs['stem_type'].unique())
    # Sort stem types with 'overall' first, then alphabetically, but only include 'overall' if present
    if 'overall' in stem_types:
        stem_types_sorted = ['overall'] + sorted([s for s in stem_types if s != 'overall'])
    else:
        stem_types_sorted = sorted(stem_types)
    # Order metrics according to preferred priority, falling back to any additional metrics
    desired_metric_order = DESIRED_METRIC_ORDER
    metrics_present = [m for m in desired_metric_order if m in corr_abs['metric'].unique()]
    other_metrics = [m for m in corr_abs['metric'].unique() if m not in metrics_present]
    metrics = metrics_present + list(other_metrics)
    

    
    latex_tables = []
    
    # Create three tables: one for SRCC, one for PCC, one for Tau
    for corr_type in ['SRCC', 'PCC', 'Tau']:
        if corr_type == 'SRCC':
            corr_col = 'abs_srcc'
        elif corr_type == 'PCC':
            corr_col = 'abs_pcc'
        else:
            corr_col = 'abs_tau'
        
        # Start LaTeX table
        latex_lines = []
        latex_lines.append(r"\begin{table}[ht]")
        latex_lines.append(r"\centering")
        # Allow caller to override caption for a specific correlation type
        if caption_by_type is not None and corr_type in caption_by_type:
            caption = caption_by_type[corr_type]
        else:
            # Use dataset-aware captions; for overall we use 'Overall <type>' phrasing
            if corr_type == 'SRCC':
                label = 'SRCC'
            elif corr_type == 'PCC':
                label = 'PCC'
            else:
                label = corr_type
            caption = f"Overall {label} between metrics and ratings"
            if 'DATASET' in globals() and DATASET == 'bake_off':
                caption += " for Bake-Off dataset."
            elif 'DATASET' in globals() and DATASET == 'gensvs':
                caption += " for GenSVS dataset."
            caption += " Metrics and ratings were pooled across models before calculating the correlation coefficients."
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{tab:correlations_{corr_type.lower()}}}")
        
        # Create column specification: l for stem type, c for each metric
        col_spec = "l" + "c" * len(metrics)
        latex_lines.append(r"\resizebox{\columnwidth}{!}{%")
        latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
        latex_lines.append(r"\hline")
        
        # Create header row
        header = r"Stem Type"
        for metric in metrics:
            display_name = metrics_display_names.get(metric, metric) if metrics_display_names is not None else metric
            header += f" & {display_name}"
        header += r" \\"
        latex_lines.append(header)
        latex_lines.append(r"\hline")
        
        # For each stem type, create a row
        for stem_type in stem_types_sorted:
            stem_data = corr_abs[corr_abs['stem_type'] == stem_type]
            
            # Find the maximum absolute correlation for this stem type (rounded to 2 decimals)
            if len(stem_data) > 0:
                max_corr_rounded = round(stem_data[corr_col].max(), 2)
            else:
                max_corr_rounded = 0
            
            # Format stem type name
            if stem_type == 'overall':
                row_str = "\\textbf{Overall}"
            else:
                row_str = stem_type.capitalize()
            
            for metric in metrics:
                metric_data = stem_data[stem_data['metric'] == metric]
                
                if len(metric_data) > 0:
                    corr_val = metric_data.iloc[0][corr_col]
                    
                    # Format with 2 decimal places
                    corr_val_rounded = round(corr_val, 2)
                    corr_str = f"{corr_val_rounded:.2f}"
                    
                    # Make bold if rounded value equals max rounded value
                    if abs(corr_val_rounded - max_corr_rounded) < 1e-9:
                        corr_str = r"\textbf{" + corr_str + "}"
                    
                    row_str += f" & {corr_str}"
                else:
                    row_str += " & --"
            
            row_str += r" \\"
            latex_lines.append(row_str)
            
            # Add a horizontal line after the overall row
            if stem_type == 'overall':
                latex_lines.append(r"\hline")
        
        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"}")
        latex_lines.append(r"\end{table}")
        
        latex_tables.append("\n".join(latex_lines))
    
    # Combine both tables with spacing
    latex_table = "\n\n".join(latex_tables)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved LaTeX tables to {output_path}")
    
    return latex_table


def main():
    """Main execution function."""

    violation_threshold = 2 # Set violation threshold for filtering listener responses

    # Load and process data (common for all OT files)
    print("\n" + "="*80)
    print("LOADING COMMON DATA")
    print("="*80)
    
    # Load data and filter according to violation threshold
    if DATASET == 'gensvs':
        # Attempt to load gensvs-specific ratings if available (user must provide a CSV in third_party/gensvs/)
        gensvs_ratings_path = Path(__file__).parent / "third_party" / "gensvs" / "gensvs_eval_data.csv"
        print(f"Using GENSVS listener responses: {gensvs_ratings_path}")
        ratings_df = load_listener_responses(violation_threshold=violation_threshold, csv_path=gensvs_ratings_path)
 
        model_cols = DEFAULT_MODELS_GENSVS
    else:
        ratings_df = load_listener_responses(violation_threshold=violation_threshold)
        model_cols = ["IRM1", "Open-UMix", "SCNet-large", "htdemucs_ft", "REP1"]
    
    metrics_csv_path = Path(__file__).parent / Path("emb_mse_results_"+DATASET) / "emb_mse_results.csv"

    # For GENSVS the ratings file format is different (per-file rows with 'file_id' and 'model_name').
    # Convert to same aggregated format expected by downstream code: rows per (track, stem type) with model columns.
    if DATASET == 'gensvs':
        # ratings_df here is the raw gensvs CSV
        # derive 'track' from 'file_id' (it already looks like 'fileid_48')
        ratings_pivot = (
            ratings_df.groupby(['file_id', 'model_name'])['DMOS'].median()
            .unstack('model_name')
            .reset_index()
        )
        ratings_pivot = ratings_pivot.rename(columns={'file_id': 'track'})
        # add stem type (vocals)
        ratings_pivot['stem type'] = 'vocals'
        # Ensure all expected model columns exist
        for m in model_cols:
            if m not in ratings_pivot.columns:
                ratings_pivot[m] = pd.NA
        # Reorder columns so that track, stem type come first
        cols = ['track', 'stem type'] + [m for m in model_cols]
        ratings_pivot = ratings_pivot[cols]
        ratings_avg = ratings_pivot
        # Convert wide to long format as expected by downstream code
        ratings_avg = ratings_avg.melt(
            id_vars=[col for col in ratings_avg.columns if col not in model_cols],
            value_vars=model_cols,
            var_name="model_name",
            value_name="rating"
        )
        ratings_avg = ratings_avg.dropna(subset=["rating"])

    else:
        ratings_avg = ratings_df.groupby(["track", "stem type"], as_index=False)[model_cols].median()

        ratings_avg = ratings_avg.melt(
            id_vars=[col for col in ratings_avg.columns if col not in model_cols],
            value_vars=model_cols,
            var_name="model_name",
            value_name="rating"
        )
        ratings_avg = ratings_avg.dropna(subset=["rating"])

    # Load metrics
    metrics_df = load_metrics(metrics_csv_path)

    # Dataset-specific metrics/rating alignment for GENSVS
    if DATASET == 'gensvs':
        # Derive 'track' column from filepath (fileid_N)
        metrics_df['track'] = metrics_df['filepath'].str.extract(r'(fileid_\d+)')
        # instrument_name should already be 'vocals'
        print(f"Transformed metrics DataFrame for gensvs: using 'track' extracted from filename")
    
    # Average metrics across multiple files per track/stem/model combination
    print(f"\n=== METRICS DEDUPLICATION ===")
    print(f"Metrics before dedup: {metrics_df.shape}")
    print(f"Filepaths per combination: {metrics_df.groupby(['track', 'instrument_name', 'model_name'])['filepath'].count().value_counts()}")
    
    # Get numeric columns to average (exclude identifier columns)
    numeric_cols = [col for col in metrics_df.columns 
                   if col not in ['track', 'instrument_name', 'model_name', 'filepath']]
    
    # Create aggregation dictionary
    agg_dict = {col: 'mean' for col in numeric_cols}
    
    # Average metrics across multiple files per track/stem/model
    metrics_df = metrics_df.groupby(['track', 'instrument_name', 'model_name']).agg(agg_dict).reset_index()
    
    print(f"Metrics after averaging: {metrics_df.shape}")
    
    # Reshape and average ratings (common for all OT files)
    print("\n" + "="*80)
    print("PROCESSING RATINGS")
    print("="*80)

    print(f"\n=== RATINGS_AVG DEBUG ===")
    print(f"Total rows: {len(ratings_avg)}")
    print(f"Unique tracks: {ratings_avg['track'].nunique()}")
    print(f"Unique models: {ratings_avg['model_name'].nunique()}")
    print(f"Unique stems: {ratings_avg['stem type'].nunique()}")
    print(f"\nBreakdown by model:")
    print(ratings_avg.groupby('model_name')['stem type'].value_counts().unstack(fill_value=0))
    print(f"\nExpected: {ratings_avg['track'].nunique()} tracks × combinations")
    
    print("\n" + "="*80)
    print("RATINGS BEFORE MERGE (first 10 rows):")
    print("="*80)
    print(ratings_avg.head(10))
    print()
    
    # Process baseline metrics and ratings
    correlate_ratings_and_metrics(
        csv_filename="baseline_metrics.csv",
        ratings_df=ratings_avg,
        metrics_df=metrics_df,
    )

    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)
    print(f"Results saved to correlation_across_songs_results_overall/")

if __name__ == "__main__":
    main()
