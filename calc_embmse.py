import os
from gensvs import EmbeddingMSE, get_all_models, cache_embedding_files
from pathlib import Path
import tqdm
import glob
import shutil
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from third_party.fadtk_mod.fad_mod import FrechetAudioDistance
from third_party.nussl.evaluation.bss_eval import _scale_bss_eval
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio


# Dataset selection: set to 'bake_off' or 'gensvs'
DATASET = 'gensvs'

# Default separation model lists (in-script variables)
DEFAULT_MODELS = ['htdemucs_ft', 'IRM1', 'Open-UMix', 'SCNet-large']
DEFAULT_MODELS_GENSVS = ['htdemucs', 'melroformer_bigvgan', 'melroformer_large', 'melroformer_small', 'sgmsvs']

# Stems to evaluate (in-script variable)
STEMS = ['bass', 'drums', 'vocals', 'other']

# Embedding model to use for embedding-MSE / FAD

EMBEDDING = 'MERT-v1-95M'#'music2latent'#'MERT-v1-95M'
#embedding calculation builds on multiprocessing library => don't forget to wrap your code in a main function
WORKERS = 2

# path to 
SEP_PATH = './audio/MSS_bake_off_eval_audio/10s_audio'
TGT_PATH = './audio/musdb18_test_10s'
if DATASET == 'gensvs':
    OUT_DIR = './emb_mse_results_gensvs'
else:
    OUT_DIR = './emb_mse_results_bake_off'
# Calculate embedding MSE
# Set to True to delete 'convert' and 'embeddings' folders before processing
DELETE_CACHE = False
# Set to True to skip BSS-eval computation if output files already exist
SKIP_BSS_EVAL = False
# Set to True to skip WAV-MSE and SPEC-MSE computation if output files already exist
SKIP_WAV_SPEC_MSE = True
# Set to True to shuffle embeddings before FAD calculation
SHUFFLE_EMBD = False

def compute_multisource_bss_eval(sep_folder: str, target_folder: str, stems: list = ['bass', 'drums', 'vocals', 'other']):
    """
    Compute BSS-eval metrics using multi-source approach where all sources are evaluated together.
    This provides meaningful SI-SIR values that measure inter-source interference.
    
    Uses _scale_bss_eval with proper multi-source input format:
    - references: (n_samples, n_sources) containing ALL reference sources
    - estimate: (n_samples, 1) containing estimate for ONE source
    - idx: which reference source to evaluate against
    
    Args:
        sep_folder: Path to folder containing separated audio files
        target_folder: Path to folder containing target/reference audio files
        stems: List of stem names to evaluate (default: ['bass', 'drums', 'vocals', 'other'])
    
    Returns:
        Dictionary mapping filepath to metrics dict {'SI-SDR': value, 'SI-SIR': value, 'SI-SAR': value}
    """
    results = {}
    
    # Get list of all separated files
    sep_files = glob.glob(os.path.join(sep_folder, '*.wav'))
    
    # Group separated files by stem
    sep_stem_files = {}
    for sep_file in sep_files:
        basename = os.path.basename(sep_file)
        for stem in stems:
            if basename == f'{stem}.wav' or basename.endswith(f'_{stem}.wav'):
                sep_stem_files[stem] = sep_file
                break
    
    if len(sep_stem_files) == 0:
        print(f"Warning: No valid stems found in {sep_folder}")
        return results
    
    # Load ALL reference stems (needed for computing interference)
    ref_sources = []
    ref_stem_order = []
    
    for stem in stems:
        # Try to find reference file
        ref_candidates = [
            os.path.join(target_folder, f'{stem}.wav'),
            os.path.join(target_folder, f'*_{stem}.wav')
        ]
        
        ref_file = None
        for candidate in ref_candidates:
            matches = glob.glob(candidate)
            if matches:
                ref_file = matches[0]
                break
        
        if ref_file and os.path.exists(ref_file):
            ref_audio, fs = sf.read(ref_file)
            if len(ref_audio.shape) > 1:
                ref_audio = np.mean(ref_audio, axis=1)
            ref_audio = ref_audio - np.mean(ref_audio)
            ref_sources.append(ref_audio)
            ref_stem_order.append(stem)
    
    if len(ref_sources) < 2:
        print(f"Info: Only {len(ref_sources)} reference stem(s) found, need 2+ for multi-source eval")
        return results
    
    # Make all reference sources the same length
    min_ref_len = min(len(r) for r in ref_sources)
    ref_sources = [r[:min_ref_len] for r in ref_sources]
    ref_sources = np.stack(ref_sources, axis=1)  # (n_samples, n_sources)
    
    # Compute metrics for each separated stem
    for stem, sep_file in sep_stem_files.items():
        # Find the index of this stem in the reference sources
        if stem not in ref_stem_order:
            print(f"Warning: No reference for {stem}, skipping")
            continue
        
        idx = ref_stem_order.index(stem)
        
        # Load separated audio
        sep_audio, fs = sf.read(sep_file)
        if len(sep_audio.shape) > 1:
            sep_audio = np.mean(sep_audio, axis=1)
        sep_audio = sep_audio - np.mean(sep_audio)
        
        # Match length with references
        sep_audio = sep_audio[:min_ref_len]
        estimate = sep_audio[:, None]  # (n_samples, 1)
        
        # Call _scale_bss_eval with ALL references and ONE estimate
        si_sdr, si_sir, si_sar, _, _, _ = _scale_bss_eval(
            ref_sources,  # All reference sources (n_samples, n_sources)
            estimate,     # Estimate for source idx (n_samples, 1)
            idx=idx,      # Which reference to evaluate against
            compute_sir_sar=True
        )
        
        key = sep_file.split('./')[-1]
        results[key] = {
            'SI-SDR': si_sdr,
            'SI-SIR': si_sir,
            'SI-SAR': si_sar
        }
    
    return results

def cleanup_cache_folders(root_path: str, delete_cache: bool = False):
    """
    Find and optionally delete 'convert' and 'embeddings' folders recursively.
    
    Args:
        root_path: Root directory to search
        delete_cache: If True, delete found folders; if False, just report them
    """
    root = Path(root_path)
    folders_to_delete = ['convert', 'embeddings']
    found_folders = []
    
    for folder_name in folders_to_delete:
        for folder in root.rglob(folder_name):
            if folder.is_dir():
                found_folders.append(folder)
                if delete_cache:
                    print(f"Deleting: {folder}")
                    shutil.rmtree(folder)
                else:
                    print(f"Found (not deleted): {folder}")
    
    if delete_cache:
        print(f"Deleted {len(found_folders)} cache folder(s)")
    else:
        print(f"Found {len(found_folders)} cache folder(s)")
    
    return found_folders

def combine_emb_mse_results(results_dir: str, output_file: str = 'summarized_emb_mse_results.csv'):
    """
    Combine all CSV files from emb_mse_results directory into a summarized CSV.
    
    Args:
        results_dir: Directory containing model folders with CSV files
        output_file: Output CSV filename
    """
    import re
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}. Skipping combine.")
        return None
    
    # Read model names from subdirectories in results_dir
    svs_model_names = [d.name for d in results_path.iterdir() if d.is_dir()]
    svs_model_names.sort()
    
    if DATASET != 'gensvs':
        metric_names = ['FADmusic2latent','FADMERT-v1-95M', 'MERT-v1-95M', 'music2latent', 'SI-SDR', 'SDR', 'SI-SIR', 'SI-SAR', 'WAV-MSE', 'SPEC-MSE']
    else:
        metric_names = ['FADmusic2latent','FADMERT-v1-95M', 'MERT-v1-95M', 'music2latent', 'SI-SDR', 'SDR', 'SI-SIR', 'SI-SAR', 'WAV-MSE', 'SPEC-MSE']

    # Dictionary to store combined data: (filepath, instrument, model) -> {embedding: mse}
    combined_data = {}
    
    # Walk through results directory - only look for embd_mse* files to avoid reading output summary
    for csv_file in results_path.rglob('*_embmse_results.csv'):
        # Extract model name and embedding type from path
        path_parts = csv_file.parts
        model_name = None
        embedding_name = None
        
        # Find model name in path (should be a subdirectory of results_path)
        for idx, part in enumerate(path_parts):
            if part in svs_model_names:
                model_name = part
                break
        
        # Extract embedding name from filename (e.g., MERT-v1-95M_embmse_results.csv)
        filename = csv_file.stem  # Remove .csv extension
        for met in metric_names:
            if met in filename:
                metric_name = met
                break
        
        if not model_name or not metric_name:
            continue
        
        # Read CSV file
        df = pd.read_csv(csv_file, header=None)
        
        # If dataframe is empty or first column looks like filepath, file has no header
        # if df.empty or (len(df) > 0 and 'audio' in str(df.iloc[0, 0])):
        #     df = pd.read_csv(csv_file, header=None)
        
        # Process each row
        for _, row in df.iterrows():
            filepath = row.iloc[0]  # First column is filepath
            mse_value = row.iloc[1]  # Second column is MSE value
            
            # Extract instrument name from filepath (bass, drums, vocals, other)
            instrument = None
            instrument_match = re.search(r'(bass|drums|vocals|other)', filepath, re.IGNORECASE)
            if instrument_match:
                instrument = instrument_match.group(1).lower()
            
            track = filepath.split(os.path.sep)[-2]             
            
            # Create composite key: (filepath, instrument, model_name)
            key = (filepath, instrument, model_name, track)
            
            if key not in combined_data:
                combined_data[key] = {}
            
            # Store MSE value with metric name
            combined_data[key][metric_name] = mse_value

    
    # Convert to DataFrame with restructured format
    if not combined_data:
        print("No data found to combine")
        return
    
    rows = []
    for (filepath, instrument, model_name, track), emb_values in combined_data.items():
        row = {
            'filepath': filepath,
            'track': track,
            'instrument_name': instrument,
            'model_name': model_name
        }
        # Add metric MSE values
        for met in metric_names:
            col_name = f"{met}-MSE"
            row[col_name] = emb_values.get(met, None)
        rows.append(row)
    
    df_combined = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = ['filepath', 'track', 'instrument_name', 'model_name']
    for met in metric_names:
        column_order.append(f"{met}-MSE")
    df_combined = df_combined[column_order]
    
    # Save combined CSV
    output_path = results_path / output_file
    df_combined.to_csv(output_path, index=False)
    print(f"Summarized results saved to: {output_path}")
    
    return df_combined

def main():
    # Select dataset via global DATASET variable (set at top of file)
    if DATASET == 'gensvs':
        sep_path = './audio/gensvs_eval_audio'
        tgt_path = os.path.join(sep_path, 'target')
        svs_model_names = DEFAULT_MODELS_GENSVS
    else:
        sep_path = SEP_PATH
        tgt_path = TGT_PATH
        svs_model_names = DEFAULT_MODELS

    stems = STEMS

    # Use global embedding selection
    embedding = EMBEDDING
    models = {m.name: m for m in get_all_models()}
    if embedding not in models:
        raise ValueError(f"Embedding model '{embedding}' not found among available models: {list(models.keys())}")
    model = models[embedding]
    embmse = EmbeddingMSE(model, audio_load_worker=WORKERS)
    fad = FrechetAudioDistance(model, audio_load_worker=WORKERS)
    # Clean up cache folders if flag is enabled
    if DELETE_CACHE:
        print(f"Cleaning cache folders from {tgt_path}...")
        cleanup_cache_folders(tgt_path, delete_cache=True)
        print(f"Cleaning cache folders from {sep_path}...")
        cleanup_cache_folders(sep_path, delete_cache=True)
        print()

    target_wavs = glob.glob(os.path.join(tgt_path, '**', '*.wav'), recursive=True)
    # discard accompaniment and mixture files from target paths
    target_wavs = [f for f in target_wavs if 'accompaniment' not in f and 'mixture' not in f]
    # discard embedding cache folders
    target_wavs = [f for f in target_wavs if 'convert' not in f and 'embeddings' not in f] 
    
    sep_wavs = glob.glob(os.path.join(sep_path, '**', '*.wav'), recursive=True)
    # discard accompaniment files from paths of separated files
    sep_wavs = [f for f in sep_wavs if 'accompaniment' not in f] 
    # discard embedding cache folders
    sep_wavs = [f for f in sep_wavs if 'convert' not in f and 'embeddings' not in f] 

    #cache all embeddings for target wavs
    for target_wav in tqdm.tqdm(target_wavs, desc="Caching target embeddings"):
        embmse.cache_embedding_file(target_wav)
      
        
    for sep_wav in tqdm.tqdm(sep_wavs, desc="Caching embeddings of separated audio files"):
        embmse.cache_embedding_file(sep_wav)
    
   
    # Determine separation and target folders differently for gensvs dataset (flat structure)
    if DATASET == 'gensvs':
        # For gensvs, each model directory contains separated files directly, and there is a single 'target' folder
        sep_folders = [os.path.join(sep_path, m) for m in svs_model_names if os.path.isdir(os.path.join(sep_path, m))]
        target_folders = [tgt_path] if os.path.isdir(tgt_path) else []
    else:
        sep_folders = glob.glob(os.path.join(sep_path, '*/*'))
        target_folders = glob.glob(os.path.join(tgt_path, '*'))
        # filter sep_folders and target_folders according to svs_model_names
        sep_folders = [f for f in sep_folders if os.path.basename(os.path.dirname(f)) in svs_model_names]
        target_folders = [f for f in target_folders if os.path.basename(f) in [os.path.basename(sf) for sf in sep_folders]]

    # For gensvs, only vocals stem is available
    if DATASET == 'gensvs':
        stems = ['vocals']

    for sep_folder in tqdm.tqdm(sep_folders, desc="Processing separated folders"):
        # For gensvs, sep_folder is a model directory; otherwise it's a per-track folder
        if DATASET == 'gensvs':
            model_name = os.path.basename(sep_folder)
            target_folder = tgt_path
            csv_dir = os.path.join(OUT_DIR, model_name)
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = Path(os.path.join(csv_dir, embedding+'_embmse_results.csv'))
            csv_path_fad = Path(os.path.join(csv_dir, 'FAD'+embedding+'_embmse_results.csv'))
        else:
            target_folder = os.path.join(tgt_path, os.path.basename(sep_folder))
            model_name = os.path.basename(os.path.dirname(sep_folder))
            csv_path = Path(os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), embedding+'_embmse_results.csv'))
            csv_path_fad = Path(os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), 'FAD'+embedding+'_embmse_results.csv'))

        if 'REP1' in model_name and DATASET != 'gensvs':
            #use vocal only target folders
            target_folder = target_folder.replace('musdb18_test_10s', 'musdb18_test_10s_vocals_only')

        os.makedirs(csv_path.parent, exist_ok=True)
        os.makedirs(csv_path_fad.parent, exist_ok=True)

        if SHUFFLE_EMBD:
            fad.mse_song2song(target_folder, sep_folder, csv_path)
            fad.score_song2song_shuffle_emb(target_folder, sep_folder, csv_path_fad)

        else:
            embmse.embedding_mse(target_folder, sep_folder, csv_path)
            fad.score_song2song(target_folder, sep_folder, csv_path_fad)

    for sep_folder in sep_folders:
        # dataset-specific path handling
        if DATASET == 'gensvs':
            model_name = os.path.basename(sep_folder)
            target_folder = tgt_path
            sisdr_csv = os.path.join(OUT_DIR, model_name, "SI-SDR_embmse_results.csv")
            sdr_csv = os.path.join(OUT_DIR, model_name, "SDR_embmse_results.csv")
            sisir_csv = os.path.join(OUT_DIR, model_name, "SI-SIR_embmse_results.csv")
            sisar_csv = os.path.join(OUT_DIR, model_name, "SI-SAR_embmse_results.csv")
            os.makedirs(os.path.join(OUT_DIR, model_name), exist_ok=True)
            files_to_check = [sisdr_csv, sdr_csv, sisir_csv, sisar_csv]
        else:
            model_name = os.path.basename(os.path.dirname(sep_folder))
            target_folder = os.path.join(tgt_path, os.path.basename(sep_folder))
            # Check if BSS-eval results already exist
            sisdr_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "SI-SDR_embmse_results.csv")
            sdr_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "SDR_embmse_results.csv")
            sisir_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "SI-SIR_embmse_results.csv")
            sisar_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "SI-SAR_embmse_results.csv")
            os.makedirs(os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder)), exist_ok=True)
            files_to_check = [sisdr_csv, sdr_csv, sisir_csv, sisar_csv]

        if SKIP_BSS_EVAL and all(os.path.exists(f) for f in files_to_check):
            print(f"Skipping BSS-eval for {sep_folder} (results already exist)")
            continue
        
        # Compute multi-source BSS-eval metrics (provides meaningful SI-SIR values) only for multi-stem datasets
        multi_source_results = {} if DATASET == 'gensvs' else compute_multisource_bss_eval(sep_folder, target_folder, stems)
        
        sep_files = glob.glob(os.path.join(sep_folder, '*.wav'))

        sisdr_dict = {}
        sdr_dict = {}
        sir_dict = {}
        sar_dict = {}

        for sep_wav in tqdm.tqdm(sep_files,desc=sep_folder+": Calculating BSS-eval metrics for separated files"):
            sep_audio, fs = sf.read(sep_wav)
            if len(sep_audio.shape) > 1:
                sep_audio = np.mean(sep_audio, axis=1)
            # build target path (dataset-specific)
            if DATASET == 'gensvs':
                # sep_wav example: .../htdemucs/separated_vocals_fileid_0.wav
                # target file: target_fileid_0.wav located in tgt_path
                basename = os.path.basename(sep_wav)
                import re
                m = re.search(r'fileid_\d+', basename)
                if m:
                    target_filename = f"target_{m.group(0)}.wav"
                    target_wav = os.path.join(tgt_path, target_filename)
                else:
                    target_wav = os.path.join(
                        tgt_path,
                        os.path.sep.join(sep_wav.split(os.path.sep)[-2:])
                    )
            else:
                target_wav = os.path.join(
                    tgt_path,
                    os.path.sep.join(sep_wav.split(os.path.sep)[-2:])
                )
            targ_audio, fs = sf.read(target_wav)
            
            # zero-mean the signals
#            sep_audio = sep_audio - np.mean(sep_audio)
#            targ_audio = targ_audio - np.mean(targ_audio)

            sisdr = scale_invariant_signal_distortion_ratio(
                torch.tensor(sep_audio).unsqueeze(0),
                torch.tensor(targ_audio).unsqueeze(0),
                fs
            ).item()

            sdr = signal_distortion_ratio(
                torch.tensor(sep_audio).unsqueeze(0),
                torch.tensor(targ_audio).unsqueeze(0),
                fs
            ).item()

            # use filename as key (or keep sep_wav if you prefer full path)
            key = sep_wav.split('./')[-1]
            sisdr_dict[key] = sisdr
            sdr_dict[key] = sdr
            
            # For multi-stem datasets compute SI-SIR and SI-SAR
            if DATASET != 'gensvs':
                # Use multi-source results for SI-SIR and SI-SAR (more accurate)
                sir_dict[key] = multi_source_results[key]['SI-SIR']
                sar_dict[key] = multi_source_results[key]['SI-SAR']
            else:
                 # For gensvs: Calculate SI-SIR and SI-SAR using available stems
                basename = os.path.basename(sep_wav)
                import re
                m = re.search(r'fileid_\d+', basename)
                file_id = m.group(0) # e.g. fileid_123
                # target_wav is already set to vocals reference
                # Construct paths for other stems
                # Note: tgt_path points to 'target' folder (vocals usually)
                gensvs_root = os.path.dirname(tgt_path)
                
                # Assuming structure: root/bass/bass_{id}.wav, etc.
                bass_path = os.path.join(gensvs_root, 'bass', f'bass_{file_id}.wav')
                drums_path = os.path.join(gensvs_root, 'drums', f'drums_{file_id}.wav')
                other_path = os.path.join(gensvs_root, 'other', f'other_{file_id}.wav')
                
                # Load all refs in order: bass, drums, vocals, other (vocals at index 2)
                stems_files = [bass_path, drums_path, target_wav, other_path]
                valid_stems = []
                refs = []
                for f_path in stems_files:
                    sig, fs_tmp = sf.read(f_path)
                    if len(sig.shape) > 1: sig = np.mean(sig, axis=1)
                    
                    rms = np.sqrt(np.mean(sig**2))
                    if rms < 1e-6:
                        print(f"Warning: {f_path} has very low RMS ({rms:.2e}), not used as reference signal for {sep_wav}")
                        continue
                    else:
                        sig = sig - np.mean(sig) # zero-mean
                        refs.append(sig)
                        valid_stems.append(f_path)
                        
                tgt_idx = [stm_idx for stm_idx, f in enumerate(valid_stems) if 'target' in f][0]
                ref_sources = np.stack(refs, axis=1) # (n_samples, 4)
                
                # Prepare estimate (vocals)
                est = sep_audio - np.mean(sep_audio)
                estimate = est[:, None] # (n_samples, 1)
                
                # Evaluate against vocals (index 2)
                _, si_sir, si_sar, _, _, _ = _scale_bss_eval(
                    ref_sources, 
                    estimate, 
                    idx=tgt_idx, 
                    compute_sir_sar=True
                )
                sir_dict[key] = si_sir
                sar_dict[key] = si_sar

        # create DataFrames
        sisdr_df = pd.DataFrame(
            [(k, v) for k, v in sisdr_dict.items()],
            columns=["filepath", "SI-SDR"]
        )

        sdr_df = pd.DataFrame(
            [(k, v) for k, v in sdr_dict.items()],
            columns=["filepath", "SDR"]
        )

        sisir_df = pd.DataFrame(
            [(k, v) for k, v in sir_dict.items()],
            columns=["filepath", "SI-SIR"]
        )
        
        sisar_df = pd.DataFrame(
            [(k, v) for k, v in sar_dict.items()],
            columns=["filepath", "SI-SAR"]
        )

        # save CSVs per separation folder (paths already defined above)
        sisdr_df.to_csv(sisdr_csv, header=False, index=False)
        sdr_df.to_csv(sdr_csv, header=False, index=False)
        sisir_df.to_csv(sisir_csv, header=False, index=False)
        sisar_df.to_csv(sisar_csv, header=False, index=False)
    
    # Separate loop for WAV-MSE and SPEC-MSE calculations
    for sep_folder in tqdm.tqdm(sep_folders, desc="Calculate WAV-MSE and SPEC-MSE of separated audio files"):
        if DATASET == 'gensvs':
            model_name = os.path.basename(sep_folder)
            wavmse_csv = os.path.join(OUT_DIR, model_name, "WAV-MSE_embmse_results.csv")
            specmse_csv = os.path.join(OUT_DIR, model_name, "SPEC-MSE_embmse_results.csv")
            os.makedirs(os.path.join(OUT_DIR, model_name), exist_ok=True)
            files_to_check = [wavmse_csv, specmse_csv]
        else:
            model_name = os.path.basename(os.path.dirname(sep_folder))
            # Check if WAV-MSE and SPEC-MSE results already exist
            wavmse_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "WAV-MSE_embmse_results.csv")
            specmse_csv = os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder), "SPEC-MSE_embmse_results.csv")
            files_to_check = [wavmse_csv, specmse_csv]
            os.makedirs(os.path.join(OUT_DIR, model_name, os.path.basename(sep_folder)), exist_ok=True)
        
        if SKIP_WAV_SPEC_MSE and all(os.path.exists(f) for f in files_to_check):
            print(f"Skipping WAV-MSE and SPEC-MSE for {sep_folder} (results already exist)")
            continue
        
        sep_files = glob.glob(os.path.join(sep_folder, '*.wav'))
        
        wav_mse_dict = {}
        spec_mse_dict = {}
        
        for sep_wav in sep_files:
            sep_audio, fs = sf.read(sep_wav)
            if len(sep_audio.shape) > 1:
                sep_audio = np.mean(sep_audio, axis=1)
            # build target path (dataset-specific)
            if DATASET == 'gensvs':
                basename = os.path.basename(sep_wav)
                import re
                m = re.search(r'fileid_\d+', basename)
                if m:
                    target_filename = f"target_{m.group(0)}.wav"
                    target_wav = os.path.join(tgt_path, target_filename)
                else:
                    target_wav = os.path.join(
                        tgt_path,
                        os.path.sep.join(sep_wav.split(os.path.sep)[-2:])
                    )
            else:
                target_wav = os.path.join(
                    tgt_path,
                    os.path.sep.join(sep_wav.split(os.path.sep)[-2:])
                )
            targ_audio, fs = sf.read(target_wav)
                        
            # compute WAV-MSE and SPEC-MSE
            wav_mse = np.mean((targ_audio - sep_audio) ** 2)
            
            sep_stft = torch.stft(torch.tensor(sep_audio), n_fft=512, hop_length=256, return_complex=True, window=torch.hann_window(512))
            targ_stft = torch.stft(torch.tensor(targ_audio), n_fft=512, hop_length=256, return_complex=True, window=torch.hann_window(512))
            spec_mse = torch.mean((torch.abs(targ_stft) - torch.abs(sep_stft)) ** 2).item()
            
            # use filename as key
            key = sep_wav.split('./')[-1]
            wav_mse_dict[key] = wav_mse
            spec_mse_dict[key] = spec_mse
        
        # create DataFrames
        wav_mse_df = pd.DataFrame(
            [(k, v) for k, v in wav_mse_dict.items()],
            columns=["filepath", "WAV-MSE"]
        )
        
        spec_mse_df = pd.DataFrame(
            [(k, v) for k, v in spec_mse_dict.items()],
            columns=["filepath", "SPEC-MSE"]
        )
        
        # save CSVs
        wav_mse_df.to_csv(wavmse_csv, header=False, index=False)
        spec_mse_df.to_csv(specmse_csv, header=False, index=False)
        
         
    # Ensure output directory exists (avoid failure when no per-model folders were produced)
    os.makedirs(OUT_DIR, exist_ok=True)
    combine_emb_mse_results(OUT_DIR, 'emb_mse_results.csv')
    print("All done.")



if __name__ == "__main__":
    main()