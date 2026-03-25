# 2025 Bake-off: Listener Ratings and Separated Audio

## Authors
Noah Jaffe, John Ashley Burgoyne

## DOI 
[https://doi.org/10.5281/zenodo.15843081](https://doi.org/10.5281/zenodo.15843081)

## Description

This dataset contains listener evaluation data and separated audio excerpts used in our perceptual study of music source separation methods. It supplements—but is distinct from—the dataset **"SiSEC18-MUS 30s Excerpts (1.0.0)"** by Stöter et al. ([https://doi.org/10.5281/zenodo.1256003](https://doi.org/10.5281/zenodo.1256003)). 

We only include `.wav` files that **we generated ourselves** using the separation models studied in our experiment. We do **not** redistribute the original IRM or REP1 separated signals from SiSEC18, nor do we redistribute any of the ground-truth data. Those parts of the dataset are available below in their corresponding datasets. 

### Contents

- `raw_listener_responses_w_violations.csv`: Listener ratings with flags for protocol violations.
- `separated_audio/`: Directory of `.wav` files that were presented to listeners during the perceptual study, generated from the MUSDB18 30-second excerpts using various source separation models.

## License

This dataset is released under a [Creative Commons BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
You are free to share and adapt the material for **non-commercial use**, provided that:
- You give appropriate credit,
- You apply the same license to derivative works,
- You do not impose additional restrictions (no DRM, etc.).

---

## CSV Column Descriptions: `raw_listener_responses_w_violations.csv`

This file contains one row per user-session and track-stem evaluation, aggregated from raw logs. The key columns are:

| Column Name                         | Description |
|------------------------------------|-------------|
| `session_uuid`                     | Unique identifier for the user session |
| `track`                            | Name of the MUSDB18 Track |
| `stem type`                        | Type of source evaluated (e.g., vocals, drums) |
| `IRM1`, `Open-UMix`, `REP1`, `SCNet-large`, `htdemucs_ft` | **Listener ratings** (0–100) for each separation method |
| `lpf_anchor`                       | Rating (0–100) for a **low-pass filtered anchor** (used as a perceptual floor) |
| `reference`                        | Rating (0–100) for the original, unprocessed audio |
| `rating_time`                      | Time (in seconds) the user took to complete this rating set |
| `rating_std`                       | Intra-set standard deviation across all ratings given by this user |
| `violation_reference_anchor_spread` | Flag (1/0) if the user rated `reference` less than 5 points above `lpf_anchor` (suggests misunderstanding of rating scale) |
| `violation_reference_high`         | Flag (1/0) if the user rated `reference` below 90 (indicates low rating of gold-standard reference) |
| `violation_variability`            | Flag (1/0) if the user's rating standard deviation was below 20, indicating insufficient use of the rating range |
| `violation_time`                   | Flag (1/0) if rating time was under 20s or over 213s (extremely fast or slow completion) |
| `violation_total`                  | Sum of all above violation flags (integer from 0 to 4) |

These columns were generated through preprocessing using the following criteria:

- **Reference vs Anchor Violation**: `reference - lpf_anchor < 5`
- **Low Reference Rating Violation**: `reference < 90`
- **Low Variability Violation**: `rating_std < 20`
- **Timing Violation**: `rating_time < 20` or `rating_time > 213` seconds

A higher `violation_total` may suggest lower data quality for a particular session or respondent.

---

## Related Resources

- Source dataset: [*MUSDB18 Dataset*](https://source-separation.github.io/tutorial/data/musdb18.html)
- Source dataset: *Stöter et al., "SiSEC18-MUS 30s Excerpts"*, Zenodo [https://doi.org/10.5281/zenodo.1256003](https://doi.org/10.5281/zenodo.1256003)
- [SIGSEP MUS 30s Cut-List Generator](https://github.com/sigsep/sigsep-mus-cutlist-generator?tab=readme-ov-file)
- Paper describing this dataset: Jaffe, N., & Burgoyne, J. A. (2025). Musical Source Separation Bake-Off: Comparing Objective Metrics with Human Perception. In Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE. Preprint available: https://arxiv.org/abs/2507.06917

---

## Citation

If you use this dataset, please cite it as: Jaffe, N., & Burgoyne, J. A. (2025). Musical Source Separation Bake-Off: Comparing Objective Metrics with Human Perception. In Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). 