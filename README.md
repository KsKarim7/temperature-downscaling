# Temperature Downscaling

This downscales hourly ERA5 2 m temperature from 0.25 degree to the 0.1 degree ERA5-Land grid using terrain and land cover, for two regions. On held-out spatial blocks in the 2024 test year, the U-Net reaches 0.7815 C RMSE in the UAE region and 0.9167 C in the Bangladesh region, against 0.8277 C and 0.9437 C for LightGBM and 1.1883 C and 1.4736 C for bilinear interpolation.

## Results at a glance

All numbers are RMSE in degrees C on the 2024 test year. Validation blocks and test blocks are different cells.

UAE region

| method | test_blocks RMSE | val RMSE | val to test gap |
| --- | --- | --- | --- |
| bilinear | 1.1883 | 1.2908 | n/a |
| LightGBM | 0.8277 | 0.6576 | 0.1701 |
| U-Net | 0.7815 | 0.7048 | 0.0767 |

Bangladesh region

| method | test_blocks RMSE | val RMSE | val to test gap |
| --- | --- | --- | --- |
| bilinear | 1.4736 | 1.3836 | n/a |
| LightGBM | 0.9437 | 0.6483 | 0.2954 |
| U-Net | 0.9167 | 0.7355 | 0.1812 |

The margin in `reports/model_margins.csv` is U-Net minus LightGBM, so a negative number means the U-Net is ahead. It is -0.0462 in the UAE region and -0.027 in the Bangladesh region. Two Kaggle runs of the same configuration differed by 0.0044, so the margin is -10.5 and -6.1364 times that noise level.

The more interesting column is the last one. LightGBM wins on validation in both regions and loses on test blocks in both, because its val to test gap is roughly twice the U-Net's. Picking on validation alone would have picked the wrong model.

## Why

A 0.25 degree reanalysis cell is about 28 km across. Inside one cell sit mountains, coastline, irrigated farmland and a city, and the coarse model represents all of it with a single temperature. ERA5-Land runs at 0.1 degree with a better land surface, so the difference between the two is a repeatable function of terrain and land cover rather than noise. I predict that difference as a residual, which keeps the model's job small and keeps the coarse field as the baseline to beat.

## Data

| source | resolution | role |
| --- | --- | --- |
| ERA5 single levels, 2 m temperature | 0.25 deg hourly | model input, the coarse field |
| ERA5-Land, 2 m temperature | 0.1 deg hourly | prediction target |
| Copernicus DEM GLO-90 | 90 m | elevation, roughness, anomaly |
| ESA WorldCover v200 | 10 m | seven land-cover fractions per cell |
| Meteostat hourly stations | point | independent validation only, never training |

Two regions, given as north, west, south, east:

| region | bbox | also covers |
| --- | --- | --- |
| UAE region | 26.5, 51.0, 22.5, 56.5 | Qatar and northern Oman |
| Bangladesh region | 26.7, 88.0, 20.6, 92.7 | Kolkata, parts of West Bengal and Meghalaya |

The region names are shorthand for the boxes, not claims about national coverage. Both are labelled "UAE region" and "Bangladesh region" everywhere in the figures and tables for that reason.

Training uses 2023 and testing uses 2024, so no test hour has a training hour from the same day.

## Method

1. Download ERA5, ERA5-Land and the DEM tiles, and read WorldCover remotely through its internal overviews rather than downloading 10 m rasters.
2. Reproject everything to the ERA5-Land 0.1 degree grid. ERA5 is bilinearly interpolated to the fine grid and padded by 0.5 degree first, so no fine cell sits outside the coarse field.
3. Build static features once per region: elevation and its standard deviation within the cell, the elevation anomaly, and seven land-cover fractions by area-weighted aggregation.
4. Train on the residual, ERA5-Land minus interpolated ERA5, and add the coarse field back at prediction time.
5. Two models: LightGBM on per-cell rows, and a U-Net on 32 by 32 crops with a masked loss so only land cells contribute.

Input channels:

| channel | what it captures |
| --- | --- |
| t2m_coarse | the ERA5 field being corrected |
| elev | height, the main driver through the lapse rate |
| elev_std | roughness within the cell, where a single height is a poor summary |
| elev_anom | height relative to the surrounding coarse cell, the terrain the coarse model cannot see |
| lc_tree, lc_shrub, lc_grass, lc_crop, lc_built, lc_bare, lc_water | surface type, which sets heat capacity and evaporation |
| lat2d, lon2d | position |
| hour_sin, hour_cos, doy_sin, doy_cos | time of day and season, in local time |

## Evaluation design

Neighbouring 0.1 degree cells are strongly correlated, so a random split puts a cell's neighbour in training and makes test error measure interpolation between known points rather than generalisation to new places. Instead the grid is divided into 5 by 5 blocks and 25 percent of eligible land blocks are held out for testing.

Validation started as held-out days on the training cells. That was wrong in a way that took a while to see: validation shared every cell with training, so early stopping and model selection rewarded memorising location. Validation blocks are now drawn spatially, 15 percent of blocks disjoint from the test blocks, combined with whole held-out days. Hours are converted to local time per region before any hourly breakdown, since 12:00 UTC is a different part of the day in each.

Making validation spatial changed what looked best. The LightGBM feature ablation in `reports/lgbm_ablation.csv` then showed that dropping the land-cover group gave the best pooled validation RMSE, 0.6516 C, which is what `reports/lgbm_choice.txt` records as the chosen configuration. Dropping position cost validation accuracy, 0.6901 against 0.6588 in the UAE region, while barely moving test-block error, 0.8318 against 0.834. That is the signature of features that help memorise rather than generalise, and it is only visible once validation is spatial.

For LightGBM the gain-based importance in `reports/lgbm_importance.csv` puts hour_sin first at 7635107.883006334 and lon2d second at 5531741.190969348, with elevation at 1848176.4699329138. A longitude feature ranking that high in a model whose job is terrain correction is a warning, not a finding.

## Training

The hyperparameter and feature study ran locally on a GTX 1650, 4 GB, in fp32, because GTX 16-series cards are unreliable in fp16. Seven configurations, 8 epochs each, in `reports/sweep.csv`:

| configuration | best val RMSE | minutes |
| --- | --- | --- |
| lr 0.001, l1, all features | 0.7263 | 11.5 |
| lr 0.001, mse, all features | 0.7392 | 8.9 |
| lr 0.0003, l1, all features | 0.7285 | 18.3 |
| lr 0.0003, mse, all features | 0.7267 | 14.1 |
| lr 0.001, l1, drop position | 0.7212 | 17.6 |
| lr 0.001, l1, drop landcover | 0.7235 | 11.9 |
| lr 0.001, l1, drop terrain | 0.7363 | 11.6 |

Dropping terrain is the only ablation that clearly hurts, 0.7363 against 0.7212 for the winner. Dropping land cover gives 0.7235 against the winner's 0.7212, a difference well inside run-to-run noise. Terrain is what the model actually uses. Note also that every run was still improving at its last epoch, so this ranks early training, not converged models.

Final training ran on Kaggle, two T4s with DistributedDataParallel and mixed precision, 40 epochs. `reports/kaggle_runs.csv` records 14 seconds per epoch there, against runs that took 11.5 to 18.3 minutes for 8 epochs locally. Best validation RMSE was 0.6667 C at epoch 40. A second run of the identical configuration in the same session reached 0.6623 C at epoch 34, so run-to-run noise is 0.0044 C. That number does real work later: it is the yardstick for whether any margin means anything. `reports/kaggle_train_log.txt` holds the full console output, including both rank lines confirming two GPUs.

`KAGGLE.md` has the notebook cells. Kaggle only trains; prediction and scoring run locally against the splits that produced the data, so the held-out blocks cannot drift.

## Results in detail

Figures: [rmse_by_hour.png](reports/figures/rmse_by_hour.png), [val_vs_test.png](reports/figures/val_vs_test.png), [geohash_rmse_uae.png](reports/figures/geohash_rmse_uae.png), [example_uae.png](reports/figures/example_uae.png).

Worst local hours for the U-Net, from `reports/worst_hours.csv`:

| region | worst hours (local) | RMSE |
| --- | --- | --- |
| UAE region | 8, 7, 6 | 1.0332, 1.0246, 0.9229 |
| Bangladesh region | 12, 13, 15 | 1.0538, 1.0525, 1.0486 |

The two regions fail at different times of day, and the physical reading differs with them. The UAE region's errors cluster at sunrise, when a dry desert surface heats quickly and small timing errors in the morning transition become large temperature errors. The Bangladesh region fails at midday, which is when convective cloud forms and shades the surface unevenly within a cell.

The U-Net does not win everywhere, and the pattern is not random. It is worse than LightGBM in 7 of 48 region-hour groups, 4 of 24 region-month groups and 54 geohash cells. The largest hourly loss is the Bangladesh region at 07 local, 0.9585 against 0.7966, and it loses again at 04, 05, 18, 21 and 23. The losses concentrate in the Bangladesh region's early morning and in its pre-monsoon months, March and April. So the aggregate win comes from being better across most of the distribution while losing in a specific and consistent corner of it.

The land-cover breakdown in `reports/metrics_by_landcover.csv` is uninformative, and it is worth saying why rather than quietly dropping it. Only classes with at least 20 test-block cells are reported. In the UAE region that leaves one class, bare, holding 244 cells, nearly every test block in the region, so the breakdown repeats the overall number. In the Bangladesh region it leaves two, crop at 281 cells and tree at 269, at 0.9226 and 0.9105, closer together than the 0.0044 run-to-run noise would let me call a difference. The regions are too homogeneous at this scale for a land-cover breakdown to say anything.

## Station validation

52 stations survive quality control, 14 in the UAE region and 38 in the Bangladesh region, compared against the nearest land cell for all of 2024. `reports/stations_summary.csv`:

| region | method | stations | median RMSE | mean bias |
| --- | --- | --- | --- | --- |
| UAE region | bilinear | 14 | 1.4242 | -0.3395 |
| UAE region | ERA5-Land | 14 | 1.7516 | -0.8118 |
| UAE region | LightGBM | 14 | 1.7378 | -0.7301 |
| UAE region | U-Net | 14 | 1.6992 | -0.8274 |
| Bangladesh region | bilinear | 38 | 1.6887 | 0.5874 |
| Bangladesh region | ERA5-Land | 38 | 1.4951 | 0.1142 |
| Bangladesh region | LightGBM | 38 | 1.4735 | 0.0467 |
| Bangladesh region | U-Net | 38 | 1.4566 | -0.0125 |

The ERA5-Land row is not a competing method. It is how the training target itself scores against the observations. Models can land slightly above or below it, as they do here, because part of the station error is representativeness: a station is a point and a cell is about 10 km across, so a smoothed field can match a point better than the noisier field it was trained on. The margins involved are small enough that I would not build an argument on them.

In the UAE region the raw bilinear field beats every model against stations, 1.4242 against 1.6992. That is not a model failure. ERA5-Land carries a -0.8118 mean bias against these 14 stations and the U-Net inherits it almost exactly at -0.8274. A model trained to reproduce ERA5-Land reproduces its offset too.

I tested the obvious explanation, that the stations are urban or coastal and so warmer than a land-only cell represents, and it does not hold. In `reports/stations_bias_drivers.csv` none of the UAE-region correlations is significant, as `reports/station_bias_stats.csv` records: built-up fraction gives rho -0.216 at p 0.459, the 3 by 3 land fraction 0.177 at p 0.544, and elevation difference 0.112 at p 0.703. Airports are less negative than non-airports, -0.7 against -0.924, the opposite of an urban heat story. The bias is negative at 13 of 14 stations across the whole range of built-up fraction, so it behaves like a regional offset, not a station-type effect.

Two stations do have a concrete explanation, and it is terrain rather than land use. Diba has an elevation difference of -345.9 m against its cell and the largest bias at -1.8725; Fujairah is at -52.6 m and -1.7605. Both are coastal towns beside the Hajar mountains, so the cell averages in terrain the station is nowhere near.

One station was excluded, recorded in `reports/stations_excluded.csv`: Meteostat id 42423 is named Jorhat, a town in eastern Assam, but its listed coordinates put it about 400 km west, and its listed elevation disagrees with the DEM for that cell by more than 500 m. The exclusion rule is metadata that cannot be right, never a large error. Removing it mattered: the one nominally significant correlation in the Bangladesh region lost its significance, settling at p 0.063, so a single bad record had been carrying it across the threshold.

## Report agent

`downscale/agent.py` is a small tool-calling agent that runs the data checks, runs the evaluation, reads the result CSVs and drafts [reports/agent_report.md](reports/agent_report.md). It talks to a free Gemini endpoint through the OpenAI-compatible API, so the project costs nothing to run. Tools: `get_run_info`, `check_data`, `list_reports`, `read_csv`, `calc`, `run_evaluation` and `write_report`.

The guardrails exist because of specific failures, in order:

- Every number in the report must appear in a file the agent actually read, at the same precision. The result is written to `reports/agent_report_check.txt`.
- Version one, kept at [reports/agent_report_v1.md](reports/agent_report_v1.md), used a correct number in the wrong context: 17544 hours is both years, not the 8784 hours of the test year it was describing. Tracing passed it, because the number was real. A specific rule now rejects that figure unless the words "both years" sit next to it.
- A smaller model, used after the first hit its daily quota, miscalculated one gap and invented another number outright. The guardrail caught both. Derived numbers then moved into code: `downscale/margins.py` writes every gap, margin and noise figure to `reports/model_margins.csv`, and the top hours to `reports/worst_hours.csv`. A `calc` tool covers anything else, and the prompt forbids mental arithmetic.
- The invented number had a root cause worth naming: `metrics_by_hour.csv` has 144 rows and `read_csv` truncated at 60, so the model never saw a single Bangladesh row and filled the gap. Truncation is now explicit in the output, the default is higher, and the tool takes a filter.
- A report that fails tracing is renamed to `agent_report_failed.md`, so a failed draft is never left sitting as the current report.

The honest limit: tracing checks numbers, not claims. The invented value also came with a wrong hour, and no check would have caught that if the number itself had happened to trace. The agent drafts; a person still has to read it. The accepted report carries a footer with the model, provider, date and traced count.

## Land-cover what-if

`downscale/whatif.py` asks what the U-Net learned to associate with built-up land. For cells at 10 percent or more built-up, it moves 0.2 of the built-up fraction into tree, holds every other input fixed, and re-runs the model over July and January 2024. A placebo applies the same shift in cells below 2 percent built-up, taking from bare or the largest non-tree class, which separates a built-up effect from a generic "more trees" response.

From `reports/whatif_summary.csv`, mean predicted change in C:

| region | month | group | cells | mean delta |
| --- | --- | --- | --- | --- |
| UAE region | july | built | 51 | -0.0154 |
| UAE region | july | placebo | 1172 | -0.2602 |
| UAE region | january | built | 51 | 0.0655 |
| UAE region | january | placebo | 1172 | 0.0669 |
| Bangladesh region | july | built | 39 | -0.0449 |
| Bangladesh region | july | placebo | 2117 | 0.043 |
| Bangladesh region | january | built | 39 | 0.0619 |
| Bangladesh region | january | placebo | 2117 | 0.092 |

The placebo is the whole result. In three of four cases it is as large as or larger than the built-up effect, and in the UAE region in July it is seventeen times larger, -0.2602 against -0.0154. The signs are also inconsistent: removing built-up area makes January warmer in both regions. What the model learned is a generic tree-fraction response, not an urban effect. Without the placebo, the +0.0655 figure could have been written up as a small urban signal.

[whatif.png](reports/figures/whatif.png) shows the hourly curves. This is a sensitivity test of a statistical model trained at about 10 km, not a physical simulation. It shows what the model associates with land cover, nothing causal. Intervention questions about urban heat need physical simulation and inputs far finer than this.

## Limitations

- The target is ERA5-Land, a model product, not observations. Agreement with it is not accuracy, and the station results show the two coming apart in the UAE region.
- 0.1 degree is about 10 km. Street-scale and neighbourhood-scale questions are out of reach, which the what-if result demonstrates rather than assumes.
- One training year and one test year. Seasonal behaviour is covered, interannual variability is not.
- One block partition and one split seed. These are a single draw, not a distribution over partitions.
- No building footprints, heights or morphology, which is most of what distinguishes one urban cell from another.
- 14 stations in the UAE region is few, and 2 of 38 Bangladesh-region stations are coastal, so the coastal contrast there rests on almost nothing.

## Next steps

- Building footprints and Landsat land surface temperature as inputs, which would give the model something urban to learn from.
- A finer target than ERA5-Land, from dense station networks where they exist.
- Generative downscaling, such as a diffusion model over the residual, which would give a distribution per cell rather than a single number.
- Forecast lead times rather than reanalysis, which is where this becomes operationally useful.

## Reproduce

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

torch is not in requirements.txt so the build can match the local GPU.

Put your CDS credentials in `%USERPROFILE%\.cdsapirc` as the Climate Data Store documents. Do not commit that file.

```powershell
.\.venv\Scripts\python -m downscale.download --region all
.\.venv\Scripts\python -m downscale.prepare --region all
.\.venv\Scripts\python -m downscale.checks --region all
.\.venv\Scripts\python -m downscale.baselines
.\.venv\Scripts\python -m downscale.sweep
```

Run one download process at a time. The Climate Data Store limits queued requests per account, and a second process gets everything rejected.

Train on Kaggle following `KAGGLE.md`, bring back `checkpoints/unet_main.pt`, then:

```powershell
.\.venv\Scripts\python -m downscale.predict --name unet_main
.\.venv\Scripts\python -m downscale.evaluate
.\.venv\Scripts\python -m downscale.stations
.\.venv\Scripts\python -m downscale.stations --drivers
.\.venv\Scripts\python -m downscale.margins
.\.venv\Scripts\python -m downscale.whatif --shift 0.2
```

The agent needs a key for the current session only:

```powershell
$env:GEMINI_API_KEY = "your-key-here"
.\.venv\Scripts\python -m downscale.agent --list-models
.\.venv\Scripts\python -m downscale.agent
```

Gemini model names change and free quotas move with them. `--list-models` shows what the endpoint currently offers; set `agent.gemini.model` in `config.yaml`, or leave it empty to pick a current Flash model automatically.

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Repo layout

```
downscale/
  config.py      load config.yaml
  download.py    ERA5, ERA5-Land, DEM, WorldCover URLs
  prepare.py     align everything onto the 0.1 deg grid
  checks.py      data sanity checks
  splits.py      spatial test and validation blocks, validation days
  features.py    channel list, groups, time and static features
  baselines.py   bilinear and LightGBM, feature ablation
  unet.py        the model
  data.py        random crop dataset, normalisation
  train.py       training, AMP, DDP
  sweep.py       hyperparameter and feature study
  predict.py     full-domain prediction and scoring
  evaluate.py    breakdowns by hour, month, land cover, geohash
  stations.py    Meteostat validation and bias drivers
  margins.py     derived gaps, margins, worst hours
  whatif.py      land-cover sensitivity test
  agent.py       tool-calling report agent and guardrails
tests/           unit tests, no network
reports/         every number in this README comes from here
KAGGLE.md        the two-GPU training run
```

## Licence

Code is MIT, see [LICENSE](LICENSE). The data keep their own licences: ERA5 and ERA5-Land and the Copernicus DEM under the Copernicus terms, ESA WorldCover under CC BY 4.0, and Meteostat station data under its own terms.
