Setup: This project targets the UAE region, which also covers Qatar and northern Oman, and the Bangladesh region, which also covers Kolkata and parts of West Bengal and Meghalaya. In the Bangladesh region, station 42423 (Jorhat / Lengrabhita) was excluded as it was mislocated by approximately 400 km.

Data: The evaluation covers the 2024 test year, comprising 8784 hours of data.

Results:

UAE region
| method | rmse_c | mae_c | bias_c | skill |
| :--- | :--- | :--- | :--- | :--- |
| bilinear | 1.1883 | 0.9247 | 0.4163 | 0.0 |
| lgbm | 0.8277 | 0.6263 | 0.0564 | 0.3035 |
| unet_main | 0.7815 | 0.5867 | 0.05 | 0.3423 |

Bangladesh region
| method | rmse_c | mae_c | bias_c | skill |
| :--- | :--- | :--- | :--- | :--- |
| bilinear | 1.4736 | 1.1952 | 0.5362 | 0.0 |
| lgbm | 0.9437 | 0.7179 | 0.0022 | 0.3596 |
| unet_main | 0.9167 | 0.6934 | -0.131 | 0.3779 |

The unet_main method generalises best to held-out test blocks in both regions. The gap between val and test_blocks performance is smaller for unet_main (UAE region: 0.0767, Bangladesh region: 0.1812) compared to lgbm (UAE region: 0.1701, Bangladesh region: 0.2954).

Hours:

UAE region (unet_main)
- 8:00: 1.0332
- 7:00: 1.0246
- 6:00: 0.9229

Bangladesh region (unet_main)
- 12:00: 1.0538
- 13:00: 1.0525
- 15:00: 1.0486

The two regions degrade at different times of day.

Stations:

In the UAE region (14 stations), era5land had a median_rmse of 1.7516 and mean_bias of -0.8118, while lgbm scored 1.7378 and -0.7301, and unet_main scored 1.6992 and -0.8274. In the Bangladesh region (38 stations), era5land had a median_rmse of 1.4951 and mean_bias of 0.1142, while lgbm scored 1.4735 and 0.0467, and unet_main scored 1.4566 and -0.0125. In the UAE region, era5land and the models share a cold bias.

Caveats:

- The test_blocks margin between unet_main and lgbm, set against run to run noise, is -0.0462 (UAE region) and -0.027 (Bangladesh region), with run_to_run_noise_c of 0.0044 and margin_over_noise of -10.5 (UAE region) and -6.1364 (Bangladesh region).
- One block partition and one split seed, so these are a single draw, not a distribution.
- ERA5-Land is a model product, not observations, so agreement with it is not accuracy.
- The UAE region station cold bias is inherited from the training target.

---

model gemini-3.1-flash-lite, provider gemini, run 2026-11-17

guardrail: traced 73 of 73 numbers
