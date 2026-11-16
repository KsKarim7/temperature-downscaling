Data checks passed for all checks in both uae and bangladesh regions with 17544 total hours evaluated.

For the UAE region test_blocks split, bilinear achieved rmse_c of 1.1883, mae_c of 0.9247, bias_c of 0.4163, and skill of 0.0. lgbm achieved rmse_c of 0.8277, mae_c of 0.6263, bias_c of 0.0564, and skill of 0.3035. unet_main achieved rmse_c of 0.7815, mae_c of 0.5867, bias_c of 0.05, and skill of 0.3423.

For the Bangladesh region test_blocks split, bilinear achieved rmse_c of 1.4736, mae_c of 1.1952, bias_c of 0.5362, and skill of 0.0. lgbm achieved rmse_c of 0.9437, mae_c of 0.7179, bias_c of 0.0022, and skill of 0.3596. unet_main achieved rmse_c of 0.9167, mae_c of 0.6934, bias_c of -0.131, and skill of 0.3779.

Comparing validation to test_blocks split performance in UAE, bilinear validation rmse_c was 1.2908 versus test_blocks 1.1883. lgbm validation rmse_c was 0.6576 versus test_blocks 0.8277. unet_main validation rmse_c was 0.7048 versus test_blocks 0.7815. In Bangladesh, bilinear validation rmse_c was 1.3836 versus test_blocks 1.4736. lgbm validation rmse_c was 0.6483 versus test_blocks 0.9437. unet_main validation rmse_c was 0.7355 versus test_blocks 0.9167.

For unet_main in UAE, highest rmse_c occurred at local_hour 8 with 1.0332 and local_hour 7 with 1.0246, while lowest skill occurred at local_hour 12 with 0.0333 and local_hour 21 with 0.0377. In Bangladesh, highest rmse_c occurred at local_hour 12 with 1.0538 and local_hour 13 with 1.0525, while lowest skill occurred at local_hour 4 with 0.1591.

Station evaluations use era5land as reference. Across 38 Bangladesh stations, era5land median_rmse was 1.4951 and mean_bias 0.1142; bilinear was 1.6887 and 0.5874; lgbm was 1.4735 and 0.0467; unet_main was 1.4566 and -0.0125. Across 14 UAE stations, era5land median_rmse was 1.7516 and mean_bias -0.8118; bilinear was 1.4242 and -0.3395; lgbm was 1.7378 and -0.7301; unet_main was 1.6992 and -0.8274.

Caveats include point to grid representativeness mismatch at observational stations, uneven spatial distribution of station networks, and potential spatial smoothing from spatial reanalysis targets.