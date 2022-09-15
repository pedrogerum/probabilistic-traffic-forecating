ARGS="--quantile_boundary=1e-3 --pred_length=3" #each time step is 5 minutes

DATASET_NAME=chic


OMP_NUM_THREADS=3 nohup python -u quantile_forecast.py  data/data_to_tune/${DATASET_NAME}/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn --config_path ./configs/test/config_chic_15_nonmonot.json --dataset_outlier_path data/chicago_outlier.gluonts $ARGS > chicago15min_nonmono.out &
OMP_NUM_THREADS=3 nohup python -u quantile_forecast.py  data/data_to_tune/${DATASET_NAME}/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn_mono --config_path ./configs/test/config_chic_15_monot.json --dataset_outlier_path data/chicago_outlier.gluonts $ARGS --monotonic True > chicago15min_mono.out &

