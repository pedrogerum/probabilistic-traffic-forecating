ARGS="--quantile_boundary=1e-3 --pred_length=15"

# DATASET_NAME=chic_d
# python quantile_forecast.py  data/${DATASET_NAME}.gluonts  ./models/${DATASET_NAME}_deepar --deepar True
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_deepar ${DATASET_NAME}_deepar ./data/examples/${DATASET_NAME} --diffed True

# python quantile_forecast.py  data/${DATASET_NAME}.gluonts  ./models/${DATASET_NAME}_mqrnn --config_path ./configs/diff.json $ARGS 
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn ${DATASET_NAME}_mqrnn ./data/examples/${DATASET_NAME} --diffed True

# python quantile_forecast.py  data/${DATASET_NAME}.gluonts  ./models/${DATASET_NAME}_mqrnn_mono --config_path ./configs/diff.json $ARGS --monotonic True
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn_mono ${DATASET_NAME}_mqrnn_mono ./data/examples/${DATASET_NAME} --diffed True


DATASET_NAME=chic
# python quantile_forecast.py  data/${DATASET_NAME}.gluonts  ./models/${DATASET_NAME}_deepar --deepar True
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_deepar ${DATASET_NAME}_deepar ./data/examples/${DATASET_NAME}

OMP_NUM_THREADS=3 nohup python -u quantile_forecast.py  data/data_to_tune/${DATASET_NAME}/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn --config_path ./configs/test/config_chic_15_nonmonot.json --dataset_outlier_path data/chicago_outlier.gluonts $ARGS > chicago15min_nonmono.out &
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn ${DATASET_NAME}_mqrnn ./data/examples/${DATASET_NAME} 
OMP_NUM_THREADS=3 nohup python -u quantile_forecast.py  data/data_to_tune/${DATASET_NAME}/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn_mono --config_path ./configs/test/config_chic_15_monot.json --dataset_outlier_path data/chicago_outlier.gluonts $ARGS --monotonic True > chicago15min_mono.out &
# python make_examples.py data/${DATASET_NAME}.gluonts ./models/${DATASET_NAME}_mqrnn_mono ${DATASET_NAME}_mqrnn_mono ./data/examples/${DATASET_NAME} 
