export REGION=us-central1
export BUCKET=keras-imdb-wiki
export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET/$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME  \
--job-dir $JOB_DIR \
--module-name trainer.model \
--package-path ./trainer/ \
--region $REGION \
--\
 --train-file gs://$BUCKET/age/train/