export REGION=us-central1
export BUCKET=keras-imdb-wiki
export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET/$JOB_NAME

gcloud ml-engine local train \
--job-dir output \
--module-name trainer.model \
--package-path ./trainer/ \
--\
 --train-file dataset