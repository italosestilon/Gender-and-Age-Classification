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
 --train-file gs://$BUCKET/NP_CELEB/train \
 --job-type 0 \
 --predict-dir gs://$BUCKET/vgg_predict_6

#gcloud ml-engine local train \
#--job-dir output \
#--module-name trainer.model \
#--package-path ./trainer/ \
#--\
# --train-file ../NP_CELEB/train \
# --job-type 0 \
# --predict-dir vgg_predict
