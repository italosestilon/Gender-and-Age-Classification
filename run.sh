export REGION=us-central1
export BUCKET=keras-imdb-wiki
export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET/$JOB_NAME 
op=$1
# Choose option
if [ "$op" = "local" ]; then
		echo "$op"
		gcloud ml-engine local train \
		--job-dir ./output/teste \
		--module-name trainer.model \
		--package-path ./trainer/ \
		--\
		--train-file /data/vgg_predict \
        --valid-file /data/valid_imagenet \
		--job-type 1 \
		--predict-dir /data/validation_predict \
        --batch-size 128 \
        --model-file output/teste\
        --epochs 200 

	else 
		echo "$op"
		gcloud ml-engine jobs submit training $JOB_NAME  \
		--job-dir $JOB_DIR \
		--module-name trainer.model \
		--package-path ./trainer/ \
		--region $REGION \
		--\
		--train-file gs://$BUCKET/NP_CELEB/train \
		--job-type 0 \
		--predict-dir gs://$BUCKET/vgg_predict_7 \		
        --batch-size 4

fi
