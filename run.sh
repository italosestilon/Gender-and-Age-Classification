export REGION=us-central1
export BUCKET=keras-imdb-wiki
export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET/$JOB_NAME 
op=$1
# Choose option
if [ "$op" = "local" ]; then
		echo "$op"
		python2 trainer/model.py \
		--job-dir /home/phillipe/models/age_models\
		--train-file /data/Adience/resize/age/test\
        --valid-file /data/Adience/age/resize/valid\
		--job-type 0 \
        --batch-size 32 \
        --epochs 20 \
        --model-file ~/models/bestModel \
        --predict-dir /data/Adience/after_imagenet/test 

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
