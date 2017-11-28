export REGION=us-central1
export BUCKET=keras-imdb-wiki
export JOB_NAME="Keras_imbd_wiki$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET/$JOB_NAME 
op=$1
# Choose option
if [ "$op" = "local" ]; then
		echo "$op"
		python2 trainer/model.py \
		--job-dir ./output/adience_output_CELEB\
		--train-file /home/estilon/datasets/Adience/train_imagenet_adience \
        --valid-file /home/estilon/datasets/valid_output\
		--job-type 3 \
        --batch-size 32 \
        --epochs 20 \
        --model-file ~/models/bestModel \
        --predict-dir /home/estilon/datasets/valid_output\

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
