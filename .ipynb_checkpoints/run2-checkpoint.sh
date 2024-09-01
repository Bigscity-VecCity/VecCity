
for dataset in 'nyc' 'chicago' 'foursquare_tky' 'singapore' 'porto' 'sanfransico'
do
    for model in 'Tale' 'SkipGram' 
    do
      python run_model.py --task poi_representation --model $model --dataset $dataset  --gpu_id 2 --config config --exp_id $model --train false
    done
done