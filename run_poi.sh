# nohup sh run_poi.sh CTLE foursquare_tky > log/poi_datasize/CTLE.log 2>&1 &
python run_model.py --task poi_representation --model $1 --dataset foursquare_tky --gpu_id $2 --config poi_config_128 --exp_id $1 --train false --choice 10
# python run_model.py --task poi_representation --model $1 --dataset foursquare_tky --gpu_id $2 --config poi_config_128 --exp_id $1 --train false --choice 20
# python run_model.py --task poi_representation --model $1 --dataset foursquare_tky --gpu_id $2 --config poi_config_128 --exp_id $1 --train false --choice 30
# python run_model.py --task poi_representation --model $1 --dataset foursquare_tky --gpu_id $2 --config poi_config_128 --exp_id $1 --train false --choice 40
# python run_model.py --task poi_representation --model $1 --dataset foursquare_tky --gpu_id $2 --config poi_config_128 --exp_id $1 --train false --choice 50



