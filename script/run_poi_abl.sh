# nohup sh script/run_poi_abl.sh > log/run_poi_abl.log 2>&1 &
# python run_model.py --task poi_representation --model CTLE --dataset nyc  --gpu_id 0 --config poi_config_128 --exp_id CTLEM --abl mh
python run_model.py --task poi_representation --model CTLE --dataset nyc  --gpu_id 0 --config poi_config_128 --exp_id CTLEL --abl lstm --train false 
# python run_model.py --task poi_representation --model CTLE --dataset nyc  --gpu_id 0 --config poi_config_128 --exp_id CTLEC --abl contr
# python run_model.py --task poi_representation --model CACSR --dataset nyc  --gpu_id 0 --config poi_config_128 --exp_id CTLET --abl te