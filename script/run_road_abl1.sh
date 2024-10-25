# nohup sh script/run_road_abl1.sh > log/run_road_abl1.log 2>&1 &
# python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 3 --config road_config_128 --exp_id JCLRNT --train false
python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 3 --config road_config_128 --exp_id JCLRNTG --abl gnn --train false
python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 3 --config road_config_128 --exp_id JCLRNTL --abl lstm --train false