# nohup sh script/run_road_abl0.sh > log/run_road_abl0.log 2>&1 &
python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 2 --config road_config_128 --exp_id JCLRNTRI --abl recon\&infer --train false
python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 2 --config road_config_128 --exp_id JCLRNTR --abl recon --train false
python run_model.py --task road_representation --model JCLRNT --dataset xa  --gpu_id 2 --config road_config_128 --exp_id JCLRNTI --abl infer --train false