#nohup sh run_road_test.sh model xa 0 > log/road_datasize/model.log 2>&1 &
python run_model.py --task road_representation --model $1 --dataset xa  --gpu_id $2 --config road_config_128_Test --exp_id 15 --train false --choice 10
# python run_model.py --task road_representation --model $1 --dataset xa  --gpu_id $2 --config road_config_128_Test --exp_id 15 --train false --choice 20
# python run_model.py --task road_representation --model $1 --dataset xa  --gpu_id $2 --config road_config_128_Test --exp_id 15 --train false --choice 30
# python run_model.py --task road_representation --model $1 --dataset xa  --gpu_id $2 --config road_config_128_Test --exp_id 15 --train false --choice 40
# python run_model.py --task road_representation --model $1 --dataset xa  --gpu_id $2 --config road_config_128_Test --exp_id 15 --train false --choice 50