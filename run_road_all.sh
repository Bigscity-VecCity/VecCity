#nohup sh run_road_all.sh JCLRNT 0 >run_poi_all.log 2>&1 &
for dataset in 'xa' 'cd' 'bj' 'porto'
do
    python run_model.py --task road_representation --model HRNR --dataset $dataset --gpu_id 3 --config road_config_128 --exp_id hrnr --train false
done

python run_model.py --task road_representation --model HRNR --dataset sanfransico --gpu_id 3 --config road_config_128 --exp_id hrnr


