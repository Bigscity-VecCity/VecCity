# nohup sh script/run_road_ntj.sh > log/run_road_ntj.log 2>&1 &
for model in 'SRN2Vec' 'HyperRoad'
do
    for dataset in 'singapore' 
    do 
        python run_model.py --task road_representation --model $model --dataset $dataset --gpu_id 0 --config road_config_128 --exp_id 15 --train false
    done
done

for model in 'SRN2Vec' 'HyperRoad' 'HRNR'
do
    for dataset in 'sanfransico' 
    do 
        python run_model.py --task road_representation --model $model --dataset $dataset --gpu_id 0 --config road_config_128 --exp_id $model --train false
    done
done

for model in 'HRNR'
do
    for dataset in 'singapore' 
    do 
        python run_model.py --task road_representation --model $model --dataset $dataset --gpu_id 0 --config road_config_128 --exp_id $model --train false
    done
done