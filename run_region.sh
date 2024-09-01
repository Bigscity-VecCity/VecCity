# nohup sh run_region.sh > log/run_HREP_abl.log 2>&1 &
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPR --abl recon
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPC --abl contr
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPI --abl infer
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPRI--abl recon\&infer
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPCI --abl contr\&infer
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id HREPG --abl gcn
# python run_model.py --task region_representation --model HREP --dataset xa  --gpu_id 3 --config dim128_region --exp_id 16 --choice 10 --train false
for model in ReMVC MGFN MVURE HDGE ZEMob
do
    python run_model.py --task region_representation --model $model --dataset xa  --gpu_id $1 --config dim128_region --exp_id 16 --choice 10 --train false
done
