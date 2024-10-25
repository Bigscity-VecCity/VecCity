<div align="center">
    <img src="./figures/VecCity.png" width="440px">
    <p> 
    	<b>
        A Taxonomy-Based Library for Electronic Map Entity Representation Learning <a href="https://arxiv.org/pdf/2306.11443.pdf" title="PDF">PDF</a>
        </b>
    </p>

------

<p align="center">
  <a href="#1-overview">Overview</a> •
  <a href="#2-quick-starting">Quick Starting</a> •
  <a href="#3-dataset-illstration">Dataset Illstration</a> •
  <a href="#4-how-to-run">How to Run</a> •
  <a href="#5-directory-structure">Directory Structure</a> •
  <a href="#6-citation">Citation</a> 
</p>
</div>

Official repository of under review paper ["*VecCity*: A Taxonomy-Based Library for Electronic Map Entity Representation Learning"](https://arxiv.org/pdf/2306.11443.pdf). Please star, watch and fork our repo for the active updates!

## 1. Overview
<div align="center">
  <img src="./figures/pipeline.jpg" width="500">
</div>

*VecCity* is an open-sourced and standardized benchmark compatible with various datasets and baseline models. The above figure illustrates the arcitecture of VecCity. 
For a given dataset, we first construct atomic files from multi-sourced city data by extracting and map entities (e.g., POIs, road segments, land parcels, etc.) and auxiliary data into corresponding atomic files. 
MapRL models encodes various entities in a unified configuration, which facilitates joint processing for various downstream tasks.
## 2. Quick Starting
### Step 1: Create a python 3.9 environment and install dependencies:

```
conda create -n VecCity python=3.9
source activate VecCity
```

### Step 2: Install library:

```bash
pip install -r ./requirements.txt
```

### Step 3: Download processed data:

You can also follow the instraction to [download processed city data](xxxxx) or [process your own dataset](xxxxxx).

### Step 4: Run a training pipeline for MapRL models:

```
python run_model.py --task poi --dataset nyc --model CTLE --exp_uid CTLE 
```

## 3. Dataset Illstration
We opensource nine city datasets in New York, Chicago, Tokyo, San Francisco, Porto, Beijing, Chendu, and Xi'an compatible with atomic files. 
As the original dataset is quite large, we have included example data, data processing code, and model code to assist researchers in understanding our work. 
The complete data sources can be found on [Beihang Pan](xxxxxx).

The above dataset construction scheme is highly reusable, one can prepare their own city data and use our code to build their personalized MapRL dataset easily. 
We will provide a detailed explanation for our data and pre-processing module in the following. 

#### 3.1 City Data

| City | #POI  | #Segment | #Parcel | #CIT | #CDT  | #OD Flow |
| :-- | --: | --: | --: | --: | --: | --: |
|NY|10,283|90,781|262|309,523|-|14,385,456|
|CHI|3,659|47,701|77|444,581|-|759,788|
|TYO|7,863|407,905|64|107,131|226,782|-|
|SIN|9,816|35,084|332|477,133|11,170|-|
|PRT|1,228|11,095|382|11,272|695,085|4014|
|SF|1,617|27,274|194|37,336|500,516|23,344|
|BJ|81,181|40,306|11,208|-|1,018,312|575,660|
|CD|17,301|6,195|1,306|-|559,729|111,642|
|XA|19,108|5,269|1,056|-|384,618|78,907|


We store the original unprocessed files in the [Beihang Pan](xxxx). To preprocess, align, and filter these files, we utilize the scripts provided by [VecCity-Dataset].  After proprocessing, the original dataset will be storaged in atomic files.
Our city dataset construction scheme is highly reusable. You can prepare your own data following either the file format in *atomic files*. This flexibility allows you to adapt the construction process to various cities and datasets easily.


#### 3.2 Atomic files

The following types of atomic files are defined:

| filename    | content                                  | example                                  |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| xxx.geo     | Each line in the file represents a map entity | geo_uid, geo_type, geo_location, geo_features|
| xxx.grel     | Each line of the file records a non-zero geographic relations between two map entity. | rel_uid, ori_geo_uid, des_geo_uid, wight, feature, timestamp|
| xxx.srel     | Store the relationship information between entities, such as areas. | rel_uid, type, origin_uid, destination_uid  |
| xxx.citraj    |  Each line of the file corresponds to a sample of a check-in trajectory. | traj_uid, sample_uid, entity_uid, timestamp |
| xxx.cdtraj    | Each line of the file corresponds to a sample of a coordinate trajectory. | traj_uid, sample_uid, entity_uid, timestamp |
| config.json | Used to store the config settings for data precessing. |                                          |

we explain the above four atomic files as follows:

**xxx.geo**: An element in the entity table consists of the following four parts:

**geo_uid, geo_type, geo_location, geo_features(multiple columns).**

```
geo_uid: This field records the unique ID for map entities.
geo_type: This field recodes the entity's type, of which the value can be ``point'', ``polyline'' or ``polygon'', corresponding to POIs, road segments, and land parcels.
geo_location: This field stores the geographical coordinates of a map entity: a single longitude and latitude for point entities, a sequence of coordinates for polylines, and a closed sequence of coordinates for polygons.
geo_features (optional): This field stores the features of a map entity, with the format and length of this field varying based on the data type and number of features.
```

**xxx.grel or xxx.srel**: An element in the ralationship table consists of the following six parts:

**rel_uid, ori_geo_uid, des_geo_uid, wight, feature(multiple columns), timestamp.**

```
rel_uid: This field records the unique ID for a relationship between two map entities.
ori_geo_uid & des_geo_uid: These two fields indicate the uIDs of the origin and destination map entities of a relation, with values matching the geo_uid listed in the *.geo* file.
wight: This field recodes the weight of the edge corresponding to a map entity relation.
feature (optional): This field stores additional features for a relation network edge. In some \rl models, relation networks are modeled as heterogeneous graph. This field can be used to store the type of heterogeneous edges. 
timestamp (optional): This field stores timestamps for a relation and is essential for dynamic graphs, where edge weights and features correspond to the edge state during a specific time period.
```

**xxx.citraj or xxx.cdtraj**: An element in the trajectory table consists of the following four parts:

**traj_uid, sample_uid, entity_uid, timestamp**.

```
traj_uid: The field specifies the unique ID of a trajectory that samples belong to.
sample_uid: The field records the order index of a sample within a trajectory.
entity_id: The field indicates the map entity to which a sample corresponds, with its value matching the *geo_uid* of the map entity listed in the *.geo* file.
timestamp: The field stores a trajectory sample's timestamp, which is in the format of Unix timestamps.
```

**xxx.config**: The config file is used to supplement the settings for data preprocessing.


## 4. How to Run

#### 4.1 Train a MapRL Model

To train and evaluate a MapRL model, use the run_model.py script:

```bash
python ./run_model.py 
  -h, --help            show this help message and exit
  --task TASK           the name of task [poi,segment,parcel]
  --model MODEL         the name of model
  --dataset DATASET     the name of dataset [nyc, tokyo, chicago, sf, porto, bj, cd, xa]
  --config_file CONFIG_FILE
                        the file name of config file
  --saved_model SAVED_MODEL
                        whether save the trained model
  --train TRAIN         whether re-train model if the model is trained before
  --exp_id EXP_ID       id of experiment
  --seed SEED           random seed
  --save_result SAVE_RESULT
                        save result or not
  --gpu GPU             whether use gpu
  --gpu_id GPU_ID       the gpu id to use
  --batch_size BATCH_SIZE
                        the batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --max_epoch MAX_EPOCH
                        the maximum epoch
  --executor EXECUTOR   the executor class name
  --evaluator EVALUATOR
                        the evaluator class name

```
**How to get the embedding?**

We storage the embedding file in **./veccity/cache/[exp_id]/evaluate_cache/[map entity]_embedding.csv**.

#### 4.2 Evaluate your map entity embedding with VecCity

To train and evaluate a MapRL model on downstream tasks, use the run_model.py script:

```bash
python run_model.py --task poi --model CTLE --dataset nyc --exp_id [exp_id] --train false
```
This script will run the CTLE model on the nyc dataset for downstream tasks of poi entity under the default configuration.

## 5 Directory Structure

The expected structure of files is:
```
VecCity
 |-- VecCity  # UrbanKG_data
 |    |-- cache
 |    |    |-- dataset_cache  # 中间文件
 |    |    |    |-- bj   
 |    |    |    |-- cd     
 |    |    |    |-- ...     
 |    |    |-- 98186  # indexed by exp_id
 |    |    |    |-- evaluate_cache # dir for evaluation result and entity embeddings    
 |    |    |    |-- model_cache # dir for model parameter        
 |    |-- config  # 
 |    |    |-- data # dir for data module configs
 |    |    |-- evaluator # dir for downstream & evaluation configs
 |    |    |-- executor # dir for pipeline configs
 |    |    |-- model # dir for model module configs
 |    |    |-- config_parser.py
 |    |    |-- task_config.json # management file for config
 |    |-- data  # dir for data module where stores the scrips for data loading
 |    |-- downstream  # dir for downstream module where stores the scripts for downstream models
 |    |-- executor  # dir for exector where stores the scripts of training process
 |    |-- pipeline  # dir for pipeline where stores the pipeline scripts
 |    |-- upstream  # dir for upstream module where stores the scripts for upstream models
 |    |-- utils  # dir for utils where stores the scripts of useful tools
 |-- raw_data
 |    |-- porto
 |    |    |-- porto.geo
 |    |    |-- porto.grel
 |    |    |-- porto.srel
 |    |    |-- porto.citraj
 |    |    |-- porto.cdtraj
 |    |    |-- config.json
 |    |-- ...
 |-- log  # dir for log
 |-- test
 |-- script # some examples for runing
 |-- run_model.py 
 |-- README.md
```

## 6 Citation
If you find our work is useful for your research, please consider citing:
```bash
@article{zhang2024vec,
  title={VecCity: A Taxonomy-Based Library for Electronic Map Entity Representation Learning},
  author={Zhang, Wentao and Wang, jingyuan and U, Leonhou},
  journal={arxiv},
  year={2024}
}
```
