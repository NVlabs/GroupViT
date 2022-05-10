# GroupViT: Semantic Segmentation Emerges from Text Supervision

GroupViT is a framework for learning semantic segmentation purely from text captions without
using any mask supervision. It learns to perform bottom-up heirarchical spatial grouping of 
semantically-related visual regions. This repository is the official implementation of GroupViT 
introduced in the paper:

[**GroupViT: Semantic Segmentation Emerges from Text Supervision**](https://arxiv.org/abs/2202.11094),
[*Jiarui Xu*](https://jerryxu.net),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-gupta),
[*Sifei Liu*](https://research.nvidia.com/person/sifei-liu),
[*Wonmin Byeon*](https://wonmin-byeon.github.io/),
[*Thomas Breuel*](http://www.tmbdev.net/),
[*Jan Kautz*](https://research.nvidia.com/person/jan-kautz),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
CVPR 2022.
<div align="center">
<img src="figs/github_arch.gif" width="85%">

</div>

## Visual Results

<div align="center">
<img src="figs/github_voc.gif" width="32%">
<img src="figs/github_ctx.gif" width="32%">
<img src="figs/github_coco.gif" width="32%">
</div>

## Links
* [Jiarui Xu's Project Page](https://jerryxu.net/GroupViT/) (with additonal visual results)
* [arXiv Page](https://arxiv.org/abs/2202.11094)


</div>

## Citation

If you find our work useful in your research, please cite:

```latex
@article{xu2022groupvit,
  author    = {Xu, Jiarui and De Mello, Shalini and Liu, Sifei and Byeon, Wonmin and Breuel, Thomas and Kautz, Jan and Wang, Xiaolong},
  title     = {GroupViT: Semantic Segmentation Emerges from Text Supervision},
  journal   = {arXiv preprint arXiv:2202.11094},
  year      = {2022},
}
```

## Environmental Setup

* Python 3.7
* PyTorch 1.8
* webdataset 0.1.103
* mmsegmentation 0.18.0
* timm 0.4.12

Instructions:

```shell
conda create -n groupvit python=3.7 -y
conda activate groupvit
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install mmsegmentation==0.18.0
pip install webdataset==0.1.103
pip install timm==0.4.12
git clone https://github.com/NVIDIA/apex
cd && apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install opencv-python==4.4.0.46 termcolor==1.1.0 diffdist einops omegaconf
pip install nltk ftfy regex tqdm
```

## Demo

* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/xvjiarui/GroupViT)

* Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mwtz6ojiThWWdRrpAZTLlLs6w3T9Fr6x)

* To run the demo from the command line:

```shell
python demo/demo_seg.py --cfg configs/group_vit_gcc_yfcc_30e.yml --resume /path/to/checkpoint --vis input_pred_label final_group --input demo/examples/voc.jpg --output_dir demo/output
```
  The output is saved in `demo/output/`.

## Benchmark Results

<table>
<thead>
  <tr>
    <th></th>
    <th>Zero-shot Classification</th>
    <th colspan="3">Zero-shot Segmentation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>config</td>
    <td>ImageNet</td>
    <td>Pascal VOC</td>
    <td>Pascal Context</td>
    <td>COCO</td>
  </tr>
  <tr>
    <td>GCC + YFCC (<a href="configs/group_vit_gcc_yfcc_30e.yml">cfg</a>)</td>
    <td>43.7</td>
    <td>52.3</td>
    <td>22.4</td>
    <td>24.3</td>
  </tr>
  <tr>
    <td>GCC + RedCaps (<a href="configs/group_vit_gcc_redcap_30e.yml">cfg</a>)</td>
    <td>51.6</td>
    <td>50.8</td>
    <td>23.7</td>
    <td>27.5</td>
  </tr>
</tbody>
</table>

Pre-trained weights `group_vit_gcc_yfcc_30e-879422e0.pth` and `group_vit_gcc_redcap_30e-3dd09a76.pth` for these models are provided by Jiarui Xu [here](https://github.com/xvjiarui/GroupViT#benchmark-results). 

## Data Preparation

During training, we use [webdataset](https://webdataset.github.io/webdataset/) for scalable data loading.
To convert image text pairs into the webdataset format, we use the [img2dataset](https://github.com/rom1504/img2dataset) tool to download and preprocess the dataset.

For inference, we use [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for semantic segmentation testing, evaluation and visualization on Pascal VOC, Pascal Context and COCO datasets.

The overall file structure is as follows:

```shell
GroupViT
├── local_data
│   ├── gcc3m_shards
│   │   ├── gcc-train-000000.tar
│   │   ├── ...
│   │   ├── gcc-train-000436.tar
│   ├── gcc12m_shards
│   │   ├── gcc-conceptual-12m-000000.tar
│   │   ├── ...
│   │   ├── gcc-conceptual-12m-001943.tar
│   ├── yfcc14m_shards
│   │   ├── yfcc14m-000000.tar
│   │   ├── ...
│   │   ├── yfcc14m-001888.tar
│   ├── redcap12m_shards
│   │   ├── redcap12m-000000.tar
│   │   ├── ...
│   │   ├── redcap12m-001211.tar
│   ├── imagenet_shards
│   │   ├── imagenet-val-000000.tar
│   │   ├── ...
│   │   ├── imagenet-val-000049.tar
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── coco
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```

The instructions for preparing each dataset are as follows.

### GCC3M

Please download the training split annotation file from [Conceptual Caption 12M](https://ai.google.com/research/ConceptualCaptions/download) and name it as `gcc3m.tsv`.

Then run `img2dataset` to download the image text pairs and save them in the webdataset format.
```
sed -i '1s/^/caption\turl\n/' gcc3m.tsv
img2dataset --url_list gcc3m.tsv --input_format "tsv" \
            --url_col "url" --caption_col "caption" --output_format webdataset\
            --output_folder local_data/gcc3m_shards
            --processes_count 16 --thread_count 64
            --image_size 512 --resize_mode keep_ratio --resize_only_if_bigger True \
            --enable_wandb True --save_metadata False --oom_shard_count 6
rename -d 's/^/gcc-train-/' local_data/gcc3m_shards/*
```
Please refer to [img2dataset CC3M tutorial](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) for more details.

### GCC12M

Please download the annotation file from [Conceptual Caption 12M](https://github.com/google-research-datasets/conceptual-12m) and name it as `gcc12m.tsv`.

Then run `img2dataset` to download the image text pairs and save them in the webdataset format.
```
sed -i '1s/^/caption\turl\n/' gcc12m.tsv
img2dataset --url_list gcc12m.tsv --input_format "tsv" \
            --url_col "url" --caption_col "caption" --output_format webdataset\
            --output_folder local_data/gcc12m_shards \
            --processes_count 16 --thread_count 64
            --image_size 512 --resize_mode keep_ratio --resize_only_if_bigger True \
            --enable_wandb True --save_metadata False --oom_shard_count 6
rename -d 's/^/gcc-conceptual-12m-/' local_data/gcc12m_shards/*
```
Please refer to [img2dataset CC12M tutorial](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md) for more details.

### YFCC14M
Please follow the [CLIP Data Preparation](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md) instructions to download the YFCC14M subset.
```
wget https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2
bunzip2 yfcc100m_subset_data.tsv.bz2
```

Then run the preprocessing script to create the subset sql db and annotation tsv files. This may take a while.
```
python convert_dataset/create_subset.py --input-dir . --output-dir . --subset yfcc100m_subset_data.tsv
```
This script will create two files: an SQLite db called `yfcc100m_dataset.sql` and an annotation tsv file called `yfcc14m_dataset.tsv`.

Then follow the [YFCC100M Download Instruction](https://gitlab.com/jfolz/yfcc100m/-/tree/master) to download the dataset and its metadata file.
```
pip install git+https://gitlab.com/jfolz/yfcc100m.git
mkdir -p yfcc100m_meta
python -m yfcc100m.convert_metadata . -o yfcc100m_meta --skip_verification
mkdir -p yfcc100m_zip
python -m yfcc100m.download yfcc100m_meta -o yfcc100m_zip
```

Finally convert the dataset into the webdataset format.
```
python convert_dataset/convert_yfcc14m.py --root yfcc100m_zip --info yfcc14m_dataset.tsv --shards yfcc14m_shards
```

### RedCaps12M

Please download the annotation file from [RedCaps](https://redcaps.xyz/).
```
wget https://www.dropbox.com/s/cqtdpsl4hewlli1/redcaps_v1.0_annotations.zip?dl=1
unzip redcaps_v1.0_annotations.zip
```

Then run the preprocessing script and `img2dataset` to download the image text pairs and save them in the webdataset format.
```
python convert_dataset/process_redcaps.py annotations redcaps12m_meta/redcaps12m.parquet --num-split 16
img2dataset --url_list ~/data/redcaps12m/ --input_format "parquet" \
            --url_col "URL" --caption_col "TEXT" --output_format webdataset \
            --output_folder local_data/recaps12m_shards
            --processes_count 16 --thread_count 64
            --image_size 512 --resize_mode keep_ratio --resize_only_if_bigger True \
            --enable_wandb True --save_metadata False --oom_shard_count 6
rename -d 's/^/redcap12m-/' local_data/recaps12m_shards/*
```

### ImageNet

Please follow the [webdataset ImageNet Example](https://github.com/tmbdev-archive/webdataset-examples/blob/master/makeshards.py) to convert ImageNet into the webdataset format.

### Pascal VOC

Please follow the [MMSegmentation Pascal VOC Preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc) instructions to download and setup the Pascal VOC dataset.

### Pascal Context

Please refer to the [MMSegmentation Pascal Context Preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context) instructions to download and setup the Pascal Context dataset.

### COCO

[COCO dataset](https://cocodataset.org/) is an object detection dataset with instance segmentation annotations.
To evaluate GroupViT, we combine all the instance masks of a catergory together and generate semantic segmentation maps.
To generate the semantic segmentation maps, please follow [MMSegmentation's documentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k) to download the COCO-Stuff-164k dataset first and then run the following

```shell
python convert_dataset/convert_coco.py local_data/data/coco/ -o local_data/data/coco/
```

## Run Experiments

### Pre-train

Train on a single node:

```shell
(node0)$ ./tools/dist_launch.sh main_group_vit.py /path/to/config $GPUS_PER_NODE
```

For example, to train on a node with 8 GPUs, run:
```shell
(node0)$ ./tools/dist_launch.sh main_group_vit configs/group_vit_gcc_yfcc_30e.yml 8
```

Train on multiple nodes:

```shell
(node0)$ ./tools/dist_mn_launch.sh main_group_vit.py /path/to/config $NODE_RANK $NUM_NODES $GPUS_PER_NODE $MASTER_ADDR
(node1)$ ./tools/dist_mn_launch.sh main_group_vit.py /path/to/config $NODE_RANK $NUM_NODES $GPUS_PER_NODE $MASTER_ADDR
```

For example, to train on two nodes with 8 GPUs each, run:

```shell
(node0)$ ./tools/dist_mn_launch.sh main_group_vit.py configs/group_vit_gcc_yfcc_30e.yml 0 2 8 tcp://node0
(node1)$ ./tools/dist_mn_launch.sh main_group_vit.py configs/group_vit_gcc_yfcc_30e.yml 1 2 8 tcp://node0
```

We used 16 NVIDIA V100 GPUs for pre-training (in 2 days) in our paper.

### Zero-shot Transfer to Image Classification

#### ImageNet

```shell
./tools/dist_launch.sh main_group_vit.py /path/to/config $NUM_GPUS --resume /path/to/checkpoint --eval
```

### Zero-shot Transfer to Semantic Segmentation

#### Pascal VOC

```shell
./tools/dist_launch.sh main_seg.py /path/to/config $NUM_GPUS --resume /path/to/checkpoint
```

#### Pascal Context

```shell
./tools/dist_launch.sh main_seg.py /path/to/config $NUM_GPUS --resume /path/to/checkpoint --opts evaluate.seg.cfg segmentation/configs/_base_/datasets/pascal_context.py
```

#### COCO

```shell
./tools/dist_launch.sh main_seg.py /path/to/config $NUM_GPUS --resume /path/to/checkpoint --opts evaluate.seg.cfg segmentation/configs/_base_/datasets/coco.py
```
