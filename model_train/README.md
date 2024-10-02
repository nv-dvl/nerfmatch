

## NeRF Training (Optional)
Notice, training NeRF models are not the focus of our project. Thus one can directly download [our pretrained NeRF models](). And put them under _outputs/_.

For completeness, we also provide code to train NeRF models. In our experiments, we use 8 32GB GPUs to train each NeRF model for 15 epoches. Here are the code examples for training NeRFs on Cambridge Landmarks and 7 Scenes. Remeber to adapt the numbers of GPUs accordingly. One can add  `--debug` option, for quick debugging.

```
# Cambridge
torchrun --nproc_per_node=8 model_train/train_nerf.py --config configs/nerf/nerf_cambridge_mip_app.yaml --scene 'ShopFacade'

# 7Scenes
torchrun --nproc_per_node=8 model_train/train_nerf.py --config configs/nerf/nerf_7scenes_mip_sfm.yaml --scene 'heads'
```

## Cache NeRF Features
To train NeRFMatch model, our scripts assume NeRF features are pre-cached for each scene. We refer users to [model_eval/README.md](../model_eval/README.md) for details.

## NeRFMatch Training
Our NeRFMatch training loads training config files stored under [configs/nerfmatch](../configs/nerfmatch). 
If you are caching NeRF features to custom directories other than our example directories, remember to adjust the `scene_dir` field according:
```
data:
    scene_dir: $NERF_FEATURE_CACHE_DIR
```
Otherwise, you can also directly overwrite this setting with `--scene_dir $NERF_FEATURE_CACHE_DIR --update_conf ` as shown in our examples below. 
Now you are ready to train NeRFMatch Mini and NeRFMatch Full models with the following commands. One can add  `--debug` option, for quick debugging. We provide training codes per dataset.

### Cambridge Landmarks
```
# NeRFMatch Mini
torchrun --nproc_per_node=8 model_train/train_nerfmatch_coarse.py \
    --config configs/nerfmatch/nerfmatch_cambridge_coarse.yaml \
    --backbone 'convformer384' --no_im_pe --no_pt_pe --temp_type 'mul' \
    --cfeat_dim 256 --coarse_layers 0 --pt_sa 0 --im_sa 0 --clr 0.0008  \
    --max_epochs 30 --cbs 16 --pair_topk 20  --aug_self_pairs 10 \
    --scene_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin' \
    --resume_version 'mip_app_inter3_last'  --update_conf --prefix 'eccv/repr' --scenes 'ShopFacade' 


# NeRFMatch
torchrun --nproc_per_node=8  model_train/train_nerfmatch_c2f.py \
    --config configs/nerfmatch/nerfmatch_cambridge_c2f.yaml \
    --backbone 'convformer384' --temp_type 'mul' --batch_size 2 \
    --max_epochs 50 --clr 0.0004 --cbs 16 --pair_topk 20 --aug_self_pairs 10  \
    --scene_dir  'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin'   \
    --resume_version 'mip_app_inter3_last' --update_conf \
    --prefix 'eccv/repr'   --scenes 'ShopFacade' 
```

### 7Scenes
```
# NeRFMatch Mini
torchrun --nproc_per_node=8 model_train/train_nerfmatch_coarse.py \
    --config configs/nerfmatch/nerfmatch_7scenes_sfm_coarse.yaml \
    --backbone 'convformer384' --no_im_pe --no_pt_pe --temp_type 'mul' \
    --cfeat_dim 256 --coarse_layers 0 --pt_sa 0 --im_sa 0 --clr 0.0008 \
    --max_epochs 30 --cbs 16 --pair_topk 30 --aug_self_pairs 10  \
    --scene_dir 'outputs/scene_dirs/7scenes/sfm/inter_layer3/#scene/mip/last_15ep/ds8lin' \
    --resume_version 'mip_inter3_last' --update_conf \
    --prefix 'eccv/repr' --scenes 'heads'  

# NeRFMatch
torchrun --nproc_per_node=8  model_train/train_nerfmatch_c2f.py \
    --config configs/nerfmatch/nerfmatch_7scenes_sfm_c2f.yaml \
    --backbone 'convformer384' --temp_type 'mul' --batch_size 2 \
    --max_epochs 50 --clr 0.0004 --cbs 16 --pair_topk 30  --aug_self_pairs 10 \
    --scene_dir  'outputs/scene_dirs/7scenes/sfm/inter_layer3/#scene/mip/last_15ep/ds8lin' \
    --resume_version 'mip_inter3_last' --update_conf \
    --prefix 'eccv/repr' --scenes 'heads'
```


torchrun --nproc_per_node=1  model_train/train_nerfmatch_c2f.py \
    --config configs/nerfmatch/nerfmatch_cambridge_c2f.yaml \
    --backbone 'convformer384' --temp_type 'mul' --batch_size 2 \
    --max_epochs 50 --clr 0.0004 --cbs 16 --pair_topk 20 --aug_self_pairs 10  \
    --scene_dir  'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin'   \
    --resume_version 'mip_app_inter3_last' --update_conf \
    --prefix 'eccv/repr'   --scenes 'ShopFacade' --debug

torchrun --nproc_per_node=1 model_train/train_nerf.py --config configs/nerf/nerf_cambridge_mip_app.yaml --scene 'ShopFacade' --debug


