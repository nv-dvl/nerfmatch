
## NeRF Novel View Synthesize
The following code allows one to check the rendered RGB and depth images for a pre-trained NeRF model. It caches the rendered images to specified `--cache_dir`. In the example, `#scene` is used as a wildcard for scene name. Depending on the sequence length, it might take quite some time to render all frames. One can add  `--debug` option, for quick debugging/checking.

```
# Cambridge
python -m model_eval.eval_nerf  --split 'test'  --img_wh 480 480 --dataset 'cambridge' \
    --ckpt 'pretrained/nerf/cambridge/mip_app/#scene_last.ckpt' --save_depth \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --cache_dir 'outputs/nerf_rendered/cambridge/mip_app/#scene_last_15ep'
```
```
# 7Scenes
python -m model_eval.eval_nerf  --split 'test'  --img_wh 480 480 --dataset '7scenes' \
    --ckpt 'pretrained/nerf/7scenes/sfm/mip/#scene_last.ckpt' --save_depth \
    --scene_anno_path 'data/annotations/7scenes_jsons/sfm/transforms_#scene_#split.json' \
    --cache_dir 'outputs/nerf_rendered/7scenes/sfm/mip/#scene_last_15ep'
```

## NeRF Feature Caching
NeRFMatch matches image features and NeRF features.
For NeRFMatch training we require to pre-cache those NeRF features. 
For NeRFMatch evaluation, currently, our code supports evaluation with NeRF feature extraction on-the-fly for settings with single image retrieval pair, i.e., `--pair_topk 1`. For case of multiple retrieval pairs, i.e., `--pair_topk N` with N > 1, you have to cache the NeRF features offline for evaluation as well. 
Therefore, in general, we recommend to pre-cache NeRF features offline, which can be done via the following command. 
This command caches all pre-defined scenes per dataset specified by `--dataset`, where `#scene` is wildcard for scene names.

```
# Cambridge
python -m model_eval.eval_nerf --cache_scene_pts --split 'train_test' \
    --downsample 8 --img_wh 480 480 --stop_layer 3 \
    --ckpt 'pretrained/nerf/cambridge/mip_app/#scene_last.ckpt' \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --cache_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep' \
    --dataset 'cambridge'
```
```
# 7Scenes
python -m model_eval.eval_nerf --cache_scene_pts --split 'train_test' \
    --downsample 8 --img_wh 480 480 --stop_layer 3 \
    --ckpt 'pretrained/nerf/7scenes/sfm/mip/#scene_last.ckpt' \
    --scene_anno_path 'data/annotations/7scenes_jsons/sfm/transforms_#scene_#split.json' \
    --cache_dir 'outputs/scene_dirs/7scenes/sfm/inter_layer3/#scene/mip/last_15ep' \
    --dataset '7scenes'
```


## Reproduce NeRFMatch Evaluation
We assumes NeRF feature have been caches following the above section. Use `--scene_dir` to specify the feature cache location. We provide the following commands to show how to evaluate our models and  reproduce our paper results. Notice, the numbers can be slightly different due to system-level environment difference. 
One can add  `--debug` option, for quick debugging/checking. And use `--ow_cache` to overwrite the existing result caches.

### Cambridge Landmarks
**NeRFMatch-Mini model without pose refinement**
```
python -m model_eval.benchmark_nerfmatch  --rthres 10 --mutual  \
    --ckpt_dir 'pretrained/nerfmatch/cambridge/coarse_mini' \
    --scene_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin' \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --cache_tag 'eccv_repr'  --feats 'layer3' 
```

For the case `--pair_topk 1`, you can omitt this option, and specify `--no_cache_pt` and `--nerf_path` such that NeRF model will render those feature during runtime. 
```
python -m model_eval.benchmark_nerfmatch  --rthres 10  --mutual \
    --ckpt_dir 'pretrained/nerfmatch/cambridge/coarse_mini' \
    --nerf_path 'pretrained/nerf/cambridge/mip_app/#scene_last.ckpt' \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --cache_tag 'eccv_repr'  --feats 'layer3' --no_cache_pt
```

**NeRFMatch-Mini with optimization-based pose refinement**

In this case, you need to specify `--nerf_path` to let it iteratively render novel views at the current pose estimation.
```
python -m model_eval.benchmark_nerfmatch  --rthres 10  --mutual \
    --inerf --inerf_optim 2 --inerf_lr 0.001 --inerf_lrd --iters 2\
    --ckpt_dir 'pretrained/nerfmatch/cambridge/coarse_mini' \
    --nerf_path 'pretrained/nerf/cambridge/mip_app/#scene_last.ckpt'  \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --scene_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin' \
    --cache_tag 'eccv_repr'  --feats 'layer3' 

```

**NeRFMatch without pose refinement**

Since NeRFMatch is more accurate than NeRF Mini, we lower the ransac threshold to `--rthres 5`.
```
python -m model_eval.benchmark_nerfmatch  --rthres 5 --solver 'colmap' --mutual \
    --ckpt_dir 'pretrained/nerfmatch/cambridge/c2f_full'  \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --scene_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin' \
    --cache_tag 'eccv_repr' --feats 'layer3' 
```

**NeRFMatch with iterative pose refinement and top10 retrieval pairs**

We provide two top-10 retreival pairs in _data/pairs_.

- _data/pairs/cambridge/#scene/pairs-query-netvlad10.txt_: the netvlad image retrieval pairs computed on real images.
- _data/pairs/cambridge/#scene/pairs-query-netvlad10-train_synth480.txt_: the netvlad image retrieval pairs computed on synthesized reference images. In this case, we do not need to keep original images as a part of the scene representation, but rather use NeRF as the only scene representation.

One can specify which pairs to use for testing with `--test_pair_txt`.
```
python -m model_eval.benchmark_nerfmatch  --rthres 5 --solver 'colmap' --mutual --iters 2 --pair_topk 10 \
    --ckpt_dir 'pretrained/nerfmatch/cambridge/c2f_full'  \
    --nerf_path 'pretrained/nerf/cambridge/mip_app/#scene_last.ckpt'  \
    --scene_anno_path 'data/annotations/cambridge_jsons/transforms_#scene_#split.json' \
    --scene_dir 'outputs/scene_dirs/cambridge/inter_layer3/#scene/mip_app/last_15ep/ds8lin' \
    --test_pair_txt 'data/pairs/cambridge/#scene/pairs-query-netvlad10-train_synth480.txt' \
    --cache_tag 'eccv_repr' --feats 'layer3' 

```

### 7Scenes
For 7Scenes, since there are lots of pairs, we use iterative pose refinement for both NeRFMatch-Mini and NeRFMatch models, which is faster than the optimization based version. It also does not benefit much from multiple retrieval pairs
since the scene is rather small, we always use `--pair_topk 1` here.  If you want to evaluate each scene in parallel for faster evaluation, you can specify the scene with `--scene`.

**NeRFMatch-Mini**
```
python -m model_eval.benchmark_nerfmatch  --rthres 10  --mutual --iters 2 \
    --ckpt_dir 'pretrained/nerfmatch/7scenes/coarse_mini' \
    --nerf_path 'pretrained/nerf/7scenes/sfm/mip/#scene_last.ckpt'  \
    --scene_anno_path 'data/annotations/7scenes_jsons/sfm/transforms_#scene_#split.json' \
    --scene_dir 'outputs/scene_dirs/7scenes/sfm/inter_layer3/#scene/mip/last_15ep/ds8lin' \
    --cache_tag 'eccv_repr'  --feats 'layer3' --scene 'heads'
```
**NeRFMatch**

Similarly, we provide two types of retrival pairs:
- _data/pairs/7scenes/#scene/pairs-query-densevlad10.txt_: the densevlad image retrieval pairs computed on real images.
- _data/pairs/7scenes/#scene/pairs-query-netvlad10-train_synth480.txt_: the netvlad image retrieval pairs computed on synthesized reference images. 

```
python -m model_eval.benchmark_nerfmatch  --rthres 5 --solver 'colmap' --mutual --iters 2 \
    --ckpt_dir 'pretrained/nerfmatch/7scenes/c2f_full' \
    --nerf_path 'pretrained/nerf/7scenes/sfm/mip/#scene_last.ckpt'  \
    --scene_anno_path 'data/annotations/7scenes_jsons/sfm/transforms_#scene_#split.json' \
    --scene_dir 'outputs/scene_dirs/7scenes/sfm/inter_layer3/#scene/mip/last_15ep/ds8lin' \
    --test_pair_txt 'data/pairs/7scenes/#scene/pairs-query-netvlad10-train_synth480.txt' \
    --cache_tag 'code_clean' --feats 'layer3' --scene 'heads'

```
