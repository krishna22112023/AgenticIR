# Installation

### Set up the environment

```bash
conda create -n agenticir python=3.10
conda activate agenticir
pip install -r installation/requirements.txt
```

### Deploy IR models

We employ the following models as single-degradation restoration tools: [DiffBIR](https://github.com/XPixelGroup/DiffBIR), [X-Restormer](https://github.com/Andrew0613/X-Restormer), [SwinIR](https://github.com/JingyunLiang/SwinIR), [HAT](https://github.com/XPixelGroup/HAT), [MPRNet](https://github.com/swz30/MPRNet), [MAXIM](https://github.com/google-research/maxim), [Restormer](https://github.com/swz30/Restormer), [DRBNet](https://github.com/lingyanruan/DRBNet), [IFAN](https://github.com/codeslake/IFAN), [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing), [DehazeFormer](https://github.com/IDKiro/DehazeFormer), and [FBCNN](https://github.com/jiaxi-jiang/FBCNN). Thanks for these awesome works.

+ Run `sh installation/deploy_tools.sh` to prepare the code, which is adapted from the official repos linked above.
+ Set up their respective environments according to the official repos.
+ Download the weights. You may need to modify the paths in files in `executor`. A tutorial will be given if necessary.
+ Run `python -m test_tool.test_tool` to check whether all tools work properly.

### Deploy [DepictQA](https://github.com/XPixelGroup/DepictQA)
+ Run `sh installation/deploy_depictqa.sh` to prepare the code, which is adapted from the official repo linked above.
+ Set up the environment according to the official repo.
+ Download the weights.
    + Download the pre-trained ViT from [this link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and put it in `DepictQA/weights/`.
    + Download the pre-trained Vicuna from [this link](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and put it in `DepictQA/weights/`.
    + Download the delta weights of DepictQA-Wild from [this link](https://huggingface.co/zhiyuanyou/DepictQA2-DQ495K/blob/main/ckpt.pt), rename it to `DQ495K_Abstractor.pt`, and put it in `DepictQA/weights/delta/`.
    + Download the delta weights fine-tuned from DepictQA-Wild from [this link](https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view?usp=drive_link) and put it in `DepictQA/weights/delta/`.

    The structure of `DepictQA/weights` should look like this:
    ```
    DepictQA/weights/
    ├── ViT-L-14.pt
    ├── vicuna-7b-v1.5/
    │   └── ...
    └── delta/
        ├── DQ495K_Abstractor.pt
        └── degra_eval.pt
    ```

PS: If you want to fine-tune DepictQA, then
+ Data preparation
    + Put the high-quality images in `DepictQA/experiments/agenticir/training_data/HQ` and the corresponding depth map in `DepictQA/experiments/agenticir/training_data/depth`, the structure of the directory `training_data` should look like this:
        ```
        training_data/
        ├── HQ/
        │   ├── 0001.png
        │   └── ...
        └── depth/
            ├── 0001
            │   └── predict_depth.mat
            └── ...
        ```
        In our implementation, we use the data from [MiOIR](https://github.com/Xiangtaokong/MiOIR?tab=readme-ov-file#step1-download-the-training-data).
    + `cd DepictQA/experiments/agenticir`
    + Run `python synthesize_lq.py` to synthesize the low-quality images.
    + Run `python build_meta.py` to synthesize the training data.
+ In `DepictQA/experiments/agenticir/config_train.yaml`, Set the weight path of ViT (line 33), Vicuna (line 42), and the [pre-trained DepictQA](https://huggingface.co/zhiyuanyou/DepictQA2-Abstractor-DQ495K/blob/main/ckpt.pt) (line 49).
+ Run `sh train.sh ids_of_gpus` to fine-tune the model. The weights will be saved to `DepictQA/experiments/agenticir/ckpt/ckpt.pt`.