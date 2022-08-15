# PointMixer (ECCV2022)
## Install

```bash
# Set up a conda environment
~/ECCV22-PointMixer/obj_cls$ bash setup.sh pm-cls
~/ECCV22-PointMixer/obj_cls$ conda activate pm-cls
```

## Usage

### Classification ModelNet40
**Train**: The dataset will be automatically downloaded, run following command to train.

By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
~/ECCV22-PointMixer$ cd obj_cls/classification_ModelNet40
# train pointMixerFinal
~/ECCV22-PointMixer/obj_cls/classification_ModelNet40$ python main.py --model pointMixerFinal
# please add other paramemters as you wish.
```


To conduct voting testing, run
```bash
# please modify the msg accrodingly
~/ECCV22-PointMixer/obj_cls/classification_ModelNet40$ python voting.py --model pointMixerFinal --msg demo
```


## Acknowledgement

We heavily borrowed codes of [PointMLP (ICLR 2022)](https://github.com/ma-xu/pointMLP-pytorch) and [PointTransformer (ICCV 2021)](https://github.com/POSTECH-CVLab/point-transformer). We thank the authors of PointMLP and PointTransformer for their work.
If you use our model or codes, please consider citing them as well.