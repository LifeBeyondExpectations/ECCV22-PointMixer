# PointMixer (ECCV2022)
## Install

```bash
# Set up a conda environment
~/point-mixer-cls$ bash setup.sh pm-cls
~/point-mixer-cls$ conda activate pm-cls
```

## Usage

### Classification ModelNet40
**Train**: The dataset will be automatically downloaded, run following command to train.

By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
cd point-mixer-cls/classification_ModelNet40
# train pointMixer
python main.py --model pointMixer
# train pointMixerFinal
python main.py --model pointMixerFinal
# please add other paramemters as you wish.
```


To conduct voting testing, run
```bash
# please modify the msg accrodingly
python voting.py --model pointMixer --msg demo
```


## Acknowledgement

We heavily borrowed codes of [PointMLP (ICLR 2022)](https://github.com/ma-xu/pointMLP-pytorch). We thank the authors of PointMLP for their work.
If you use our model or codes, please consider citing [PointMLP (ICLR 2022)](https://openreview.net/forum?id=3Pbra-_u76D) as well.