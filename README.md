# SECAD-Net
![teaser](https://github.com/BunnySoCrazy/SECAD-Net/blob/main/assets/teaser_github.gif)
This repository provides the official code of SECAD-Net.


## Dependencies

Install python package dependencies:

```bash
$ pip install -r requirements.txt
```


## Dataset

We used the ABC dataset processed by [CAPRI-Net](https://github.com/FENGGENYU/CAPRI-Net), please download it from the link [abc_all.zip](https://drive.google.com/file/d/1DqyZw8zpCiEJMSYp6J6IocMB_IYMwYL1/view) they provided.

## Training & Fine-tuning & Testing

We provide you with basic experiment scripts:

```bash
$ sh scripts/train.sh
$ sh scripts/fine-tuning.sh
$ sh scripts/test.sh
```

## Pre-trained model

Download the pretrained model from this [link](https://drive.google.com/file/d/1uP-AoCqAMOQ9Q2W4fwk0iN5sGCY6UC4x/view?usp=share_link) and put it under `exp_log/ABC/ModelParameters/`, then run the fine-tuning code.

## Acknowledgement

We would like to thank and acknowledge referenced codes from [CSGStumpNet](https://github.com/kimren227/CSGStumpNet), [CAPRI-Net](https://github.com/FENGGENYU/CAPRI-Net) and [DeepCAD](https://github.com/ChrisWu1997/DeepCAD).
