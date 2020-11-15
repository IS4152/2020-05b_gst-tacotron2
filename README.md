# GST-Tacotron 2

An emotional speech synthesis research project conducted as part of [IS4152](https://is4152.github.io/) coursework. This repository contains code that can be used to train a speech synthesis model that attempts to generate speech-like sounds to express a chosen emotion.

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Set up repository

1. Clone this repo: `git clone https://github.com/taneliang/mellotron.git`
1. CD into this repo: `cd mellotron`
1. Initialize submodule: `git submodule init; git submodule update`

## Set up dependencies

1. Check CUDA toolkit version: `nvcc --version`. NB: This is the toolkit version, which may be different from the version reported by nvidia-smi.
1. Create Python 3 virtual environment: `python3 -m venv .env-cuda<CUDA version>`
1. Activate venv, by running one of the following:
    - `bash`/`sh`: `source .env-cudaxxx/bin/activate`
    - `csh`: `source .env-cudaxxx/bin/activate.csh`
    - `fish`: `source .env-cudaxxx/bin/activate.fish`
1. Install [PyTorch 1.0]. As the time this was written, these are the instructions:
    - CUDA 10.0: `pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/cu100/torch_stable.html`
    - CUDA 10.1: `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
    - CUDA 10.2 or 11.0: `pip install torch torchvision`
1. Install [Apex]:
    ```sh
    pushd ..
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    popd
    ```
1. Install Python requirements: `pip install -r requirements.txt`

### Set up data for training

1. EmoV-DB:
    1. Download the [EmoV-DB dataset](https://github.com/numediart/EmoV-DB)
    1. Normalize it: `ls */*/*.wav | xargs -I % sh -c 'mkdir -p ../out/$(dirname %) && sox % --rate 16000 -c 1 -b 16 ../out/%'` 
    1. Trim leading and trailing silences: `ls */*/*.wav | xargs -I @ sh -c 'mkdir -p ../out-no-silence/$(dirname @) && sox @ --rate 16000 -c 1 -b 16 ../out-no-silence/@ silence 1 0.1 1% reverse silence 1 0.1 1% reverse'`
    1. (Optional) Manually trim non-verbal expressions:
        1. Generate a CSV file to be manually filled in with trim timestamps: `./genmanualtrimlist.py`
        1. Use the CSV file to trim files: `./createcleanemovdb.py`
1. LJSpeech:
    1. Download the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
    1. Normalize it: `mkdir ../../LJSpeech-1.1/wavs && ls *.wav | xargs -I % sh -c 'sox % --rate 16000 -c 1 -b 16 ../../LJSpeech-1.1/wavs/%'`
1. Generate filelist files:
    ```sh
    cd scripts
    vim ./genfilelist.py # Configure the script before running
    ./genfilelist.py
    cd ..
    ```

## Training
1. Update the filelists inside the filelists folder to point to your data
2. `python train.py --output_directory=outdir --log_directory=logdir`
3. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the emotion embedding layer is [ignored]

1. `python train.py --output_directory=outdir --log_directory=logdir -c models/mellotron_libritts.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. `jupyter notebook --ip=127.0.0.1 --port=31337`
2. Load inference.ipynb 
3. (optional) Download our published [WaveGlow](https://drive.google.com/open?id=1okuUstGoBe_qZ4qUEF8CcwEugHP7GM_b) model

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis.

## Acknowledgements
This project is a slight modification of [Mellotron](https://github.com/NVIDIA/mellotron), developed by Rafael Valle, Jason Li, Ryan Prenger and Bryan Catanzaro.

In turn, Mellotron uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft), 
[Chengqi Deng](https://github.com/KinglittleQ/GST-Tacotron),
[Patrice Guyot](https://github.com/patriceguyot/Yin), as described in our code.

[ignored]: https://github.com/NVIDIA/mellotron/blob/master/hparams.py#L22
[paper]: https://arxiv.org/abs/1910.11997
[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[LibriTTS]: https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI
[LJS]: https://drive.google.com/open?id=1UwDARlUl8JvB2xSuyMFHFsIWELVpgQD4
[pytorch]: https://github.com/pytorch/pytorch#installation
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp