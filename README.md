# 211201, ETRI 음성인식 수업, ESPNet을 이용해서 ASR을 만든다.

12000시간으로 학습해도 wer 16 음성데이터가 어마어마하게 많아야 하고 질도 중요하다. 함부로 접근하지 말자

숫자를 다 한글로, '?, !'등은 없애고 학습하는게 더 좋다. 한글은 글자단위로, 영어는 sentencepiece 단위로 학습한다.

음성을 Kaldi로 melspectrogram으로 변환한 후 CNN(256dim)+Transformer(n=12)로 sequence를 생성한다



# https://github.com/pkyoung/a1003


# Prepare Backend.AI Env.

* Login: `address_hidden`
* Create Private Folder
    shared
* Create Session
    * Select Environments: `pytorch-espnet`
    * Select Version: `210325`
    * Mount Folders: `a1003` and `shared`
    * Max `CPU`, `RAM`, `Shared Memory`, `GPU`
    * 1 `Session`

# Start Session
    * (unblock popup)
    * Try **copy and paste**

## Prepare for tensorboard

    mkdir -p /home/work/logs
    ln -s /home/work/a1003/models/ref1x /home/work/logs/ref1x

## Try Apps
    * `Visual Studio Code`
    * `Console`
    * `Tensorboard`

## Download scripts

    git clone https://github.com/pkyoung/a1003.git ./train.1
    cd train.1
    cd data && tar xvzf ks.tgz && cd ..
    ln -s /opt/kaldi/egs/wsj/s5/steps .
    ln -s /opt/kaldi/egs/wsj/s5/utils .
    


# Train ASR

## Choie: Set training corpus

    cat data/ks/uttid.01 data/ks/uttid.03 data/ks/uttid.05 > data/ks/uttid.train
    cat data/ks/uttid.01 data/ks/uttid.03 > data/ks/uttid.train
    cat data/ks/uttid.01 > data/ks/uttid.train

## Prepare data

    source path.sh
    mkdir -p data/train data/test data/dev
    filter_scp.pl data/ks/uttid.test data/ks/wav.scp > data/test/wav.scp
    filter_scp.pl data/ks/uttid.dev data/ks/wav.scp > data/dev/wav.scp
    filter_scp.pl data/ks/uttid.train data/ks/wav.scp > data/train/wav.scp
    filter_scp.pl data/ks/uttid.test data/ks/text > data/test/text
    filter_scp.pl data/ks/uttid.dev data/ks/text > data/dev/text
    filter_scp.pl data/ks/uttid.train data/ks/text > data/train/text
    awk '{print $1 " " $1}' data/test/wav.scp > data/test/spk2utt
    awk '{print $1 " " $1}' data/dev/wav.scp > data/dev/spk2utt
    awk '{print $1 " " $1}' data/train/wav.scp > data/train/spk2utt
    cp data/test/spk2utt data/test/utt2spk
    cp data/dev/spk2utt data/dev/utt2spk
    cp data/train/spk2utt data/train/utt2spk

## Run training

    ./steps/make_fbank.sh data/test
    ./steps/make_fbank.sh data/dev
    ./steps/make_fbank.sh data/train

    bash stage3-5.sh
    bash stage9.sh
    bash stage10.sh

    ## testing
    bash stage11.sh

## Tensorboard

    cd /home/work
    mkdir -p logs

    ## tensorboard 위치는 환경에 맞게
    ## train.1 은 아무 이름이나 구분되는 이름으로
    ln -s /home/work/train.1/exp/exp01a/tensorboard ./logs/train.1

# Testing

## Prepare Data

* Prepare files `wav.scp`,`text`,`spk2utt`,`utt2spk` in `data/mydata`
* If you don't have your data,

```
    mkdir data/mydata
    cp /home/work/a1003/db/navidlg/* data/mydata
    ## ignore cp -r ... message
```

## Run inference

* Edit inference.sh and run it

    ./inference.sh

* Measure WER and CER

    python local/uttwer.py data/mydata/text result/text
    python local/uttcer.py data/mydata/text result/text

# Pretrained models

* 훈련데이터

| model | training data | hours  | elapsed |
| ---   | ---           | ---    | ---     |
| 01    | 01            | 173.9h | 8h 8m   |
| 03    | 03            | 192.3h | 8h 55m  |
| 2x    | 01 + 03       | 366.3h | 16h 4m  |
| 3x    | 01 + 03 + 05  | 563.7h | 40h 50m (1080ti x1) |

* 모델 위치

    /home/work/a1003/models/ref01/valid.acc.ave_10best.pth
    /home/work/a1003/models/ref03/valid.acc.ave_10best.pth
    /home/work/a1003/models/ref2x/valid.acc.ave_10best.pth
    /home/work/a1003/models/ref3x/valid.acc.ave_10best.pth

* tensorboard

```
    ln -s /home/work/a1003/models/ref1x/exp01a/tensorboard /home/work/logs/ref1x
    ln -s /home/work/a1003/models/ref2x/exp01a/tensorboard /home/work/logs/ref2x
    ln -s /home/work/a1003/models/ref3x/exp01a/tensorboard /home/work/logs/ref3x
    ln -s /home/work/a1003/models/ref3x_sa/exp01a/tensorboard /home/work/logs/ref3x_sa
    ln -s /home/work/a1003/models/ref1x_lr5/exp01a/tensorboard /home/work/logs/ref1x_lr5
```
* Evaluation results: data/test

| model    | CER | WER |
| ------   | --- | ---    |
| ref1x    |   |  |
| ref2x    |   |  |
| ref3x    |   |  |
| ref3x+sa

## Evaluate mydata with pretrained models

* Edit inference.sh and run it

    model=/home/work/a1003/models/ref01/valid.acc.ave_10best.pth
    model=/home/work/a1003/models/ref03/valid.acc.ave_10best.pth
    model=/home/work/a1003/models/ref2x/valid.acc.ave_10best.pth
    model=/home/work/a1003/models/ref3x/valid.acc.ave_10best.pth

    bash inference.sh
    
    inference.sh 파일에서 아래 부분의 데이터, 모델경로 수정
    # Configuraiton
    idir=data/mydata
    odir=./result
    model=exp/exp01a/42epoch.pth
    nj=8

    # End of Configuraiton
    source path.sh  # 꼭 추가
    mkdir -p $odir/log
    mdir=$(dirname $model)

* Measure CER/WER

    python local/uttcer.py data/mydata/text result/text  # 정답txt, 추론txt
    python local/uttwer.py data/mydata/text result/text  # 정답txt, 추론txt
    

