# cs4650-final-project
## Setup
# Installations
```bash
conda env create -f environment.yml
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
# Download the models
1. English Language Model: https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz
2. German Language Model: https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.gz

# Extract the models to the top level directory of this repo and just run the notebook file

## Preprocess
```bash
# Download and prepare the data
cd fairseq/examples/translation/
bash prepare-iwslt14.sh
cd ../../..

# Preprocess/binarize the data
TEXT=fairseq/examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
    --srcdict wmt19.de/dict.txt
    --tgtdict wmt19.en/dict.txt
```

## Train
# Translation Model
```bash
CUDA_VISIBLE_DEVICES=0 python train.py     data-bin/iwslt14.tokenized.de-en     --arch transformer_iwslt_de_en --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000     --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 512     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
