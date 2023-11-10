#!/usr/bin/env bash
DATADIR=./datasets
RESULTSDIR=./results/bos/algospeak
QUEIRESDIR=./datasets/algospeak

mkdir -p "${RESULTSDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${QUEIRESDIR}"

python bos-pred.py --queries "${DATADIR}/algospeak/algospeak.txt" --save "${RESULTSDIR}/algospeak_vectors.txt" --model "./results/bos/demo/model.bos"
python algospeak-embedding-analysis.py --target "${RESULTSDIR}/algospeak_vectors.txt"