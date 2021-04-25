#!/bin/bash
python train_DocNADE_MVT_MST.py --dataset ./datasets/20NSshort --docnadeVocab ./datasets/20NSshort/vocab_docnade.vocab --model ./model/20NSshort_ALL_BERT --num-cores 1 --use-glove-prior True --use-fasttext-prior True --use-bert-prior True --bert-reps-path ./datasets/BERT/20NSshort --lambda-glove 1.0 --activation tanh --use-embeddings-prior True --lambda-embeddings manual --lambda-embeddings-list 1.0 0.5 0.1 1.0 --learning-rate 0.0001 --batch-size 100 --num-steps 5000000000 --log-every 2 --validation-bs 1 --test-bs 1 --validation-ppl-freq 100000000 --validation-ir-freq 12 --test-ir-freq 100000000 --test-ppl-freq 100000000 --num-classes 20 --multi-label False --patience 100 --hidden-size 200 --vocab-size 1448 --trainfile ./datasets/20NSshort/training.txt --valfile ./datasets/20NSshort/validation.txt --testfile ./datasets/20NSshort/test.txt --reload False --reload-model-dir BERT_results/20NSshort_ALL_BERT_emb_glove_1.0_emb_lambda_manual_1.0_0.5_0.1_1.0_ftt__bert__act_tanh_hid_200_vocab_1448_lr_0.0001_gvt_loss_True_manual0.1_0.01_0.001_0.1_projection_cp_1.0__26_5_2020 --W-old-path-list ./W_DocNADE_ir/W_20NS.npy ./W_DocNADE_ir/W_TMN.npy ./W_DocNADE_ir/W_R21578.npy ./W_DocNADE_ir/W_AGnews.npy --W-old-vocab-path-list ./datasets/20NS/vocab_docnade.vocab ./datasets/TMN/vocab_docnade.vocab ./datasets/R21578/vocab_docnade.vocab ./datasets/AGnews/vocab_docnade.vocab --gvt-loss True --gvt-lambda manual --gvt-lambda-init 0.1 0.01 0.001 0.1 --projection True --concat-projection True --concat-projection-lambda 1.0