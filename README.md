## About
This repository consists of the implementations for the models proposed in the paper titled "**Multi-source Multi-view Transfer Learning in Neural Topic Modeling with Pretrained Topic and Word Embeddings**" accepted at **NAACL2021**.

	@article{gupta2021multi,
	title={Multi-source Neural Topic Modeling in Multi-view Embedding Spaces},
	author={Gupta, Pankaj and Chaudhary, Yatin and Sch{\"u}tze, Hinrich},
	journal={arXiv preprint arXiv:2104.08551},
	year={2021}
	}

*NOTE*: This code has been built upon the DocNADEe code.


## Requirements
Requires Python 3 (tested with `3.6.5`). The remaining dependencies can then be installed via:

	$ pip install -r requirements.txt
	$ python -c "import nltk; nltk.download('all')"

*NOTE*: installation of correct dependencies and version ensure the correct working of code.


## Data format
"datasets" directory contains different sub-directories for different datasets. Each sub-directory contains CSV format files for training, validation and test sets. The CSV files in the directory must be named accordingly: "**training_docnade.csv**", "**validation_docnade.csv**", "**test_docnade.csv**". For this task, each CSV file (prior to preprocessing) consists of 2 string fields with a comma delimiter - the first is the label and the second is the document body (in bag-of-words representation). Each sub-directory also contains vocabulary file named "**vocab_docnade.vocab**", with 1 vocabulary token per line.


## How to use
The script **train_DocNADE_MVT_MST.py** will train the DocNADE-MVT model and save it in a repository based on perplexity per word (PPL) or information retrieval (IR). It will also log all the training information in the same model folder. Here's how to use the script:

> $ python train_DocNADE_MVT_MST.py --dataset  --docnadeVocab  --model  --num-cores  --use-glove-prior  --use-fasttext-prior  --lambda-glove  --activation  --use-embeddings-prior  --lambda-embeddings  --lambda-embeddings-list  --learning-rate  --batch-size  --num-steps  --log-every  --validation-bs  --test-bs  --validation-ppl-freq  --validation-ir-freq  --test-ir-freq  --test-ppl-freq  --num-classes  --multi-label  --patience  --hidden-size  --vocab-size  --reload  --reload-model-dir  --W-old-path-list  --W-old-vocab-path-list  --gvt-loss  --gvt-lambda  --gvt-lambda-init  --projection  --concat-projection  --concat-projection-lambda 
>
> - Option ``dataset`` is the path to the input dataset.
> - Option ``docnadeVocab`` is the path to vocabulary file used by DocNADE.
> - Option ``model`` is the path to model output directory.
> - Option ``use-glove-prior`` is whether to include glove embedding prior or not.
> - Option ``use-fasttext-prior`` is whether to include fasttext embedding prior or not.
> - Option ``lambda-glove`` lambda value for glove embeddings.
> - Option ``learning-rate`` is learning rate.
> - Option ``batch-size`` is batch size for training data.
> - Option ``num-steps`` is the number of steps to train for.
> - Option ``log-every`` is to print training loss after this many steps.
> - Option ``validation-bs`` is the batch size for validation evaluation.
> - Option ``test-bs`` is the batch size for test evaluation.
> - Option ``validation-ppl-freq`` is to evaluate validation PPL and NLL after this many steps.
> - Option ``validation-ir-freq`` is to evaluate validation IR after this many steps.
> - Option ``test-ir-freq`` is to evaluate test IR after this many steps.
> - Option ``test-ppl-freq`` is to evaluate test PPL and NLL after this many steps.
> - Option ``num-classes`` is number of classes.
> - Option ``patience`` is patience for early stopping criterion.
> - Option ``hidden-size`` is size of the hidden layer.
> - Option ``activation`` is which activation to use: sigmoid|tanh|relu.
> - Option ``vocab-size`` is the vocabulary size.
> - Option ``projection`` is whether to use projection matrix A or not.
> - Option ``reload`` is whether to reload model or not.
> - Option ``reload-model-dir`` is path of directory for which model to be reloaded.
> - Option ``use-embeddings-prior`` is whether to use embeddings prior E from source dataset or not.
> - Option ``lambda-embeddings`` is whether lambda for LVT is static or trainable: manual|automatic.
> - Option ``lambda-embeddings-list`` is a list of lambda parameter for E.
> - Option ``W-old-path-list`` is list of paths of source topic matrices Z.
> - Option ``W-old-vocab-path-list`` is path of source dataset vocabulary.
> - Option ``gvt-loss`` is whether to include topic matrix Z or not.
> - Option ``gvt-lambda`` is whether gamma for Z is static or trainable: manual|automatic.
> - Option ``gvt-lambda-init`` is value of gamma parameter for topic transfer using Z matrices.
> - Option ``concat-projection`` is whether to concatenate prior embeddings or not.
> - Option ``concat-projection-lambda`` is the value of lambda (weight) before adding projected prior embeddings into DocNADE.
> - Option ``use-bert-prior`` is whether to use BERT contextualized embedings as prior or not.
> - Option ``bert-reps-path`` is the path for BERT contextualized embedings.

**Local View Transfer (LVT)**:

	set parameter ``use-embeddings-prior`` to True
	set parameter ``lambda-embeddings-list`` for lambda parameter of LVT accordingly

**Global View Transfer (GVT)**:

	set parameter ``gvt_loss`` to True
	set parameter ``gvt_lambda_init`` for gamma parameter of GVT accordingly

**Multi View Transfer (MVT)**:

	set parameters for LVT and GVT as mentioned above

*NOTE*: Remove a parameter from the list in configuration file if it is not required when running an experiment.
*NOTE*: Sample scripts for three different cases (with best parameters setting) have been provided with the code.


## Dataset and Saved models directories
	datasets directory     ->  ./datasets/
	saved model directory  ->  ./model/


## Model Files
	train_docNADE_MVT_MST.py  ->  Main training file
	model_MVT_MST.py          ->  Model file


## Script Files
	train_20NSshort_ALL_docnade_tanh_LL.sh -> Script file to run MST + MVT for 20NSshort dataset (IR)


## Directory structure containing results of training
Example **model_dir**: 
"20NSshort_ALL_BERT_emb_glove_1.0_emb_lambda_manual_1.0_0.5_0.1_1.0_ftt__bert__act_tanh_hid_200_vocab_1448_lr_0.0001_gvt_loss_True_manual0.1_0.01_0.001_0.1_projection_cp_1.0__2_6_2020"

	**Results directory**           ->  ./<model_dir>/

	**Saved PPL model directory**   ->  ./model/<model_dir>/model_ppl/

	**Saved IR model directory**    ->  ./model/<model_dir>/model_ir/

	**Saved logs model directory**  ->  ./model/<model_dir>/logs/

	**Training information**        ->  ./model/<model_dir>/logs/training_info.txt

	**Reload IR results**           ->  ./model/<model_dir>/logs/reload_info_ir.txt

	**Reload PPL results**          ->  ./model/<model_dir>/logs/reload_info_ppl.txt


## In case of reload use following command line arguments

	--reload              True
	--reload-model-dir:   <model_dir>/
