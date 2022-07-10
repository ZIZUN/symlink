# symlink
Code for "JBNU-CCLab at SemEval-2022 Task 12: Machine Reading Comprehension and Span Pair Classification for Linking Mathematical Symbols to Their Descriptions", SemEval@NAACL2022 (1st at all subtasks) [paper](https://aclanthology.org/2022.semeval-1.231/)

## Requirements
* [PyTorch](http://pytorch.org/) >= 1.7.1
* pytorch-lightning==0.9.0
* tokenizers==0.9.3
* pandas==1.3.3
* sklearn
* transformers==4.10.2

## Process

1. Environment Setting
```console
pip install -r ./NER/requirements.txt
pip install -r ./RE/requirements.txt
```

3. Entity model(Train, infer)
```console
bash ./NER/scripts/mrc_ner/reproduce/semeval.sh
bash ./NER/scripts/mrc_ner/nested_inference.sh
```

3. Relation model(Train, infer, ensemble and post-process)
```console
bash ./RE/run.sh base [bsz]
python ./RE/inference.py
python ./RE/get_ensemble_result.py
```

4. Make submission
```console
python ./make_result/make_result.py
```

## References
* [mrc-for-flat-nested-ner](https://github.com/ShannonAI/mrc-for-flat-nested-ner)

## Q&A
If you encounter any problem, leave an issue in the github repo.
