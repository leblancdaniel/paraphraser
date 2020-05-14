# Unsupervised Data Augmentation

## Install dependencies and download necessary data

```shell
conda create --name {env_name} --python=3.6
pip install tensorflow==1.13
pip install tensor2tensor==1.8
pip install absl-py
pip install nltk
```

Or simply:

```shell
pip install -r requirements.txt
```

Then, download 'punkt' from the 'nltk' library:

```shell
python -c "import nltk; nltk.download('punkt')"
```

## Run back translation data augmentation for your dataset

The following command translates the provided example file. It automatically
splits paragraphs into sentences, translates English sentences to French and
then translates them back into English. Finally, it composes the paraphrased
sentences into paragraphs. Go to the *back_translate* directory and run:

```shell
bash download.sh
bash run.sh
```
The first run will throw an error that it can't find *vocab.enfr.large.32768* 
under *checkpoints* because the file is created under another name.  
Simply rename the created *vocab.* file to the expected filename and run again.

The output is quite verbose because of versioning issues of TF and T2T.

### Guidelines for hyperparameters:

There is a variable *sampling_temp* in the bash file. It is used to control the
diversity and quality of the paraphrases. Increasing sampling_temp will lead to
increased diversity but worse quality. Surprisingly, diversity is more important
than quality for many tasks we tried.

We suggest trying to set sampling_temp to 0.7, 0.8 and 0.9. If your task is very
robust to noise, sampling_temp=0.9 or 0.8 should lead to improved performance.
If your task is not robust to noise, setting sampling temp to 0.7 or 0.6 should
be better.

If you want to do back translation to a large file, you can change the replicas
and worker_id arguments in run.sh. For example, when replicas=3, we divide the
data into three parts, and each run.sh will only process one part according to
the worker_id.

## General guidelines for setting hyperparameters:

UDA works out-of-box and does not require extensive hyperparameter tuning, but
to really push the performance, here are suggestions about hyperparamters:

*   It works well to set the weight on unsupervised objective *'unsup_coeff'*
    to 1.
*   Use a lower learning rate than pure supervised learning because there are
    two loss terms computed on labeled data and unlabeled data respecitively.
*   If your have an extremely small amount of data, try to tweak
    'uda_softmax_temp' and 'uda_confidence_thresh' a bit. For more details about
    these two hyperparameters, search the "Confidence-based masking" and
    "Softmax temperature control" in the paper.
*   Effective augmentation for supervised learning usually works well for UDA.


## Acknowledgement

A large portion of the code is taken from
[BERT](https://github.com/google-research/bert) and
[RandAugment](https://github.com/tensorflow/models/tree/master/research/autoaugment).
Thanks!

## Citation

Please cite this paper if you use UDA.

```
@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}
```

## Disclaimer

This is not an officially supported Google product.
