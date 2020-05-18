# Unsupervised Data Augmentation

## Install dependencies and download necessary data
This program is very sensitive to versioning issues, so it's important to install these versions...

In a virtual environment, like conda, with Python v3.6:

```shell
conda create --name {env_name} python=3.6
```

Install dependencies one-by-one:

```shell
pip install tensorflow==1.13.2
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

The following command augments the provided FAQ.csv file. It automatically
splits paragraphs into sentences, translates English sentences to French and
then translates them back into English. Then, it composes the paraphrased
sentences into paragraphs corresponding to the 'Response/Answer' columns. 
Finally, it removes any duplicated paraphrases and appends them to a final 
.csv file. Go to the *back_translate* directory and run:

```shell
bash download.sh
bash run.sh
```
The first run will throw an error that it can't find *vocab.enfr.large.32768* under 
*checkpoints* because the file is created by T2T under another name.  Simply rename 
the created *vocab.* file to the expected filename and run again.

The output is quite verbose because of versioning issues of TF and T2T.

You can run this program on the original document multiple times, to further augment
the data.

### Guidelines for hyperparameters:

There is a variable *sampling_temp* in the bash file. It is used to control the
diversity and quality of the paraphrases. Increasing sampling_temp will lead to
increased diversity but worse quality. Surprisingly, diversity is more important
than quality for many tasks.

You can also specify the number of iterations you would like to run, i.e. the 
number of paraphrases you would like to create for each row on a single run.

If you want to do back translation to a large file, you can change the replicas
and worker_id arguments in run.sh. For example, when replicas=3, we divide the
data into three parts, and each run.sh will only process one part according to
the worker_id.

In trying to find the best balance of new paraphrases, keep in mind the tradeoff 
between *sampling_temp* and *iterations*.  The more iterations you run, the higher 
% of duplicate paraphrases are removed.  The lower the diversity, the higher % of 
duplicate paraphrases are likely to be created and thus removed.

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
