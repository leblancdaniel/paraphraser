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
pip install pandas
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
.csv file. 

Go to the *back_translate* directory and run:

```shell
bash download.sh
```

Next, under the *checkpoints* folder, rename *vocab.translate_enfr_wmt32k.32768.subwords* 
to *vocab.enfr.large.32768*, and run the following command:

```shell
bash run.sh
```

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


## Citation

You can find the original paper on UDA here: https://arxiv.org/abs/1904.12848
