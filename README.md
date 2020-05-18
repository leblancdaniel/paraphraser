# Unsupervised Data Augmentation

The program augments the provided FAQ.csv file in 5 main steps:
1) Split paragraphs into sentences; 
2) Translate English sentences to French; *(languages can be changed)*
3) Translate them back into English; 
4) Compose the paraphrased sentences back into paragraphs, and;  
5) Remove any duplicated paraphrases and append them to a new .csv file 

## Install dependencies and download necessary data
This program is sensitive to versioning issues, so it's important to install the specified
library versions to run.  If you have more compatible versions, please feel free to update.

In a virtual/conda environment with Python v3.6:

```shell
conda create --name {env_name} python=3.6
```

Install dependencies either one-by-one:

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

Go to the *back_translate* directory and run:

```shell
bash download.sh
```

## Run back translation data augmentation for your dataset

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
