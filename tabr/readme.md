###TABR GUIDELINE(https://mamba.readthedocs.io/en/latest/installation.html)

To set up the environment:
```
git clone https://github.com/yandex-research/tabular-dl-tabr
wget https://huggingface.co/datasets/puhsu/tabular-benchmarks/resolve/main/data.tar -O tabular-dl-tabr.tar.gz
tar -xvf tabular-dl-tabr.tar.gz
```

To creat a new dataset:

run tabr_data_preprocess.ipynb       #creat .npy files in the malaria folder 


To reproduce the results：
```
python bin/go.py exp/malaria/0-tuning.toml --force
```
