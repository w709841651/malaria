1.1clone: download software and data

1.2creat_dataset.ipynb:  creat .npy files in my_dataset according to the readme file

my_dataset:        my own dataset 

classif-cat-large-0-road-safety: their dataset(can operate normally) There are a lot of different datasets in their data folder, and I chose this one, which seems to match my data type

0-kl-tuning.toml:  this file was copied from exp/tabr/why/classif-cat-large-0-road-safety/0-tuning.toml and i only changed the path 

run:
```
CUDA_VISIBLE_DEVICES=0 srun -p 10-10-10-17 python bin/go.py exp/0-kl-tuning.toml --force
```
error: 

```
Matplotlib created a temporary config/cache directory at /tmp/matplotlib-foqdkfgb because the default path (/home/zy/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Removing the existing output
[I 2024-02-06 17:17:04,365] A new study created in memory with name: no-name-c6631399-b50f-4605-9080-a215a5d00e62
================================================================================
[>>>] exp/0-kl-tuning | 2024-02-06 17:17:03.073267
--------------------------------------------------------------------------------
{'seed': 0,
 'function': 'bin.tabr.main',
 'space': {'seed': 0,
           'batch_size': 64,
           'patience': 16,
           'n_epochs': inf,
           'context_size': 96,
           'data': {'seed': 0,
                    'cache': True,
                    'path': ':data/my_dataset',
                    'num_policy': 'quantile',
                    'cat_policy': 'ordinal',
                    'y_policy': None},
           'model': {'num_embeddings': None,
                     'd_main': ['_tune_', 'int', 16, 384],
                     'context_dropout': ['_tune_', 'uniform', 0.0, 0.6],
                     'd_multiplier': 2.0,
                     'encoder_n_blocks': 0,
                     'predictor_n_blocks': 1,
                     'mixer_normalization': 'auto',
                     'dropout0': ['_tune_', 'uniform', 0.0, 0.6],
                     'dropout1': 0.0,
                     'normalization': 'LayerNorm',
                     'activation': 'ReLU'},
           'optimizer': {'type': 'AdamW',
                         'lr': ['_tune_', 'loguniform', 3e-05, 0.001],
                         'weight_decay': ['_tune_', '?loguniform', 0.0, 1e-06, 0.0001]}},
 'n_trials': 100,
 'timeout': None,
 'sampler': {}}
--------------------------------------------------------------------------------
  0%|          | 0/100 [00:00<?, ?it/s]Creating the output
Using cached dataset: build_dataset__my_dataset__quantile__ordinal__None__None__0__6fb5db787ae75a524425e31a0c42c2d8.pickle

...
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [105,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [106,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [107,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [108,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [109,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [110,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [111,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [112,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [113,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [114,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [115,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [116,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [117,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [118,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [119,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [120,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [121,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [122,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [123,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [124,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [125,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [126,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:91: operator(): block: [5,0,0], thread: [127,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
eval_batch_size = 16384
eval_batch_size = 8192
eval_batch_size = 4096
eval_batch_size = 2048
eval_batch_size = 1024
eval_batch_size = 512
eval_batch_size = 256
eval_batch_size = 128
eval_batch_size = 64
eval_batch_size = 32
eval_batch_size = 16
eval_batch_size = 8
eval_batch_size = 4
eval_batch_size = 2
eval_batch_size = 1
eval_batch_size = 0
  0%|          | 0/100 [00:18<?, ?it/s]
[W 2024-02-06 17:13:35,274] Trial 0 failed because of the following error: KeyError('val')
Traceback (most recent call last):
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/_optimize.py", line 213, in _run_trial
    value_or_values = func(trial)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/tune.py", line 153, in objective
    report = function(raw_config, Path(tmp) / 'output')
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/tabr.py", line 503, in main
    lib.print_metrics(mean_loss, metrics)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/lib/util.py", line 455, in print_metrics
    f'(val) {metrics["val"]["score"]:.3f}'
KeyError: 'val'
Traceback (most recent call last):
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/go.py", line 52, in <module>
    lib.run_cli(main)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/lib/util.py", line 535, in run_cli
    return fn(**vars(args))
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/go.py", line 36, in main
    bin.tune.main(tuning_config, tuning_output, continue_=continue_, force=force)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/tune.py", line 185, in main
    study.optimize(
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/study.py", line 400, in optimize
    _optimize(
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/_optimize.py", line 66, in _optimize
    _optimize_sequential(
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/_optimize.py", line 163, in _optimize_sequential
    trial = _run_trial(study, func, catch)
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/_optimize.py", line 264, in _run_trial
    raise func_err
  File "/ceph-data/zy/software/anaconda3/envs/pyt/lib/python3.9/site-packages/optuna/study/_optimize.py", line 213, in _run_trial
    value_or_values = func(trial)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/tune.py", line 153, in objective
    report = function(raw_config, Path(tmp) / 'output')
  File "/ceph-data/zy/wxy/tabular-dl-tabr/bin/tabr.py", line 503, in main
    lib.print_metrics(mean_loss, metrics)
  File "/ceph-data/zy/wxy/tabular-dl-tabr/lib/util.py", line 455, in print_metrics
    f'(val) {metrics["val"]["score"]:.3f}'
KeyError: 'val'
Faiss assertion 'err == cudaSuccess' failed in virtual void faiss::gpu::StandardGpuResourcesImpl::deallocMemory(int, void*) at /home/conda/feedstock_root/build_artifacts/faiss-split_1636459943780/work/faiss/gpu/StandardGpuResources.cpp:518; details: Failed to cudaFree pointer 0x7f45d6075600 (error 710 device-side assert triggered)
```
