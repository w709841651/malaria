### TAB-DDPM GUIDELINE https://github.com/yandex-research/tab-ddpm

### DATASET:
malaria

### To run TabDDPM tuning:
```
python scripts/tune_ddpm.py malaria 248 synthetic mlp ddpm_tune --eval_seeds
```

### To run TabDDPM pipeline:
```
python scripts/pipeline.py --config exp/malaria/ddpm_mlp_best/config.toml --train --sample
```
