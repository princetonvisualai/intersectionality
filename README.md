# Scripts to run

Can also add command line arguments like "--experiments 1 2" to specify which of the 4 experiments to run
- 0: Identities to include - API (Sec. 4.1)
- 1: Identities to include - Other (Sec. 4.2)
- 2: Evaluation methods (Sec. 6.1)
- 3: Structure in the data (Sec. 5.2)

```
python3 main.py --train_models
python3 main.py --interpret_results
```

# Codebases drawn from
- GRY from https://github.com/algowatchpenn/GerryFair
- RDC from https://fairlearn.org/v0.5.0/api_reference/fairlearn.reductions.html
- GRP from https://github.com/frstyang/fairness-with-overlapping-groups
- RWT from https://github.com/google-research/google-research/blob/master/label_bias/README.md

- data from https://github.com/zykls/folktables

# Packages required
- folktables
- sklearn
- torch
- pandas
