# LyricPsych:Task Runner

This repository contains codes for the task simulation using various lyrics based feature set. For all experiments, we adopted the full-factorial design with respect to the variables of interests (set of features x systems).

We expect feature files are extracted and corresponding target (label) files also pre-processed. We provide our data that are used for the study. (TODO: link to those files)


## Getting Started

To get the package setup, one can simply clone the repository and install the dependencies. We recommend the virtual environment. Current task runs are tested on python `3.6`, `3.7` empirically (concrete unit test will be added).

```console
(venv) $ git clone https://github.com/eldrin/lyricpsych-tasks.git
(venv) $ pip install -r requirements.txt
```

If there was no significant error, you are good to go!


## Run Tasks

In the repository, there are 3 python executable scripts in `cli` directory. Each of them are dedicated for running one of the tasks we tested: `music autotagging`, `music genre classification`, and `music recommendation`. For instance, `music autotagging` task can be run by the following command:

```console
(venv) $ python cli/run_autotagging.py \
  /path/to/auto_tagging_label_data.h5 \
  /path/to/lyrics_feature_data.h5 \
  data/autotagging_design_fullfactorial.json 0 \
  --out-fn=/path/to/save/output.csv \
  --n-rep=5
```

Last two arguments are the experimental design matrix file for the corresponding task and the target row (starting from 0) to be simulated, respectively. By default, each run will repeat `5` trials with different random splits per each time. The output will be stored in the specified (`--out-fn`) location with `csv` format.

Similarly, one can run other task simulations with similar manner:

```console
(venv) $ python cli/run_genreclf.py \
  /path/to/genreclf_label_data.h5 \
  /path/to/lyrics_feature_data.h5 \
  data/genreclf_design_fullfactorial.json 0 \
  --out-fn=/path/to/save/output.csv \
  --n-rep=5
```

...and finally.

```console
(venv) $ python cli/run_recsys.py \
  /path/to/recsys_label_data.h5 \
  /path/to/lyrics_feature_data.h5 \
  data/recsys_design_fullfactorial.json 0 \
  --out-fn=/path/to/save/output.csv \
  --n-rep=5
```

## Current Status and Contribution

Current status of the codebase is mostly for the reproducibility, not focusing the generalization of this experimental instrument going beyond the work we have conducted. We are going to update a few things, but there is no major milestones yet.

However, we are always welcome to any kind of issue / bug reports and suggestions. Feel free to drop those on issue board.


## Authors

Jaehun Kim


## LICENSE

This reproduction package is distributed under MIT LICENSE.


## TODO

- [ ] put links to the file in here
- [ ] docstring
- [ ] testing / CI
- [ ] better README


## Reference

TBD
