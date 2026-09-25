# Options by Construction: Transforming NetHack's Action Space into an Easier MDP Without Discovery

PPO on `NetHackChallenge-v0`. A grammar over the primitive keystroke table enumerates 227 closed-loop `(I, π, β)` options (Sutton, Precup and Singh 1999); nothing is discovered or trained to produce them. Three action tables (`action`, `option`, `both`) are compared at a shared budget of 10^7 primitive steps, five seeds, primitive (SMDP) discounting. A 64-option grammar catalogue ends at median return 100.42 (observed range 91.79-101.14) against 12.54 (6.48-15.76) for primitives.

## Key results

![Episodic return, three action tables, n=64](NLE/plots/0830-172036__Challenge__exp1/return_curve.png)

Last point of the 100-episode moving-average curve on `NetHackChallenge-v0` (`n=64`, grammar, `discount=primitive`, 10^7 primitive steps, five seeds). Median across seeds; band is the observed range.


| Condition | Catalogue                   | Median | Range        |
| --------- | --------------------------- | ------ | ------------ |
| `action`  | primitives only             | 12.54  | 6.48-15.76   |
| `option`  | grammar, n=64               | 100.42 | 91.79-101.14 |
| `both`    | primitives + that catalogue | 81.24  | 70.73-97.96  |




## What is being compared

Three conditions, identical in every other respect:


| Condition | Action table                               |
| --------- | ------------------------------------------ |
| `action`  | the environment's own primitives           |
| `option`  | a catalogue of temporally extended options |
| `both`    | the primitives followed by the catalogue   |


`both` is the control. It offers strictly more choices than the baseline, so a gain there cannot be explained by primitives having been taken away.

## Setup

- Linux and an NVIDIA GPU. `NLE/main.py` and `navix/main.py` .
- conda env `options-tcap` from `[environment.yml](environment.yml)`: Python 3.11, PyTorch (`cu128`), JAX (`cuda12`), `nle`, `navix`, and the `cmake` / `bison` / `flex` stack NLE needs to build NetHack.

```bash
conda env create -f environment.yml
conda activate options-tcap
```



## Reproduce

No separate dataset. `nle` in `environment.yml` builds NetHack. Episode logs are not included, so reproducing a figure means training first, then plotting it with `plot.py`:

```bash
cd NLE
python main.py --sweep exp1 --seeds 5
python plot.py --cells '*__exp1'
```

Print a matrix without a GPU: `python main.py --sweep exp1 --dry-run`.

## Repo layout

```
NLE/              PyTorch, CleanRL PPO, NetHack. Primary.
NLE/test_files/   pytest
NLE/plots/        figures and the CSVs they were drawn from
navix/            JAX, Navix MiniGrid. Testbed.
navix/test_files/ pytest
```



## Method notes

- **Seeds.** Published NLE cells use five training seeds. The runner default is `3`.
- **Budget.** 10^7 primitive steps on the pushed Challenge groups; episodes truncated at 5,000 primitive steps. Learning rate anneals against `frames / budget`. An option cell therefore buys fewer gradient updates than an `action` cell at the same budget. That confound favours `action`.
- **Metric.** Median across seeds of the last point of the 100-episode moving-average return; the band is the observed range.
- **x-axis.** `primitive_step` for every cross-condition comparison. Decision counts are not comparable.

