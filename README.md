# Corridor MPC: Towards Optimal and Safe Trajectory Tracking
We present a framework for safe and optimal
trajectory tracking by combining Model Predictive Control
and Sampled-Data Control Barrier functions. This framework,
which we call Corridor MPC, safely and robustly keeps the state
of the system within a corridor that is defined as a permissible
error around a reference trajectory. By incorporating SampledData Control Barrier functions into an MPC framework, we
guarantee safety for the continuous-time system in the sense
of staying within the corridor and practical stability in the
sense of converging to the reference trajectory

## Running the demo
To install the package, make sure to have installed [Python Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) in your system. Then, run the following commands:

1. Clone the repository:
```shell
git clone git@github.com:KTH-DHSG/corridor_mpc.git
```

_Optional_: Configure `poetry` to create a local virtual environment:
```
poetry config virtualenvs.in-project true
```

2. Install the `corridor_mpc` library:
```shell
poetry install
```

3. Run the demo script:
```shell
poetry run python scripts/run_tracking.py
```

## Citing this work
To cite this work, please use the following BibTeX entry:
```
@inproceedings{9867764,
  author = {Roque, P. and Cortez, W. Shaw and Lindemann, L. and Dimarogonas, D. V.},
  booktitle = {2022 American Control Conference (ACC)},
  title = {Corridor MPC: Towards Optimal and Safe Trajectory Tracking},
  year = {2022},
  pages = {2025-2032},
  doi = {10.23919/ACC53348.2022.9867764},
}
```