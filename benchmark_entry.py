from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench = Benchmark('./')

run_benchmark(bench, max_runs=1000, n_jobs=4, n_repetitions=1, timeout=30,
              solver_names=["PDCD", ],
              dataset_names=['MEG'])
