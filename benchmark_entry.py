from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench = Benchmark('./')

run_benchmark(bench, max_runs=1000, n_jobs=1, n_repetitions=1,
              timeout=20,
              solver_names=["skglm", "PDCD", "Chambolle-Pock"],
              dataset_names=['Simulated'])
