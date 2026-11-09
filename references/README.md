# Reference implementation

This project uses the `nanolm` implementation from:

- Repository: https://github.com/lkk688/DeepDataMiningLearning
- Package location inside that repository: `DeepDataMiningLearning/llm/nanolm`

The dependency is kept outside this repository at:

`../_deps/DeepDataMiningLearning`

Run `bash scripts/bootstrap_upstream.sh` from the project root. On the first run,
`references/upstream_commit.txt` is replaced with the exact upstream Git commit that
was cloned. Commit that resolved SHA after the smoke test succeeds. Later setup runs
will check out the pinned commit instead of following `main`.

The experiment definitions in this repository are adapted from the public `nanolm`
examples and tutorial. The upstream source code is not vendored here.
