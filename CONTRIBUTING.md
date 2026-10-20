# Contributing

Thank you for using LeWi and CSFRBench. This file says how to report a problem, how to add a method, and how the
benchmark stays stable.

## Report a problem

Open an issue at https://github.com/Zhou-Hangyu/lewi-code/issues with:

* what you ran (the command or a short script) and what you expected;
* the output, and the `environment:` line of `csfrbench verify`;
* your operating system, Python version and how you installed the package.

A `VERIFICATION FAILED` line is a problem to report; a `close` check is not (results differ in the last digits between
processors and library versions).

## Add a method

A new method is a subclass of `csfrbench.methods.Method`; `examples/new_method.py` shows the whole path: the method,
a search space frozen on the development instances 6016-6019, 32 configurations per instance and size, and the
comparison with the records (`csfrbench analyze --runs runs/mine --with-record`). Report it as `csfrbench/README.md` asks, under
"Reporting a result". A pull request that adds a method to the package should include its search space, its tests,
and its results under the official protocol.

## Change the code

1. Fork the repository and make a branch.
2. Install with the tests: `pip install -e ".[test]"`.
3. Run `python -m pytest -q` (about 12 minutes on one core) and `csfrbench verify`; both must pass.
4. Open a pull request that says what changed and why.

## The benchmark definition stays fixed

The targets, the data pools, the protocol and the records define the benchmark; they have their own version, 1.5
(`csfrbench.BENCHMARK_VERSION`). The identifiers of the target families, the names of the random streams, the
constants of the normalization, the search spaces and the instance lists are part of it. A change to any of them is a
new benchmark version, never a silent change: it needs a new version number, a CHANGELOG entry, new records, and a
note on what results remain comparable. Changes to the code that keep every result the same (a faster engine, a new
interface, documentation) change only the package version.

## Versions and releases

The package version follows semantic versioning (`pyproject.toml`, `csfrbench.__version__`, `lewi.__version__`,
always the same). Every release has a tag `vX.Y.Z` and a CHANGELOG entry that names the benchmark version it
implements. When the paper's arXiv identifier is announced, the citation in `README.md`, `CITATION.cff` and the project
page gets it in the same release.
