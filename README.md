# merfishing

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/lhqing/merfishing/Test/main
[link-tests]: https://github.com/lhqing/merfishing/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/merfishing

Code for MERFISH analysis

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install merfishing:

<!--
1) Install the latest release of `merfishing` from `PyPI <https://pypi.org/project/merfishing/>`_:

```bash
pip install merfishing
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lhqing/merfishing.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lhqing/merfishing/issues
[changelog]: https://merfishing.readthedocs.io/latest/changelog.html
[link-docs]: https://merfishing.readthedocs.io
[link-api]: https://merfishing.readthedocs.io/latest/api.html

## TODO list

### User steps

1. copy the merfish_raw (Tbs) and merfish_output (100Gbs - 1Tb) to network file system
2. run yap merfish preprocessing
3. send tar.gz path to Joe Nery for tape archive

### YAP merfish processing

-   raw
    1.  tar.gz the raw files
    1.  tar.gz the output files
    1.  delete raw files once tar.gz successfully
-   output
    1.  Save all TIFF images into zarr dataset, this reduce the file size by three folds
    2.  if there is smFISH images, do the spot detection with big-fish package automated pipeline
    3.  delete original TIFF files
