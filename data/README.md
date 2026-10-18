# Dataset retrieval

Expected prepared input: `data/house_sale.csv`, UTF-8, 141,782,218 bytes.
SHA-256: `c9ec2ded82c2e93ab7ad13ad6aed2f6d540d890c3d8d089a8ce9fdb65a3bb7e6`.

Install Git, Git LFS and Python 3.11. From a normal repository-root checkout, initialise local LFS configuration and install the pinned dependencies first:

```powershell
git lfs install --local
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m src.fetch_dataset
.\.venv311\Scripts\python.exe -m src.run_all --output report/reproduced-evaluation --figures report/reproduced-figures
```

On Linux/macOS use `python3.11 -m venv .venv311` and `.venv311/bin/python` for the Python commands. Run from the repository root, not from `data/`.

The helper downloads the exact already-joined export, automatically fetching historical commit `419ebacf59de3723fa7252fed7d972badcffe583` if absent (including shallow clones). It verifies SHA-256, refuses to overwrite an existing different file, and does not stage or restore dataset tracking. Do not re-encode or resave the CSV. `data/*` remains ignored.

Git LFS uses origin's endpoint and Git's configured credential helper. For this public repository retrieval succeeded without an interactive credential prompt. If the server requests authentication, authenticate using your Git credential manager with access to origin's LFS objects; credentials are not stored in project files. Availability depends on the LFS server and access/quota. Ordinary `git lfs pull` at current HEAD does not restore this untracked dataset.

Historical LFS retrieval was tested on 18 October 2026; the result matched both the expected hash and byte count. A fresh-clone test is recorded in the reproduction documentation. This verifies dataset retrieval, not full-data model results. A fresh full end-to-end reproduction remains unverified; the 400-row smoke check does not establish final numerical results.

Original source: https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project

Repository loading/cleaning code consumes a joined export; no original join script or complete recipe was found in the inspected history. A random source CSV is not equivalent. If LFS retrieval fails, ask the data owner for the exact hash-matching export. To recreate it instead, we need the exact source version/archive and filenames, join keys and join type/cardinality, row/duplicate ordering, transformations, and export/encoding settings. These are currently unknown; no speculative preparation script is supplied.

Keep raw data and personal listing information out of commits. Point evaluation to another local file with `--data`; do not edit `src/data_prep.py` to change paths.
