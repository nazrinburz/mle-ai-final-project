"""Retrieve the exact historical joined export without tracking raw data."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


DATA_SHA256="c9ec2ded82c2e93ab7ad13ad6aed2f6d540d890c3d8d089a8ce9fdb65a3bb7e6"
DATA_COMMIT="419ebacf59de3723fa7252fed7d972badcffe583"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=root/"data"/"house_sale.csv")
    args=parser.parse_args()
    if args.output.exists():
        if digest(args.output)!=DATA_SHA256:
            parser.error("Destination exists with a different hash; choose another path.")
        print(f"Verified existing dataset: {args.output}")
        return
    storage=root/"tmp"/"lfs"
    available=subprocess.run(
        ["git","cat-file","-e",f"{DATA_COMMIT}^{{commit}}"],cwd=root,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    )
    if available.returncode:
        subprocess.run(["git","fetch","origin",DATA_COMMIT],cwd=root,check=True)
    subprocess.run([
        "git","-c",f"lfs.storage={storage.as_posix()}","lfs","fetch",
        "origin",DATA_COMMIT,"--include=data/house_sale.csv","--exclude=",
    ],cwd=root,check=True)
    source=storage/"objects"/DATA_SHA256[:2]/DATA_SHA256[2:4]/DATA_SHA256
    if digest(source)!=DATA_SHA256:
        raise ValueError("Historical dataset hash mismatch.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an existing local dataset.
    with source.open("rb") as src,args.output.open("xb") as dst:
        shutil.copyfileobj(src,dst)
    print(f"Retrieved {args.output}; SHA-256 {digest(args.output)}")


if __name__=="__main__":
    main()
