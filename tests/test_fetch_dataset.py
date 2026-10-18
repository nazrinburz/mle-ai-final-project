import hashlib
import sys

import pytest

from src import fetch_dataset


def test_existing_verified_dataset_needs_no_network(tmp_path, monkeypatch):
    target=tmp_path/"dataset.csv"
    target.write_bytes(b"verified data")
    monkeypatch.setattr(fetch_dataset,"DATA_SHA256",hashlib.sha256(target.read_bytes()).hexdigest())
    monkeypatch.setattr(sys,"argv",["fetch_dataset","--output",str(target)])

    def unexpected(*args, **kwargs):
        pytest.fail("An existing verified dataset must not trigger Git commands")

    monkeypatch.setattr(fetch_dataset.subprocess,"run",unexpected)
    fetch_dataset.main()
    assert target.read_bytes()==b"verified data"


def test_existing_different_dataset_is_preserved(tmp_path, monkeypatch):
    target=tmp_path/"dataset.csv"
    target.write_bytes(b"local data")
    monkeypatch.setattr(sys,"argv",["fetch_dataset","--output",str(target)])
    with pytest.raises(SystemExit) as error:
        fetch_dataset.main()
    assert error.value.code==2
    assert target.read_bytes()==b"local data"
