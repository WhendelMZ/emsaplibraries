import pytest

from emsaplibraries.electrostatics import run_apbs, run_pdb2pqr
from emsaplibraries.preprocessing import run_mafft


def test_run_mafft_checks_missing_tool(monkeypatch, tmp_path):
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="mafft"):
        run_mafft(tmp_path / "in.fasta", tmp_path / "out.fasta")


def test_run_apbs_checks_missing_tool(monkeypatch):
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="apbs"):
        run_apbs("protein.pdb", "protein", [0, 0, 0], [1, 1, 1])


def test_run_pdb2pqr_checks_missing_tool(monkeypatch):
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="pdb2pqr"):
        run_pdb2pqr("protein.pdb", "protein")
