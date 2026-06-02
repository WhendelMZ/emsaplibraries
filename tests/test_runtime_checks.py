import pytest

from emsaplibraries.external import run_apbs, run_mafft, run_pdb2pqr, run_propka


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


def test_run_propka_checks_missing_tool(monkeypatch):
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="propka3"):
        run_propka("protein.pdb")


def test_run_propka_returns_generated_pka(monkeypatch, tmp_path):
    pdb_file = tmp_path / "protein.pdb"
    pdb_file.write_text("HEADER\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: name)

    def fake_run(command, **kwargs):
        assert command == ["propka3", str(pdb_file)]
        assert kwargs == {"check": True}
        (tmp_path / "protein.pka").write_text("PKA\n", encoding="utf-8")

    monkeypatch.setattr("emsaplibraries.external.subprocess.run", fake_run)

    assert run_propka(pdb_file) == tmp_path.joinpath("protein.pka").relative_to(
        tmp_path
    )
