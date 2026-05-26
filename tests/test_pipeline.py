from pathlib import Path

import pytest

from emsaplibraries.indicators import ProteinMetricsResult
from emsaplibraries.pipeline import ProteinPipelineResult, process_single_protein


def test_pipeline_result_legacy_tuple_formats_metrics():
    result = ProteinPipelineResult(
        protein_name="protein",
        p_sasa=1.234,
        q_sasa=-2.345,
        ecpi_percent=50.0,
        see=3.456,
        hse=0.125,
        pka_sasa=7.891,
        abse=-0.444,
        pdb_file=Path("protein.pdb"),
        pqr_file=Path("protein.pqr"),
        dx_file=Path("protein.dx"),
        apbs_input_file=Path("protein.in"),
        apbs_log_file=Path("protein.out"),
    )

    assert result.as_legacy_tuple() == (
        "protein",
        "1.23",
        "-2.35",
        "50.00",
        "3.46",
        "0.12",
        "7.89",
        "-0.44",
    )
    assert isinstance(result.p_sasa, float)


def test_process_single_protein_checks_missing_pdb2pqr(monkeypatch, tmp_path):
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="pdb2pqr"):
        process_single_protein(
            tmp_path / "protein.pdb",
            tmp_path / "aux",
            [0, 0, 0],
            [1, 1, 1],
        )


def test_process_single_protein_checks_missing_apbs(monkeypatch, tmp_path):
    def fake_which(name):
        return "/usr/bin/pdb2pqr" if name == "pdb2pqr" else None

    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", fake_which)

    with pytest.raises(RuntimeError, match="apbs"):
        process_single_protein(
            tmp_path / "protein.pdb",
            tmp_path / "aux",
            [0, 0, 0],
            [1, 1, 1],
        )


def test_process_single_protein_checks_missing_propka(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_which(name):
        return name if name in {"pdb2pqr", "apbs"} else None

    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", fake_which)
    monkeypatch.setattr(
        "emsaplibraries.pipeline.subprocess.run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "emsaplibraries.pipeline.generate_apbs_in_fixed",
        lambda *args, **kwargs: "protein.in",
    )
    monkeypatch.setattr(
        "emsaplibraries.pipeline.find_dx_file",
        lambda name: tmp_path / "protein.dx",
    )
    with pytest.raises(RuntimeError, match="propka3"):
        process_single_protein(
            tmp_path / "protein.pdb",
            tmp_path / "aux",
            [0, 0, 0],
            [1, 1, 1],
        )


def test_process_single_protein_subprocess_boundary(monkeypatch, tmp_path):
    calls = []
    pdb_file = tmp_path / "protein.pdb"
    pdb_file.write_text("HEADER\n", encoding="utf-8")
    dx_file = tmp_path / "protein.dx"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("emsaplibraries._runtime.shutil.which", lambda name: name)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "pdb2pqr":
            Path(command[-1]).write_text("PQR\n", encoding="utf-8")
        elif command[0] == "apbs":
            dx_file.write_text("DX\n", encoding="utf-8")
        elif command[0] == "propka3":
            Path("protein.pka").write_text("PKA\n", encoding="utf-8")

    monkeypatch.setattr("emsaplibraries.pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("emsaplibraries.pipeline.find_dx_file", lambda name: dx_file)
    metric_calls = []

    def fake_calculate_protein_metrics(pdb, pqr, dx, pka=None):
        metric_calls.append((pdb, pqr, dx, pka))
        return ProteinMetricsResult(
            protein_name="protein",
            pdb_file=pdb,
            pqr_file=pqr,
            dx_file=dx,
            pka_file=pka,
            p_sasa=1.0,
            q_sasa=2.0,
            ecpi_percent=3.0,
            see=4.0,
            hse=6.0,
            pka_sasa=5.0,
            abse=7.0,
            p_sasa_numerator=0.0,
            p_sasa_denominator=0.0,
            q_sasa_numerator=0.0,
            q_sasa_total_sasa=0.0,
            ecpi_data={"percent_exposed_charge": 3.0},
        )

    monkeypatch.setattr(
        "emsaplibraries.pipeline.calculate_protein_metrics",
        fake_calculate_protein_metrics,
    )

    result = process_single_protein(
        pdb_file,
        tmp_path / "aux",
        [0, 0, 0],
        [1, 1, 1],
    )

    assert calls == [
        ["pdb2pqr", "--ff=PARSE", "--with-ph=7", str(pdb_file), "protein.pqr"],
        ["apbs", "protein.in"],
        ["propka3", str(pdb_file)],
    ]
    assert result.as_legacy_tuple() == (
        "protein",
        "1.00",
        "2.00",
        "3.00",
        "4.00",
        "6.00",
        "5.00",
        "7.00",
    )
    assert result.pqr_file == tmp_path / "aux" / "protein.pqr"
    assert result.dx_file == tmp_path / "aux" / "protein.dx"
    assert result.apbs_input_file == tmp_path / "aux" / "protein.in"
    assert result.apbs_log_file == tmp_path / "aux" / "protein.out"
    assert result.pka_file == tmp_path / "aux" / "protein.pka"
    assert metric_calls == [
        (
            pdb_file,
            Path("protein.pqr"),
            dx_file,
            Path("protein.pka"),
        )
    ]
