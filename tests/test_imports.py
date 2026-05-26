def test_top_level_imports():
    import emsaplibraries

    assert callable(emsaplibraries.calculate_q_sasa)
    assert callable(emsaplibraries.calculate_p_sasa)
    assert callable(emsaplibraries.calculate_see)
    assert callable(emsaplibraries.calculate_protein_metrics)
    assert emsaplibraries.ProteinMetricsResult
    assert not hasattr(emsaplibraries, "process_single_protein")
    assert not hasattr(emsaplibraries, "ProteinPipelineResult")


def test_public_submodule_imports():
    import emsaplibraries.electrostatics
    import emsaplibraries.external
    import emsaplibraries.indicators
    import emsaplibraries.pipeline
    import emsaplibraries.preprocessing
    import emsaplibraries.structure

    assert emsaplibraries.indicators.parse_pqr
    assert not hasattr(emsaplibraries.indicators, "process_single_protein")
    assert emsaplibraries.pipeline.process_single_protein
    assert emsaplibraries.pipeline.ProteinPipelineResult
    assert not hasattr(emsaplibraries.electrostatics, "run_apbs")
    assert not hasattr(emsaplibraries.electrostatics, "run_pdb2pqr")
    assert not hasattr(emsaplibraries.preprocessing, "run_mafft")
    assert emsaplibraries.external.run_apbs
    assert emsaplibraries.external.run_pdb2pqr
    assert emsaplibraries.external.run_mafft
    assert emsaplibraries.external.run_propka
    assert emsaplibraries.structure.cif_to_pdb
