def test_top_level_imports():
    import emsaplibraries

    assert callable(emsaplibraries.calculate_q_sasa)
    assert callable(emsaplibraries.calculate_p_sasa)
    assert callable(emsaplibraries.calculate_see)


def test_public_submodule_imports():
    import emsaplibraries.electrostatics
    import emsaplibraries.indicators
    import emsaplibraries.preprocessing
    import emsaplibraries.structure

    assert emsaplibraries.indicators.parse_pqr
    assert emsaplibraries.electrostatics.run_apbs
    assert emsaplibraries.preprocessing.run_mafft
    assert emsaplibraries.structure.cif_to_pdb
