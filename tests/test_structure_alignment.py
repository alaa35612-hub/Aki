import importlib.util
import pathlib

import pytest


def _load_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "حبيبي ٢.py"
    spec = importlib.util.spec_from_file_location("habibi2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


mod = _load_module()


def test_fix_structure_after_bos_marks_latest_slots():
    bk = mod.DigitalBookkeeping()
    bk.arrBCLabel = [1, 2, 3]
    bk.arrBCLine = [4, 5, 6]
    bk.arrIdmLabel = [7, 8, 9]
    bk.arrIdmLine = [10, 11, 12]
    bk.arrHLLabel = [13, 14, 15]
    bk.arrHLCircle = [16, 17, 18]

    bk.fixStrcAfterBos()

    assert bk.arrBCLabel[-1] is None
    assert bk.arrBCLine[-1] is None
    assert bk.arrIdmLabel[-1] is None
    assert bk.arrIdmLine[-1] is None
    assert bk.arrHLLabel[-1] is None and bk.arrHLLabel[-2] is None
    assert bk.arrHLCircle[-1] is None and bk.arrHLCircle[-2] is None


def test_fix_structure_after_choch_matches_pine_offsets():
    bk = mod.DigitalBookkeeping()
    bk.arrBCLabel = [1, 2, 3, 4]
    bk.arrBCLine = [5, 6, 7, 8]
    bk.arrIdmLabel = [9, 10, 11, 12]
    bk.arrIdmLine = [13, 14, 15, 16]
    bk.arrHLLabel = [17, 18, 19, 20]
    bk.arrHLCircle = [21, 22, 23, 24]

    bk.fixStrcAfterChoch()

    assert bk.arrBCLabel[-1] is None and bk.arrBCLabel[-2] is None
    assert bk.arrBCLine[-1] is None and bk.arrBCLine[-2] is None
    assert bk.arrIdmLabel[-2] is None
    assert bk.arrIdmLine[-2] is None
    # nth deletions keep length but blank out indices 2 and 3 from the tail
    assert bk.arrHLLabel[-2] is None and bk.arrHLLabel[-3] is None
    assert bk.arrHLCircle[-2] is None and bk.arrHLCircle[-3] is None


def test_idm_take_after_bos_resets_last_l_and_applies_bos_fix():
    cfg = mod.Settings(structure_type="Choch with IDM", mitigation_mode="WICK")
    ind = mod.SMCIndicator(cfg)
    ind.findIDM = True
    ind.isCocUp = True
    ind.isBosUp = True
    ind.isPrevBos = True
    ind.idmLow = 100.0
    ind.lastL = 100.0
    ind.lastLBar = 40
    ind.H = 150.0
    ind.HBar = 90

    ind.bk.arrLastL = [80.0, 95.0]
    ind.bk.arrLastLBar = [8, 9]
    ind.bk.arrBCLabel = [1, 2, 3]
    ind.bk.arrBCLine = [4, 5, 6]
    ind.bk.arrIdmLabel = [7, 8, 9]
    ind.bk.arrIdmLine = [10, 11, 12]
    ind.bk.arrHLLabel = [13, 14, 15]
    ind.bk.arrHLCircle = [16, 17, 18]

    ind._activate_idm_from_candidates = lambda trend_up: None

    ind._idm_take_unlock(True, hi=140.0, lo=90.0, cl=95.0, i=5)

    assert not ind.findIDM and not ind.isBosUp
    assert ind.lastH == pytest.approx(ind.H)
    assert ind.lastL == pytest.approx(95.0)
    assert ind.lastLBar == 9
    assert ind.bk.arrHLLabel[-1] is None and ind.bk.arrHLLabel[-2] is None
    assert ind.bk.arrHLCircle[-1] is None and ind.bk.arrHLCircle[-2] is None


def test_idm_take_after_choch_uses_choch_fix_without_anchor_reset():
    cfg = mod.Settings(structure_type="Choch with IDM", mitigation_mode="WICK")
    ind = mod.SMCIndicator(cfg)
    ind.findIDM = True
    ind.isCocUp = True
    ind.isBosUp = True
    ind.isPrevBos = False
    ind.idmLow = 50.0
    ind.lastL = 50.0
    ind.lastLBar = 2

    ind.bk.arrBCLabel = [1, 2, 3, 4]
    ind.bk.arrBCLine = [5, 6, 7, 8]
    ind.bk.arrIdmLabel = [9, 10, 11, 12]
    ind.bk.arrIdmLine = [13, 14, 15, 16]
    ind.bk.arrHLLabel = [17, 18, 19, 20]
    ind.bk.arrHLCircle = [21, 22, 23, 24]

    ind._activate_idm_from_candidates = lambda trend_up: None

    ind._idm_take_unlock(True, hi=70.0, lo=40.0, cl=45.0, i=6)

    assert ind.lastL == 50.0  # لا يتم إرجاعه لأن isPrevBos = False
    assert ind.bk.arrBCLabel[-1] is None and ind.bk.arrBCLabel[-2] is None
    assert ind.bk.arrBCLine[-1] is None and ind.bk.arrBCLine[-2] is None
    assert ind.bk.arrHLLabel[-2] is None and ind.bk.arrHLLabel[-3] is None
    assert ind.bk.arrHLCircle[-2] is None and ind.bk.arrHLCircle[-3] is None


def test_idm_take_downtrend_resets_last_h_after_prev_bos():
    cfg = mod.Settings(structure_type="Choch with IDM", mitigation_mode="WICK")
    ind = mod.SMCIndicator(cfg)
    ind.findIDM = True
    ind.isCocDn = True
    ind.isBosDn = True
    ind.isPrevBos = True
    ind.idmHigh = 200.0
    ind.lastH = 200.0
    ind.lastHBar = 30
    ind.L = 120.0
    ind.LBar = 11

    ind.bk.arrLastH = [150.0, 180.0]
    ind.bk.arrLastHBar = [15, 18]
    ind.bk.arrBCLabel = [1, 2, 3]
    ind.bk.arrBCLine = [4, 5, 6]
    ind.bk.arrIdmLabel = [7, 8, 9]
    ind.bk.arrIdmLine = [10, 11, 12]
    ind.bk.arrHLLabel = [13, 14, 15]
    ind.bk.arrHLCircle = [16, 17, 18]

    ind._activate_idm_from_candidates = lambda trend_up: None

    ind._idm_take_unlock(False, hi=210.0, lo=190.0, cl=205.0, i=7)

    assert not ind.findIDM and not ind.isBosDn
    assert ind.lastL == pytest.approx(ind.L)
    assert ind.lastLBar == ind.LBar
    assert ind.lastH == pytest.approx(180.0)
    assert ind.lastHBar == 18
    assert ind.bk.arrHLLabel[-1] is None and ind.bk.arrHLLabel[-2] is None
    assert ind.bk.arrHLCircle[-1] is None and ind.bk.arrHLCircle[-2] is None
