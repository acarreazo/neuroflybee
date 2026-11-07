"""Casos límite y ramas de error (v25).

El resto de la suite prueba el camino feliz. Aquí se fuerzan las entradas raras: semillas
fuera de toda partición, escenarios sin ruta factible, métricas con una sola clase, estadística
sin datos y configuraciones inválidas. Cada prueba cubre una rama que antes nadie ejecutaba
(se detectó con ``pytest --cov=neuroflybee --cov-report=term-missing``).
"""
import csv
import math

import numpy as np
import pytest
import yaml

from neuroflybee.config import ProjectConfig, load_config
from neuroflybee.control import registry
from neuroflybee.control.estimation import PositionEstimator
from neuroflybee.domain.geometry import distance
from neuroflybee.evaluation import significance as sig
from neuroflybee.evaluation.benchmark import wilson_interval
from neuroflybee.evaluation.closed_loop import summarize
from neuroflybee.evaluation.metrics import classification_report, roc_auc_binary, roc_auc_ovr_macro
from neuroflybee.models.bee_spec import BeeNetConfig, load_bee_config
from neuroflybee.models.spec import load_fly_config
from neuroflybee.sim.grid import all_reachable, astar, reachable
from neuroflybee.sim.scenarios import (SPLIT_RANGES, ScenarioGenerator, Split, seeds_for,
                                       split_of)
from neuroflybee.training.config import load_train_config
from neuroflybee.training.dagger_config import load_dagger_config
from neuroflybee.utils.seeding import make_rng, set_global_seed


@pytest.fixture()
def gen() -> ScenarioGenerator:
    return ScenarioGenerator(ProjectConfig())


# ------------------------------------------------------------------ semillas y particiones
@pytest.mark.parametrize("seed", [-1, 11_000, 19_999, 21_000, 10 ** 9])
def test_split_of_rejects_seeds_outside_every_split(seed):
    with pytest.raises(ValueError, match="fuera de toda partición"):
        split_of(seed)


@pytest.mark.parametrize("split", list(SPLIT_RANGES))
def test_split_of_first_and_last_seed_of_each_split(split):
    rng = SPLIT_RANGES[split]
    assert split_of(rng[0]) == split
    assert split_of(rng[-1]) == split


def test_seeds_for_edge_cases():
    assert seeds_for(Split.TEST, 0) == []
    full = seeds_for(Split.VAL, len(SPLIT_RANGES[Split.VAL]))
    assert full[0] == SPLIT_RANGES[Split.VAL][0] and full[-1] == SPLIT_RANGES[Split.VAL][-1]
    with pytest.raises(ValueError):
        seeds_for(Split.VAL, len(SPLIT_RANGES[Split.VAL]) + 1)
    with pytest.raises(ValueError):
        seeds_for(Split.TEST, 5, offset=len(SPLIT_RANGES[Split.TEST]) - 4)


def test_seeds_for_rejects_unknown_split():
    with pytest.raises(ValueError):
        seeds_for("no_existe", 3)


# ------------------------------------------------------------------ escenarios sin ruta
def test_scenario_knows_its_split(gen):
    assert gen.generate("nominal", 20_000).split == Split.TEST
    assert gen.generate("nominal", 0).split == Split.TRAIN


def test_unknown_family_is_a_clear_error(gen):
    with pytest.raises(KeyError, match="Familia desconocida"):
        gen.generate("lluvia_de_ranas", 20_000)


def test_generate_raises_when_no_feasible_world(gen, monkeypatch):
    monkeypatch.setattr(gen, "_try_world", lambda scfg, rng: None)
    with pytest.raises(RuntimeError, match="No se pudo generar"):
        gen.generate("nominal", 20_000)


def test_waypoints_impossible_when_none_can_be_far_from_base(gen, monkeypatch):
    # Ningún punto puede quedar a 1e6 m de la base: no hay ruta posible.
    monkeypatch.setattr(ScenarioGenerator, "WAYPOINT_MIN_FROM_BASE", 1e6)
    scfg = gen.family_config("nominal")
    assert gen._place_waypoints(scfg, np.random.default_rng(0), np.array([30.0, 30.0])) is None
    # y _try_world lo propaga como «no factible» (None) en vez de fallar a medias
    assert gen._try_world(scfg, np.random.default_rng(0)) is None
    with pytest.raises(RuntimeError):
        gen.generate("nominal", 20_000)


def test_try_world_returns_none_when_route_is_blocked(gen, monkeypatch):
    monkeypatch.setattr(gen, "_is_feasible", lambda base, waypoints, obstacles: False)
    assert gen._try_world(gen.family_config("nominal"), np.random.default_rng(1)) is None


# ------------------------------------------------------------------ cuadrícula
def test_reachable_from_an_occupied_cell_is_empty():
    occ = np.zeros((4, 4), bool)
    occ[1, 1] = True
    assert not reachable(occ, (1, 1)).any()


def test_astar_edge_cases():
    free = np.zeros((5, 5), bool)
    assert astar(free, (0, 0), (0, 0)) == [(0, 0)]           # ya está en la meta
    blocked_goal = free.copy()
    blocked_goal[4, 4] = True
    assert astar(blocked_goal, (0, 0), (4, 4)) is None        # meta ocupada
    blocked_start = free.copy()
    blocked_start[0, 0] = True
    assert astar(blocked_start, (0, 0), (4, 4)) is None       # salida ocupada
    wall = free.copy()
    wall[:, 2] = True                                         # muro de lado a lado
    assert astar(wall, (0, 0), (4, 4)) is None
    assert not all_reachable(wall, (0, 0), [(0, 1), (4, 4)])
    assert all_reachable(free, (0, 0), [(4, 4), (0, 4)])


# ------------------------------------------------------------------ métricas con una sola clase
def test_report_with_a_single_class_present():
    y = np.zeros(5, dtype=int)
    rep = classification_report(y, y, 4)
    assert rep.accuracy == 1.0
    assert rep.f1_macro == 1.0            # sólo cuentan las clases que aparecen
    assert rep.support == (5, 0, 0, 0)


def test_report_when_everything_is_wrong():
    rep = classification_report(np.zeros(5, dtype=int), np.ones(5, dtype=int), 4)
    assert rep.accuracy == 0.0
    assert rep.f1_macro == 0.0


def test_report_as_dict_has_every_field():
    d = classification_report(np.array([0, 1]), np.array([0, 1]), 2).as_dict()
    assert set(d) == {"accuracy", "f1_macro", "f1_per_class", "support"}


def test_auroc_is_nan_without_both_classes():
    scores = np.array([0.1, 0.9, 0.5])
    assert math.isnan(roc_auc_binary(np.array([True, True, True]), scores))
    assert math.isnan(roc_auc_binary(np.array([False, False, False]), scores))


def test_auroc_with_constant_scores_is_one_half():
    assert roc_auc_binary(np.array([1, 1, 0, 0]), np.full(4, 0.5)) == pytest.approx(0.5)


def test_auroc_macro_with_a_single_class_is_nan():
    y = np.zeros(6, dtype=int)
    probs = np.full((6, 3), 1 / 3)
    assert math.isnan(roc_auc_ovr_macro(y, probs, 3))


# ------------------------------------------------------------------ estadística
def test_wilson_interval_edges():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0 and 0.0 < hi < 0.2
    lo, hi = wilson_interval(30, 30)
    assert hi == 1.0 and 0.8 < lo < 1.0
    lo, hi = wilson_interval(15, 30)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_exact_sign_test_branches():
    assert sig._exact_sign_test(np.zeros(5)) == 1.0               # sin discordantes
    assert sig._exact_sign_test(np.array([1.0, -1.0])) == 1.0     # empate total
    assert sig._exact_sign_test(np.ones(4)) == pytest.approx(2 / 16)


def test_holm_and_cohen_h_edges():
    assert sig.holm([]) == []
    assert sig.holm([0.01]) == [0.01]
    adjusted = sig.holm([0.04, 0.01, 0.03])
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert sig.cohen_h(0.5, 0.5) == 0.0
    assert sig.cohen_h(0.8, 0.3) == pytest.approx(-sig.cohen_h(0.3, 0.8))


def test_compare_without_common_scenarios_returns_none():
    a = {("nominal", 20_000): 1.0}
    b = {("wind", 20_001): 0.0}                                   # ningún escenario en común
    comp = sig.Comparison("g", "a", "b")
    cfg = sig.SignificanceConfig(n_bootstrap=20, n_permutations=20)
    assert sig.compare(a, b, comp, sig.TOTAL, cfg, np.random.default_rng(0)) is None
    sa, sb = sig.paired_differences(a, b, "nominal")
    assert len(sa) == 0 and len(sb) == 0


def test_run_significance_without_data_reports_skipped():
    cfg = sig.SignificanceConfig(n_bootstrap=10, n_permutations=10,
                                 comparisons=(sig.Comparison("g", "a", "b"),))
    report = sig.run_significance([], cfg)
    assert report.results == []
    assert report.skipped == ["g: a vs b (total)"]


def test_significance_config_rejects_bad_macro_prefix(tmp_path):
    path = tmp_path / "sig.yaml"
    path.write_text(yaml.safe_dump({"comparaciones": [{"grupo": "g", "a": "x", "b": "y", "macro": "a1"}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="sólo pueden tener letras"):
        sig.load_significance_config(path)


def test_significance_config_empty_file_uses_defaults(tmp_path):
    path = tmp_path / "vacio.yaml"
    path.write_text("", encoding="utf-8")
    cfg = sig.load_significance_config(path)
    assert cfg.comparisons == ()
    assert cfg.n_bootstrap == sig.SignificanceConfig().n_bootstrap


def test_load_episodes_reads_every_row(tmp_path):
    path = tmp_path / "ep.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["grupo", "etiqueta", "condicion", "semilla_escenario", "exito"])
        w.writeheader()
        w.writerow({"grupo": "g", "etiqueta": "a", "condicion": "nominal", "semilla_escenario": 20_000, "exito": 1})
    rows = sig.load_episodes(path)
    assert len(rows) == 1
    assert sig.scenario_outcomes(rows, "g", "a") == {("nominal", 20_000): 1.0}
    assert sig.scenario_outcomes(rows, "g", "otro") == {}


# ------------------------------------------------------------------ configuraciones
def test_default_and_empty_configs(tmp_path):
    assert load_config(None) == ProjectConfig()
    path = tmp_path / "vacio.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == ProjectConfig()


def test_config_override_keeps_the_other_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"world": {"width": 123.0}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.world.width == 123.0
    assert cfg.world.height == ProjectConfig().world.height


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_existe.yaml")


def test_train_config_none_and_yaml(tmp_path):
    assert load_train_config(None).epochs == 15
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"epochs": 2, "seeds": [5, 6]}), encoding="utf-8")
    cfg = load_train_config(path)
    assert cfg.epochs == 2 and cfg.seeds == (5, 6)         # la lista del YAML pasa a tupla
    assert cfg.batch_size == load_train_config(None).batch_size


def test_other_model_configs_default_when_no_path():
    assert load_dagger_config(None) is not None
    assert load_fly_config(None) is not None
    assert load_bee_config(None) == BeeNetConfig()


@pytest.mark.parametrize("kwargs", [{"fusion": "inventada"}, {"head": "inventada"}, {"bins": 70}])
def test_bee_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BeeNetConfig(**kwargs)


# ------------------------------------------------------------------ utilidades sueltas
def test_distance_accepts_plain_lists():
    assert distance([0, 0], [3, 4]) == 5.0


def test_relocalize_blends_with_the_current_estimate():
    est = PositionEstimator()
    est.position = np.array([0.0, 0.0])
    est.relocalize(np.array([10.0, 0.0]), weight=0.5)
    assert est.position == pytest.approx([5.0, 0.0])
    est.relocalize(np.array([1.0, 1.0]))                       # peso por defecto = 1: reemplaza
    assert est.position == pytest.approx([1.0, 1.0])


def test_registry_unknown_controller_lists_the_options():
    with pytest.raises(KeyError, match="Disponibles"):
        registry.create("controlador_que_no_existe")


def test_summarize_of_no_episodes_is_empty():
    assert summarize([]) == {}


def test_global_seed_makes_runs_repeatable():
    import random

    set_global_seed(123)
    a = (random.random(), float(np.random.rand()))
    set_global_seed(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b
    assert make_rng(7).random() == make_rng(7).random()
    assert make_rng(7).random() != make_rng(8).random()


# ------------------------------------------------------------------ entrenamiento (necesita PyTorch)
def test_trainer_rejects_unknown_mode():
    torch = pytest.importorskip("torch")
    from neuroflybee.training.trainer import Trainer

    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="mode"):
        Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), torch.nn.MSELoss(),
                lambda *a: {}, "score", torch.device("cpu"), mode="medio")


def test_resolve_device():
    torch = pytest.importorskip("torch")
    from neuroflybee.training.trainer import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_fly_loader_rejects_a_checkpoint_of_another_kind(tmp_path):
    torch = pytest.importorskip("torch")
    from neuroflybee.training.checkpoints import load_fly_checkpoint

    path = tmp_path / "abeja.pt"
    torch.save({"kind": "bee"}, path)
    with pytest.raises(ValueError, match="no es un checkpoint de la mosca"):
        load_fly_checkpoint(path)
