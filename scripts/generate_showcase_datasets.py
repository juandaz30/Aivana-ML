"""Utility script to create near-ideal demo datasets for every built-in model.

Each dataset mirrors a realistic scenario while remaining clean, perfectly
separable (when appropriate) and numerically stable so the algorithms in this
project can shine during demos.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DemoDataset:
    filename: str
    model: str
    description: str
    columns: List[str]
    rows: List[List[float]]


def _linear_regression_rows() -> List[List[float]]:
    hours = list(range(1, 21)) + [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    return [[float(h), float(12 * h + 40)] for h in hours]


def _logistic_regression_rows() -> List[List[float]]:
    at_risk = [
        (3200, 1.1),
        (3600, 1.2),
        (3800, 1.3),
        (4200, 1.4),
        (4500, 1.5),
        (5000, 1.6),
        (5400, 1.7),
        (5800, 1.8),
        (6000, 1.9),
        (6200, 1.95),
        (6400, 2.0),
        (6600, 2.05),
        (6800, 2.1),
        (6900, 2.15),
        (7000, 2.2),
    ]
    protective = [
        (7200, 2.2),
        (7600, 2.25),
        (8000, 2.3),
        (8400, 2.35),
        (8800, 2.4),
        (9200, 2.5),
        (9600, 2.6),
        (10000, 2.7),
        (10400, 2.8),
        (10800, 2.9),
        (11200, 3.0),
        (11600, 3.1),
        (12000, 3.2),
        (12400, 3.3),
        (12800, 3.4),
    ]
    rows: List[List[float]] = []
    rows.extend([[steps, water, 1] for steps, water in at_risk])
    rows.extend([[steps, water, 0] for steps, water in protective])
    return rows


def _perceptron_rows() -> List[List[float]]:
    base_pattern = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ]
    # Replicate the AND pattern over many cycles to simulate sensor logs.
    return [row[:] for _ in range(12) for row in base_pattern]


def _decision_tree_rows() -> List[List[float]]:
    rows: List[List[float]] = []
    income_levels = range(48, 101, 4)
    debt_ratios = range(20, 51, 5)
    for income_k, debt_ratio in product(income_levels, debt_ratios):
        credit_history = max(2, min(15, round(income_k / 10 + 2)))
        approve = 1 if debt_ratio < 33 or income_k >= 88 else 0
        rows.append([float(income_k), float(debt_ratio), float(credit_history), approve])
    return rows


def _naive_bayes_rows() -> List[List[float]]:
    rainy_profiles = [
        (17, 93, 8),
        (18, 91, 9),
        (19, 90, 10),
        (20, 88, 11),
        (21, 86, 12),
        (18, 89, 9),
        (19, 92, 10),
        (20, 87, 12),
        (17, 94, 8),
        (22, 85, 13),
        (16, 95, 7),
        (18, 93, 9),
        (19, 88, 11),
        (21, 90, 12),
        (22, 89, 13),
    ]
    sunny_profiles = [
        (26, 48, 18),
        (27, 46, 19),
        (28, 44, 20),
        (29, 42, 21),
        (30, 40, 22),
        (31, 38, 23),
        (32, 36, 24),
        (33, 34, 25),
        (27, 50, 19),
        (28, 48, 20),
        (29, 46, 21),
        (30, 44, 22),
        (31, 42, 23),
        (32, 40, 24),
        (33, 38, 25),
    ]
    rows: List[List[float]] = []
    rows.extend([[temp, humidity, wind, 1] for temp, humidity, wind in rainy_profiles])
    rows.extend([[temp, humidity, wind, 0] for temp, humidity, wind in sunny_profiles])
    return rows


def _mlp_rows() -> List[List[float]]:
    pattern = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    return [row[:] for _ in range(15) for row in pattern]


def _kmeans_rows() -> List[List[float]]:
    clusters = [
        (2.0, 2.1),
        (2.2, 1.9),
        (1.8, 2.3),
        (2.4, 2.2),
        (1.9, 1.8),
        (7.8, 8.4),
        (8.2, 7.9),
        (7.5, 8.1),
        (8.4, 8.2),
        (7.7, 7.6),
        (3.0, 9.2),
        (2.8, 8.8),
        (3.2, 9.0),
        (2.6, 9.1),
        (3.4, 8.7),
    ]
    extra = [
        (2.1, 2.4),
        (1.7, 2.0),
        (7.9, 8.6),
        (8.5, 7.7),
        (3.3, 9.4),
        (2.5, 8.6),
        (8.1, 8.3),
        (2.2, 8.9),
        (7.6, 8.0),
        (3.5, 9.1),
    ]
    all_points = clusters + extra
    return [[x, y] for x, y in all_points]


def _pca_rows() -> List[List[float]]:
    rows: List[List[float]] = []
    for base in range(10, 61, 2):
        sensor_a = float(base)
        sensor_b = round(0.8 * base + 0.2, 2)
        sensor_c = round(0.5 * base + 0.3 * sensor_b, 2)
        sensor_d = round(1.2 * base + 0.5, 2)
        rows.append([sensor_a, sensor_b, sensor_c, sensor_d])
    return rows


DATASETS: List[DemoDataset] = [
    DemoDataset(
        filename="linear_regression_study_hours.csv",
        model="LinearRegression",
        description=(
            "Notas de examen que crecen de forma perfectamente lineal con las horas "
            "de práctica. Ideal para mostrar una relación causa-efecto sencilla."
        ),
        columns=["study_hours", "exam_score"],
        rows=_linear_regression_rows(),
    ),
    DemoDataset(
        filename="logistic_regression_heart_risk.csv",
        model="LogisticRegression",
        description=(
            "Riesgo de salud (0 = bajo, 1 = alto) definido por hábitos diarios. Las "
            "clases son linealmente separables combinando pasos diarios y agua bebida."
        ),
        columns=["daily_steps", "water_intake_liters", "heart_risk"],
        rows=_logistic_regression_rows(),
    ),
    DemoDataset(
        filename="perceptron_safety_gate.csv",
        model="Perceptron",
        description=(
            "Puerta lógica AND para una línea de producción: la salida solo es segura "
            "cuando ambos sensores leen condiciones correctas."
        ),
        columns=["door_closed", "safety_light_on", "line_is_safe"],
        rows=_perceptron_rows(),
    ),
    DemoDataset(
        filename="decision_tree_loan_screening.csv",
        model="DecisionTreeClassifier",
        description=(
            "Evaluación crediticia con una regla clara: aprobar cuando la deuda es "
            "baja o el ingreso es muy alto. El árbol aprende la lógica exacta."
        ),
        columns=["income_k", "debt_ratio", "credit_history_years", "loan_approved"],
        rows=_decision_tree_rows(),
    ),
    DemoDataset(
        filename="naive_bayes_weather.csv",
        model="NaiveBayes",
        description=(
            "Pronóstico binario de lluvia con distribuciones claramente distintas de "
            "temperatura, humedad y viento para cada clase."
        ),
        columns=["temperature_c", "humidity_percent", "wind_speed_kmh", "will_rain"],
        rows=_naive_bayes_rows(),
    ),
    DemoDataset(
        filename="mlp_factory_alarm.csv",
        model="MLPClassifier",
        description=(
            "Patrón XOR (activación de alarma solo cuando exactamente un sensor "
            "detecta un obstáculo). Se necesita una capa oculta para aprenderlo."
        ),
        columns=["left_sensor_triggered", "right_sensor_triggered", "alarm_on"],
        rows=_mlp_rows(),
    ),
    DemoDataset(
        filename="kmeans_store_zones.csv",
        model="KMeans",
        description=(
            "Ubicaciones de clientes agrupadas en tres zonas geográficas bien "
            "separadas para un plan logístico."
        ),
        columns=["x_coord", "y_coord"],
        rows=_kmeans_rows(),
    ),
    DemoDataset(
        filename="pca_wearable_sensors.csv",
        model="PCA",
        description=(
            "Mediciones de un wearable donde casi todas las columnas son combinaciones "
            "lineales de un mismo esfuerzo físico. PCA recupera las componentes reales."
        ),
        columns=["sensor_a", "sensor_b", "sensor_c", "sensor_d"],
        rows=_pca_rows(),
    ),
]


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        target_file = output_dir / ds.filename
        with target_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ds.columns)
            writer.writerows(ds.rows)
        print(f"✔ Generated {ds.filename} for {ds.model}: {ds.description}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "backend" / "datasets" / "showcase"
    run(output_dir)


if __name__ == "__main__":
    main()
