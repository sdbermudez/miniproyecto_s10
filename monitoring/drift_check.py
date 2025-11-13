from evidently import Report
from evidently.presets import DataDriftPreset

import pandas as pd

# Carga datasets de ejemplo
ref = pd.read_csv("data/heart_train.csv")
cur = pd.read_csv("data/heart_test.csv")

# Genera reporte
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("reports/drift_report.html")

print("Drift report generado: reports/drift_report.html")
