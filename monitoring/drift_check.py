from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import os

# Crear carpeta de salida
os.makedirs("reports", exist_ok=True)

# Cargar datos
ref = pd.read_csv("data/heart_train.csv")
cur = pd.read_csv("data/heart_test.csv")

# Crear reporte de deriva
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# Guardar reporte HTML (nuevo método en Evidently 0.7)
html = report.as_html()
with open("reports/drift_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Reporte de deriva generado: reports/drift_report.html")
