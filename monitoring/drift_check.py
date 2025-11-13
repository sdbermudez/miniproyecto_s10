from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import os

# Crear carpeta de salida
os.makedirs("reports", exist_ok=True)

# Cargar datos
ref = pd.read_csv("data/cleaned/heart_train.csv")
cur = pd.read_csv("data/cleaned/heart_test.csv")

# Generar reporte
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# Guardar salida HTML manualmente
# Evidently >=0.7 solo genera JSON o notebooks, así que convertimos el JSON a HTML básico
report_json = report.json()

html_output = f"""
<html>
<head><title>Data Drift Report</title></head>
<body>
<h1>Data Drift Report</h1>
<pre>{report_json}</pre>
</body>
</html>
"""

with open("reports/drift_report.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("Reporte de deriva generado: reports/drift_report.html")
