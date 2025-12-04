This project benchmarks multiple statistical, machine-learning and deep-learning models to forecast Alpine snow depth using historical snow-depth data only (no exogenous features).
It provides both a full training pipeline and a PyQt6 GUI application for clean comparison across models and stations.

The dataset used is the SAFRAN–CROCUS snowpack model output (1975–2020) for four Alpine stations:
Tignes, Les 2 Alpes, Serre Chevalier, Col de Porte.

Dependencies : 
- core : pandas, pathlib, os, pyqt6, numpy
- models : matplotlib.pyplot, statsmodels, prophet, xgboost

How to run : 
1. Use default dataset or add more in `/cleaned v1/`, then run `/utils/cleaner.py` if you add yours.
2. run `/app.py` for GUI, `/main.py` for CLI.

Kaggle : use the provided drag-and-drop ready files in `/kaggle/`, or :
https://www.kaggle.com/work/collections/17079326

Contact : pablo.ferreiraa10@gmail.com - https://www.linkedin.com/in/pablo-frr/
