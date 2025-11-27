# COMPAS Bias Audit - Deliverables

Files in this folder:
- audit_compas.py        : Main script to load COMPAS, train a model, compute fairness metrics, and save plots.
- utils.py               : Helper functions for plotting and report generation.
- report.txt             : 300-word summary report (generated content).
- requirements.txt       : Python dependencies.

Notes:
1. AIF360 (aif360) can be difficult to install due to optional dependencies. If installation fails, refer to AIF360 docs: https://aif360.mybluemix.net/ (or the project's GitHub).
2. To run locally:
   - Create a virtual environment: python3 -m venv venv
   - Activate: source venv/bin/activate
   - Install: pip install -r requirements.txt
   - Run: python audit_compas.py
3. If you encounter AIF360 install issues, you can still run the script by:
   - Replacing AIF360 dataset loading with a CSV copy of COMPAS and adjusting dataset wrappers.
   - Or run inside a Docker image with preinstalled AIF360.

