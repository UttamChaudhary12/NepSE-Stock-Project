# CNN experiment outputs

Open `CNN_Report.html` in a browser to read the report with embedded code and
result snapshots. It is self-contained and can be printed to PDF.

The implementation and executed outputs are in sections 12–18 of
`../Uttam_EDA.ipynb`. These sections run independently of the original EDA.
Run them from the project folder with Python, PyTorch, pandas, numpy, polars,
matplotlib and scikit-learn installed.

To rerun the experiment and rebuild the report and all snapshots together:

```powershell
python implement_cnn.py
```

This replaces the generated CNN appendix and experiment outputs while preserving
the original EDA cells. It uses `data/stockpati_floorsheet_full.csv`; the small
daily aggregate cache is reused when the source size and modification time match.

- `experiments.csv`: configurations and best validation macro F1.
- `training_history.csv`: per-epoch training and validation measurements.
- `classification_report.csv`: test precision, recall, F1 and support.
- `baseline_comparison.csv`: CNN and always-HOLD test metrics.
- `split_summary.csv`: sequence counts, date ranges and class counts.
- `test_predictions.csv`: stock/date-aligned actual and predicted classes.
- `liquidity_slices.csv`: performance by current-session trade count.
- `stock_cnn.pt`: selected model weights and preprocessing metadata.
- `environment.json`: package versions, seed, device and source signature.
- `*.png`: actual code/result snapshots used in the report.

Sections 12–17 refresh the notebook experiment outputs. Rebuilding the standalone
HTML report and code/table snapshots requires the script command above.
