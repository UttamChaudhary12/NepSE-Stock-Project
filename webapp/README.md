# NEPSE Signals: Django web app

A web app that shows BUY / HOLD / SELL signals for NEPSE stocks from the tuned LSTM ensemble in
`../Uttam_EDA.ipynb` (section 29). A signal is the model's estimate for the **next 5 trading sessions**.

## Run it

```bash
cd webapp
pip install -r requirements.txt
python manage.py migrate
python manage.py build_signals          # score every stock-day (about a minute)
python manage.py runserver
```

Open http://127.0.0.1:8000/.

`build_signals` reads the model from `../improved_results/lstm_tuned.pt` and the cached daily tables
`../improved_results/daily_ohlcv_brokers.parquet` and `broker_flow.parquet`, which the notebook creates.
After adding new trades to `../data/stockpati_floorsheet_full.csv`, run
`python manage.py build_signals --from-csv` to re-aggregate the CSV and score the new sessions.

## Pages

| URL | What it shows |
|---|---|
| `/` | All signals for a date, filterable by signal, sector, liquidity and search, with probabilities and realised outcomes |
| `/stock/<SYMBOL>/` | Price chart coloured by signal, probability history, out-of-sample hit rate, recent signals |
| `/about/` | Model configuration, held-out test metrics and backtest |
| `/api/signals/?date=YYYY-MM-DD` | The same signals as JSON |
| `/admin/` | Django admin (create a user with `python manage.py createsuperuser`) |

## How it works

- `predictor/ml/features.py`: daily aggregation and feature engineering, copied from the notebook so the app
  computes exactly the inputs the model was trained on.
- `predictor/ml/lstm.py`: loads the 5-seed LSTM ensemble, applies the stored training-period scaler, averages
  probabilities, and applies the class bias chosen on validation data.
- `predictor/management/commands/build_signals.py`: scores every stock-day with at least 20 sessions of history and
  stores the results in SQLite, together with the realised 5-session outcome where it is known.

Signals dated before the test start (2026-05-27) are in the training period. The app labels them, because their
hit rates overstate real accuracy.

## Settings

Override these with environment variables: `NEPSE_MODEL_PATH`, `NEPSE_FLOORSHEET_CSV`, `DJANGO_SECRET_KEY`,
`DJANGO_DEBUG=0` and `DJANGO_ALLOWED_HOSTS`. The defaults are for local development only.

## Caveat

This is a research model trained on one year of data. Its held-out macro F1 is modest, and in the test-period backtest
no strategy made money after costs. It is not investment advice.
