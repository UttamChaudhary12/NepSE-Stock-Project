"""Score every stock-day with the tuned LSTM ensemble and store the signals.

    python manage.py build_signals              # use the cached daily parquet files
    python manage.py build_signals --from-csv   # re-aggregate the floorsheet CSV first
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from predictor.ml.features import aggregate_brokers, aggregate_daily, build_features
from predictor.ml.lstm import Ensemble
from predictor.models import ModelRun, Signal, Stock

CLASSES = np.array(['SELL', 'HOLD', 'BUY'])


def _csv(path):
    return pd.read_csv(path).to_dict('records') if Path(path).exists() else []


class Command(BaseCommand):
    help = 'Compute LSTM signals for every stock-day and store them in the database.'

    def add_arguments(self, parser):
        parser.add_argument('--from-csv', action='store_true',
                            help=f'Re-aggregate {settings.FLOORSHEET_CSV} instead of the cached parquet')

    def handle(self, *args, from_csv=False, **options):
        start = time.time()
        if from_csv:
            self.stdout.write(f'Aggregating {settings.FLOORSHEET_CSV} (takes a minute)...')
            daily, broker = aggregate_daily(settings.FLOORSHEET_CSV), aggregate_brokers(settings.FLOORSHEET_CSV)
        else:
            daily, broker = pl.read_parquet(settings.DAILY_PARQUET), pl.read_parquet(settings.BROKER_PARQUET)
        model = Ensemble(settings.MODEL_PATH)
        ck = model.ckpt
        frame = build_features(daily, broker, model.sectors, ck['horizon'], ck['k_vol'], ck['min_move'])
        probs, signal = model.predict(frame)
        test_start = pd.Timestamp(ck['test_start'])
        keep = np.flatnonzero(signal >= 0)
        f = frame.iloc[keep]
        self.stdout.write(f'Scored {len(keep):,} stock-days in {time.time() - start:.0f}s; writing database...')

        with transaction.atomic():
            Signal.objects.all().delete()
            Stock.objects.all().delete()
            first = frame.drop_duplicates('symbol', keep='last')
            Stock.objects.bulk_create([Stock(symbol=r.symbol, name=r.company_name, sector=str(r.sector),
                                             promoter=bool(r.promoter)) for r in first.itertuples()])
            known = f.target.to_numpy() >= 0
            rows = [Signal(stock_id=r.symbol, date=r.date.date(), close=r.close,
                           change=None if pd.isna(r.ret_1) else r.ret_1, volume=r.volume,
                           turnover=r.turnover, trade_count=r.trade_count,
                           p_sell=pr[0], p_hold=pr[1], p_buy=pr[2], signal=CLASSES[s],
                           threshold=r.threshold,
                           fwd_return=r.fwd_return if k else None,
                           actual=CLASSES[r.target] if k else '',
                           in_sample=bool(k and r.label_end < test_start))
                    for r, pr, s, k in zip(f.itertuples(), probs[keep], signal[keep], known)]
            Signal.objects.bulk_create(rows, batch_size=5000)
            results = Path(settings.MODEL_PATH).parent
            ModelRun.objects.create(
                model_path=str(settings.MODEL_PATH), data_through=frame.date.max().date(),
                test_start=test_start.date(),
                info={'config': ck['config'], 'epochs': ck['epochs'], 'seeds': ck['seeds'], 'bias': ck['bias'],
                      'n_features': len(ck['features']), 'horizon': ck['horizon'], 'k_vol': ck['k_vol'],
                      'min_move': ck['min_move'], 'val_macro_f1': ck['val_macro_f1'],
                      'test_macro_f1': ck['test_macro_f1'],
                      'comparison': _csv(results/'tuned_comparison.csv'),
                      'backtest': _csv(results/'tuned_backtest.csv')})
        self.stdout.write(self.style.SUCCESS(
            f'Stored {len(rows):,} signals for {len(first)} stocks through {frame.date.max().date()} '
            f'({time.time() - start:.0f}s total).'))
