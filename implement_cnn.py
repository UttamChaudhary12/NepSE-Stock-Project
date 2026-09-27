"""Build and execute the self-contained CNN appendix; preserve the original EDA."""
import base64
import contextlib
import html
import io
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = ROOT / 'cnn_results'
OUT.mkdir(exist_ok=True)
path = ROOT / 'Uttam_EDA.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
original = []
for cell in nb['cells']:
    if '## 12. CNN experiment' in ''.join(cell['source']):
        break
    original.append(cell)
cells = []
def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip().splitlines(True)))
def code(s):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip().splitlines(True)))

md('''
## 12. CNN experiment: next-observed-session classification

This implementation adapts the supplied **FashionMNIST CNN** to the week-2
StockPati floorsheet dataset. The earlier EDA proposed an LSTM; this experiment
instead uses the CNN required by this assignment. A **1D convolution across 20
observed stock sessions** replaces image convolution. The shared structure is
two convolution–ReLU–pool blocks, flatten, dropout, a linear classifier, Adam,
and cross-entropy. There are three output classes rather than ten.

**Run sections 12–18 independently** (the earlier full-CSV pandas EDA is not
required). Dependencies: Python, numpy, pandas, polars, matplotlib,
scikit-learn, and PyTorch. Run from the notebook's folder. Outputs go to
`cnn_results/`. The full source CSV is aggregated without loading all trade
columns into pandas. The daily cache is invalidated by CSV size and modification time.

The prediction is made after the final observed trade of a session. SELL means
next observed session return < -1%; BUY means > +1%; HOLD includes both boundaries.
These are experimental class names, not validated trading recommendations.
''')
code('''
from pathlib import Path
import copy, json, random, platform
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score)

OUT = Path('cnn_results')
OUT.mkdir(exist_ok=True)
SEED = 42
CLASSES = ['SELL', 'HOLD', 'BUY']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(min(4, torch.get_num_threads()))
def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_all()
print('Device:', DEVICE, '| PyTorch:', torch.__version__)
source = Path('data/stockpati_floorsheet_full.csv')
signature = {'size': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns}
cache, stamp = OUT/'daily_ohlcv.parquet', OUT/'daily_source.json'
if cache.exists() and stamp.exists() and json.loads(stamp.read_text()) == signature:
    daily = pl.read_parquet(cache)
else:
    daily = (pl.scan_csv(source, try_parse_dates=True,
                        schema_overrides={'contract_no': pl.Int64})
        .group_by(['symbol', 'date']).agg(
            pl.col('rate').sort_by('contract_no').first().alias('open'),
            pl.col('rate').max().alias('high'),
            pl.col('rate').min().alias('low'),
            pl.col('rate').sort_by('contract_no').last().alias('close'),
            pl.col('quantity').sum().alias('volume'),
            pl.col('amount').sum().alias('turnover'),
            pl.len().alias('trade_count'))
        .collect().sort(['symbol', 'date']))
    daily.write_parquet(cache)
    stamp.write_text(json.dumps(signature))
prices = pd.DataFrame(daily.to_dicts())
prices['date'] = pd.to_datetime(prices['date'])
print('Daily rows:', len(prices), '| symbols:', prices.symbol.nunique(),
      '| dates:', prices.date.nunique())
''')
md('''
## 13. Causal features and leakage controls

Each input has shape **[9 features, 20 sessions]**. Features are close return,
open/previous-close return, intraday range/close, close/open return,
log(1+volume), log(1+turnover), log(1+trade count), VWAP/close deviation,
and calendar-day gap since the previous observation. Absolute stock prices and
symbol IDs are excluded. No future price is an input. Gaps are measured rather
than filled with invented sessions.

Dates are split globally 70%/15%/15%. A label crossing the training or validation
boundary is purged. Windows may use historical context from the preceding
partition, which is available at prediction time. Scaling is fitted only on
the unique feature rows used by training windows; class weights use training
labels only. Windows never cross symbols. The first 20-session window also
needs a preceding price for return features, so some EDA rows are excluded.
''')
code('''
WINDOW = 20
g = prices.groupby('symbol', sort=False)
prev = g['close'].shift(1)
prices['close_return'] = prices['close']/prev - 1
prices['open_return'] = prices['open']/prev - 1
prices['range_fraction'] = (prices['high']-prices['low'])/prices['close']
prices['intraday_return'] = prices['close']/prices['open'] - 1
for col in ['volume', 'turnover', 'trade_count']:
    prices['log_'+col] = np.log1p(prices[col])
prices['vwap_deviation'] = prices['turnover']/prices['volume']/prices['close'] - 1
prices['gap_days'] = g['date'].diff().dt.days
prices['next_date'] = g['date'].shift(-1)
prices['next_return'] = g['close'].shift(-1)/prices['close'] - 1
prices['target'] = np.select([prices.next_return < -.01, prices.next_return > .01], [0, 2], default=1)
FEATURES = ['close_return', 'open_return', 'range_fraction', 'intraday_return',
            'log_volume', 'log_turnover', 'log_trade_count', 'vwap_deviation', 'gap_days']
dates = np.sort(prices.date.unique())
VAL_START, TEST_START = pd.Timestamp(dates[int(len(dates)*.70)]), pd.Timestamp(dates[int(len(dates)*.85)])
raw = prices[FEATURES].to_numpy(dtype=np.float64)
indices = {'train': [], 'validation': [], 'test': []}
ends = {k: [] for k in indices}
for _, stock in prices.groupby('symbol', sort=False):
    rows = stock.index.to_numpy()
    for j in range(WINDOW-1, len(rows)):
        end = rows[j]
        r = prices.loc[end]
        window = rows[j-WINDOW+1:j+1]
        if pd.isna(r.next_date) or not np.isfinite(raw[window]).all():
            continue
        if r.date < VAL_START and r.next_date < VAL_START:
            split = 'train'
        elif VAL_START <= r.date < TEST_START and r.next_date < TEST_START:
            split = 'validation'
        elif r.date >= TEST_START:
            split = 'test'
        else:
            continue
        indices[split].append(window)
        ends[split].append(end)
assert all(len(v) for v in indices.values())
scaler = StandardScaler().fit(raw[np.unique(np.asarray(indices['train']))])
scaled = scaler.transform(raw).astype(np.float32)
X = {k: scaled[np.asarray(v)].transpose(0, 2, 1).copy() for k,v in indices.items()}
y = {k: prices.loc[ends[k], 'target'].to_numpy(dtype=np.int64) for k in ends}
assert prices.loc[ends['train'], 'next_date'].max() < VAL_START
assert prices.loc[ends['validation'], 'next_date'].max() < TEST_START
for split in indices:
    assert np.isfinite(X[split]).all()
    assert np.all(prices.symbol.to_numpy()[np.asarray(indices[split])] ==
                  prices.symbol.to_numpy()[ends[split]][:, None])
counts = np.bincount(y['train'], minlength=3)
assert np.all(counts > 0)
weights = torch.tensor(len(y['train'])/(3*counts), dtype=torch.float32, device=DEVICE)
split_table = pd.DataFrame([{'split': k, 'samples': len(y[k]),
    'first_prediction': str(prices.loc[ends[k], 'date'].min().date()),
    'last_prediction': str(prices.loc[ends[k], 'date'].max().date()),
    **dict(zip(CLASSES, np.bincount(y[k], minlength=3).tolist()))} for k in y])
print('Validation starts:', VAL_START.date(), '| test starts:', TEST_START.date())
print(split_table.to_string(index=False))
print('Training class weights:', weights.cpu().numpy())
split_table.to_csv(OUT/'split_summary.csv', index=False)
''')
md('''
## 14. CNN architecture and controlled hyperparameter experiments

With padding=1 and kernel size 3, each convolution preserves the temporal
length; pooling reduces 20 → 10 → 5. Channel widths are 32 and 64, so the
linear classifier receives 64 × 5 values. Dropout is disabled during evaluation.

Four runs use the same seed, split, features, class-weighted loss, batch size
128 and 10-epoch budget. Starting from learning rate 0.001 and dropout 0.5,
we change learning rate to 0.0003, dropout to 0.2, then both. Adam weight decay
is 0.0001. Each run keeps the epoch with the best **validation macro F1**;
the best run is chosen by that same metric. Test labels play no role in selection.
These are single-seed comparisons, not statistical evidence of superiority.
''')
code('''
class StockCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(len(FEATURES), 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64*(WINDOW//4), 3))
    def forward(self, x):
        return self.net(x)

datasets = {k: TensorDataset(torch.from_numpy(X[k]), torch.from_numpy(y[k])) for k in X}
def loader(split, shuffle=False):
    return DataLoader(datasets[split], batch_size=128, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED), num_workers=0)

@torch.no_grad()
def predict(model, split):
    model.eval()
    return np.concatenate([model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                           for xb, _ in loader(split)])

CONFIGS = [dict(name='reference', lr=.001, dropout=.5),
           dict(name='lower_lr', lr=.0003, dropout=.5),
           dict(name='lower_dropout', lr=.001, dropout=.2),
           dict(name='both_changes', lr=.0003, dropout=.2)]
EPOCHS = 10
history, results, states = [], [], {}
print(StockCNN())
for cfg in CONFIGS:
    seed_all()
    model = StockCNN(cfg['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=.0001)
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none')
    train_loader = loader('train', shuffle=True)
    best_score, best_epoch = -1., 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        loss_sum, weight_sum, correct = 0., 0., 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_values = loss_fn(logits, yb)
            denominator = weights[yb].sum()
            loss = loss_values.sum()/denominator
            loss.backward()
            optimizer.step()
            loss_sum += loss_values.sum().item()
            weight_sum += denominator.item()
            correct += (logits.argmax(1) == yb).sum().item()
        vp = predict(model, 'validation')
        score = f1_score(y['validation'], vp, labels=[0,1,2], average='macro', zero_division=0)
        history.append(dict(run=cfg['name'], epoch=epoch, train_loss=loss_sum/weight_sum,
                            train_accuracy=correct/len(y['train']),
                            val_accuracy=accuracy_score(y['validation'], vp), val_macro_f1=score))
        if score > best_score:
            best_score, best_epoch = score, epoch
            states[cfg['name']] = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{cfg['name']:14s} epoch {epoch:2d}: loss={loss_sum/weight_sum:.4f}, validation F1={score:.4f}", flush=True)
    results.append({**cfg, 'batch_size': 128, 'epochs': EPOCHS,
                    'best_epoch': best_epoch, 'val_macro_f1': best_score})
experiments = pd.DataFrame(results).sort_values('val_macro_f1', ascending=False, kind='stable')
history = pd.DataFrame(history)
experiments.to_csv(OUT/'experiments.csv', index=False)
history.to_csv(OUT/'training_history.csv', index=False)
winner = experiments.iloc[0].to_dict()
model = StockCNN(winner['dropout']).to(DEVICE)
model.load_state_dict(states[winner['name']])
print('Validation-selected experiments:')
print(experiments.to_string(index=False))
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in history.groupby('run', sort=False):
    axes[0].plot(h.epoch, h.train_loss, marker='.', label=name)
    axes[1].plot(h.epoch, h.val_macro_f1, marker='.', label=name)
axes[0].set(title='Training weighted cross-entropy', xlabel='Epoch', ylabel='Loss')
axes[1].set(title='Validation macro F1 (selection metric)', xlabel='Epoch', ylabel='Macro F1')
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT/'learning_curves.png', dpi=160)
plt.show()
''')
md('''
## 15. Final held-out classifier evaluation

Evaluate the validation-selected checkpoint once on the final date block.
Precision measures how often a predicted class is correct; recall measures
how many actual members of that class are recovered. F-score here means
**F1 = 2 × precision × recall / (precision + recall)**. Macro F1 averages
equally across classes; weighted F1 weights by support. Undefined precision
or recall is reported as zero. Matrix rows are actual classes, columns predicted.
The always-HOLD baseline uses the same test samples. Its zero recall on SELL
and BUY demonstrates why accuracy alone is insufficient.
''')
code('''
test_pred = predict(model, 'test')
baseline_pred = np.full_like(y['test'], 1)
report = pd.DataFrame(classification_report(y['test'], test_pred, labels=[0,1,2],
    target_names=CLASSES, output_dict=True, zero_division=0)).T
comparison = pd.DataFrame([{'model': name,
    'accuracy': accuracy_score(y['test'], pred),
    'balanced_accuracy': balanced_accuracy_score(y['test'], pred),
    'macro_f1': f1_score(y['test'], pred, labels=[0,1,2], average='macro', zero_division=0),
    'weighted_f1': f1_score(y['test'], pred, labels=[0,1,2], average='weighted', zero_division=0)}
    for name,pred in [('CNN', test_pred), ('Always HOLD', baseline_pred)]])
print('Chosen configuration:', winner)
print(classification_report(y['test'], test_pred, labels=[0,1,2], target_names=CLASSES, digits=4, zero_division=0))
print(comparison.to_string(index=False))
report.to_csv(OUT/'classification_report.csv')
comparison.to_csv(OUT/'baseline_comparison.csv', index=False)
test_details = prices.loc[ends['test'], ['symbol','date','next_date','trade_count','next_return']].copy()
test_details['actual'] = np.asarray(CLASSES)[y['test']]
test_details['predicted'] = np.asarray(CLASSES)[test_pred]
test_details.to_csv(OUT/'test_predictions.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, norm, title in zip(axes, [None, 'true'], ['Test confusion matrix: counts', 'Test confusion matrix: recall by row']):
    ConfusionMatrixDisplay.from_predictions(y['test'], test_pred, labels=[0,1,2],
        display_labels=CLASSES, normalize=norm, cmap='Blues', ax=ax,
        values_format='d' if norm is None else '.2f', colorbar=False)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(OUT/'confusion_matrix.png', dpi=160)
plt.show()
slice_rows = []
for name, mask in [('Fewer than 20 trades', test_details.trade_count.to_numpy() < 20),
                   ('At least 20 trades', test_details.trade_count.to_numpy() >= 20)]:
    if mask.any():
        slice_rows.append({'slice': name, 'samples': int(mask.sum()),
            'macro_f1': f1_score(y['test'][mask], test_pred[mask], labels=[0,1,2], average='macro', zero_division=0)})
slices = pd.DataFrame(slice_rows)
print('Liquidity slice diagnostics (no further tuning):')
print(slices.to_string(index=False))
slices.to_csv(OUT/'liquidity_slices.csv', index=False)
''')
md('''
## 16. Save the model and verify reloading

The checkpoint stores the selected weights, feature order, scaling parameters,
class mapping, window, split boundaries, and experiment configuration. A new
prediction must reproduce the same causal feature calculation and window order.
''')
code('''
torch.save({'state_dict': states[winner['name']], 'config': winner,
            'features': FEATURES, 'classes': CLASSES, 'window': WINDOW,
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
            'validation_start': str(VAL_START.date()), 'test_start': str(TEST_START.date()),
            'seed': SEED}, OUT/'stock_cnn.pt')
saved = torch.load(OUT/'stock_cnn.pt', map_location='cpu', weights_only=True)
reloaded = StockCNN(saved['config']['dropout']).to(DEVICE)
reloaded.load_state_dict(saved['state_dict'])
model.eval()
reloaded.eval()
with torch.no_grad():
    probe = torch.from_numpy(X['test'][:128]).to(DEVICE)
    assert torch.equal(model(probe), reloaded(probe))
print('Checkpoint reload verified: identical logits on 128 held-out inputs.')
environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
               'numpy': np.__version__, 'pandas': pd.__version__, 'polars': pl.__version__,
               'device': str(DEVICE), 'seed': SEED, 'source': signature}
(OUT/'environment.json').write_text(json.dumps(environment, indent=2))
''')
md('''
## 17. Discoveries and limitations

The following report text is generated from the executed experiment, so the
reported numbers are measured rather than illustrative. Changing learning rate
and dropout can affect both fitting and validation performance; the best epoch
need not be the last epoch. Test performance is an assessment, not a tuning input.
''')
code('''
cnn, hold = comparison.iloc[0], comparison.iloc[1]
reference = experiments.set_index('name').loc['reference', 'val_macro_f1']
discoveries = [
    f"Used {len(y['train']):,} training, {len(y['validation']):,} validation and {len(y['test']):,} test sequences from the week-2 floorsheet.",
    f"Selected {winner['name']} (learning rate {winner['lr']}, dropout {winner['dropout']}) at epoch {int(winner['best_epoch'])}; validation macro F1 {winner['val_macro_f1']:.4f}.",
    f"The selected validation macro F1 differs from the reference by {winner['val_macro_f1']-reference:+.4f}.",
    f"Held-out CNN accuracy {cnn.accuracy:.4f}, balanced accuracy {cnn.balanced_accuracy:.4f}, macro F1 {cnn.macro_f1:.4f}, weighted F1 {cnn.weighted_f1:.4f}.",
    f"Always-HOLD accuracy {hold.accuracy:.4f} and macro F1 {hold.macro_f1:.4f}; CNN differences are {cnn.accuracy-hold.accuracy:+.4f} accuracy and {cnn.macro_f1-hold.macro_f1:+.4f} macro F1.",
]
for name in CLASSES:
    row = report.loc[name]
    discoveries.append(f"{name}: precision {row['precision']:.4f}, recall {row['recall']:.4f}, F1 {row['f1-score']:.4f}, support {int(row['support'])}.")
cm = confusion_matrix(y['test'], test_pred, labels=[0,1,2])
errors = cm.copy()
np.fill_diagonal(errors, 0)
a,b = np.unravel_index(errors.argmax(), errors.shape)
discoveries.append(f"Largest off-diagonal confusion: actual {CLASSES[a]} predicted {CLASSES[b]} ({errors[a,b]:,} samples).")
discoveries.append('Limitations: one seed and one time split; overlapping windows and shared market dates create correlated samples. No confidence intervals or profitability claim. Prices are unadjusted last-trade proxies; corporate actions, sparse trading, changing class balance and irregular session gaps can affect labels and generalization. The 1% threshold and 20-session window were fixed, not tuned. A wider search, repeated seeds and walk-forward validation are future work.')
print('\\n\\n'.join(discoveries))
(OUT/'discoveries.txt').write_text('\\n\\n'.join(discoveries), encoding='utf-8')
''')
md('''
## 18. Report and snapshots

`cnn_results/CNN_Report.html` contains the dataset/method description, actual
hyperparameter and evaluation tables, discoveries, and snapshots of code,
training curves and confusion matrices. Code snapshots are rendered directly
from the executed cells; result figures and table snapshots come from the actual
run, not mock outputs. The notebook also embeds all cell outputs and plots.
''')

scope = {'__name__': '__main__'}
import matplotlib.pyplot as plt
counter = 0
for cell in cells:
    if cell['cell_type'] != 'code':
        continue
    counter += 1
    cell['execution_count'] = counter
    buffer = io.StringIO()
    def show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format='png', dpi=130, bbox_inches='tight')
            cell['outputs'].append({'output_type':'display_data', 'metadata':{},
                'data':{'image/png':base64.b64encode(png.getvalue()).decode(), 'text/plain':['<Figure>']}})
        plt.close('all')
    plt.show = show
    print(f'Executing CNN cell {counter}', flush=True)
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(''.join(cell['source']), f'CNN cell {counter}', 'exec'), scope)
    except Exception:
        print(buffer.getvalue())
        raise
    if buffer.getvalue():
        cell['outputs'].insert(0, {'output_type':'stream', 'name':'stdout', 'text':buffer.getvalue().splitlines(True)})
        print(buffer.getvalue(), flush=True)
    nb['cells'] = original + cells
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')

# Snapshot code and actual tabular results using the same rendered font pipeline.
def snapshot(text, filename, title):
    lines = text.splitlines()
    fig, ax = plt.subplots(figsize=(13, max(3, .22*len(lines)+.7)))
    fig.patch.set_facecolor('#f5f7fa')
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=14, pad=14)
    ax.text(0, 1, text, va='top', family='monospace', fontsize=9, transform=ax.transAxes)
    fig.savefig(OUT/filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

architecture = next(''.join(c['source']) for c in cells if c['cell_type']=='code' and 'class StockCNN' in ''.join(c['source']))
snapshot(architecture.split('CONFIGS =')[0], 'code_architecture.png', 'Executed code: 1D CNN and evaluation')
snapshot(architecture[architecture.index('CONFIGS ='):architecture.index('experiments =')], 'code_training.png', 'Executed code: hyperparameters and training')
snapshot(scope['experiments'].to_string(index=False)+'\n\n'+scope['comparison'].to_string(index=False)+'\n\n'+scope['report'].round(4).to_string(), 'results_snapshot.png', 'Measured experiment and held-out results')

def embedded(name, caption):
    data = base64.b64encode((OUT/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
report_html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Uttam: CNN Experiment Report</title>
<style>body{font:16px/1.6 Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172b40}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border:1px solid #ccd5df;text-align:right}img{max-width:100%}figure{margin:24px 0}figcaption{color:#526274}pre{white-space:pre-wrap}@media print{figure,table{break-inside:avoid}}</style>
<h1>StockPati Floorsheet: CNN Classification Experiment</h1>
<p>Student: Uttam · Dataset: week-2 NEPSE floorsheet · Model: PyTorch 1D CNN</p>
<h2>Dataset and task</h2><p>The local CSV is aggregated to stock-day OHLCV using contract-number trade ordering. The target is the next observed stock-session last-trade return: SELL below −1%, BUY above +1%, otherwise HOLD. Final observations without a future price are excluded. Labels describe this experiment, not established trading advice.</p>
<h2>Method and leakage prevention</h2><p>The FashionMNIST architecture is adapted from 2D image convolution to 1D temporal convolution. Nine causal features over 20 observed sessions feed Conv1d(9,32,3), ReLU, pooling, Conv1d(32,64,3), ReLU, pooling, flatten, dropout and Linear(320,3). Padding preserves length before pooling. Features are close and open returns relative to previous close, intraday range and return, log volume/turnover/trade count, VWAP deviation and observation gap in days.</p>
<p>Global date boundaries allocate 70%/15%/15% to train/validation/test. Targets crossing a partition boundary are removed. Historical context from earlier dates is permitted; no window crosses stocks. Scaling uses only feature rows in training windows and class weights use only training labels. Prediction takes place after the final trade of the current session.</p>'''
report_html += scope['split_table'].to_html(index=False)
report_html += '<h2>Hyperparameter experiment</h2><p>Four configurations use seed 42, batch size 128, 10 epochs, Adam weight decay 0.0001 and training-derived class weights. Each run keeps its best validation macro-F1 epoch. Only validation selects the final model; all results below use that frozen model. Single-seed outcomes do not establish statistical significance.</p>'
report_html += scope['experiments'].round(5).to_html(index=False)
report_html += embedded('code_architecture.png', 'Snapshot 1. Actual executed architecture and prediction code.')
report_html += embedded('code_training.png', 'Snapshot 2. Actual executed configuration and training loop.')
report_html += embedded('learning_curves.png', 'Snapshot 3. Training loss and validation macro F1 for all configurations.')
report_html += '<h2>Held-out results</h2><p>Precision is the correct fraction of class predictions; recall is the recovered fraction of actual class members. F1 is their harmonic mean. Macro F1 gives equal weight to classes; weighted F1 uses class support. Undefined metrics are zero.</p>'
report_html += scope['comparison'].round(4).to_html(index=False)
report_html += scope['report'].round(4).to_html()
report_html += embedded('confusion_matrix.png', 'Snapshot 4. Actual test confusion matrix: counts and row-normalized recall.')
report_html += embedded('results_snapshot.png', 'Snapshot 5. Actual experiment table, baseline comparison and classification report.')
report_html += '<h2>Discoveries</h2>'+''.join('<p>'+html.escape(s)+'</p>' for s in scope['discoveries'])
report_html += '<h2>Liquidity diagnostics</h2>'+scope['slices'].round(4).to_html(index=False)
report_html += '<h2>Reproducibility and artifacts</h2><p>Run notebook sections 12–18 from the project folder. The notebook contains executed outputs. cnn_results contains CSV tables, per-sample predictions, PNG snapshots and stock_cnn.pt with model weights and preprocessing metadata. Reloading was checked for identical logits on 128 inputs. GPU/platform differences may affect exact reproduction.</p><pre>'+html.escape(json.dumps(scope['environment'], indent=2))+'</pre></html>'
(OUT/'CNN_Report.html').write_text(report_html, encoding='utf-8')
print('Saved executed notebook and self-contained report:', OUT/'CNN_Report.html')
