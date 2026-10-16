"""Tuned LSTM ensemble from Uttam_EDA.ipynb section 29: load the checkpoint and score stock-days."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import MIN_HISTORY


class TunedLSTM(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout, pool):
        super().__init__()
        self.pool = pool
        self.rnn = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden*(2 if pool == 'last+mean' else 1), 3)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = {'last': lambda: out[:, -1], 'mean': lambda: out.mean(1),
             'last+mean': lambda: torch.cat([out[:, -1], out.mean(1)], 1)}[self.pool]()
        return self.head(self.drop(h))


class Ensemble:
    def __init__(self, path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.ckpt = torch.load(path, map_location=self.device, weights_only=True)
        cfg = self.ckpt['config']
        self.features, self.sectors = self.ckpt['features'], self.ckpt['sectors']
        self.window = cfg['window']
        self.bias = np.asarray(self.ckpt['bias'])
        self.models = []
        for state in self.ckpt['state_dicts']:
            model = TunedLSTM(len(self.features) + len(self.sectors), cfg['hidden'], cfg['layers'],
                              cfg['dropout'], cfg['pool']).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def matrix(self, frame):
        """Standardise with the training-period scaler stored in the checkpoint."""
        raw = frame[self.features].to_numpy(np.float64)
        z = (raw - np.asarray(self.ckpt['scaler_mean']))/np.asarray(self.ckpt['scaler_std'])
        z = np.nan_to_num(np.clip(z, -5, 5), nan=0.).astype(np.float32)
        onehot = pd.get_dummies(frame.sector).reindex(columns=self.sectors, fill_value=False)
        return torch.from_numpy(np.hstack([z, onehot.to_numpy(np.float32)])).to(self.device)

    @torch.no_grad()
    def predict(self, frame, batch=4096):
        """Probabilities and biased class for every row with enough history (others are NaN / -1).

        `frame` must be sorted by symbol then date with a default RangeIndex, as build_features returns.
        """
        S = self.matrix(frame)
        ends = np.flatnonzero(frame.pos.to_numpy() >= MIN_HISTORY)
        offsets = torch.arange(-self.window + 1, 1, device=self.device)
        probs = np.zeros((len(ends), 3))
        for model in self.models:
            for i in range(0, len(ends), batch):
                e = torch.as_tensor(ends[i:i+batch], device=self.device)
                probs[i:i+batch] += torch.softmax(model(S[e[:, None] + offsets]), 1).cpu().numpy()
        probs /= len(self.models)
        full = np.full((len(frame), 3), np.nan)
        full[ends] = probs
        signal = np.full(len(frame), -1)
        signal[ends] = (np.log(probs + 1e-9) + self.bias).argmax(1)
        return full, signal
