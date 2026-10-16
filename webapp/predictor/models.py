from django.db import models


class Stock(models.Model):
    symbol = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=200)
    sector = models.CharField(max_length=40)
    promoter = models.BooleanField(default=False)

    class Meta:
        ordering = ['symbol']

    def __str__(self):
        return self.symbol


class Signal(models.Model):
    SELL, HOLD, BUY = 'SELL', 'HOLD', 'BUY'
    CHOICES = [(SELL, 'Sell'), (HOLD, 'Hold'), (BUY, 'Buy')]

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='signals')
    date = models.DateField(db_index=True)
    close = models.FloatField()
    change = models.FloatField(null=True, help_text='Return since the previous observed session')
    volume = models.BigIntegerField()
    turnover = models.FloatField()
    trade_count = models.IntegerField()
    p_sell = models.FloatField()
    p_hold = models.FloatField()
    p_buy = models.FloatField()
    signal = models.CharField(max_length=4, choices=CHOICES, db_index=True)
    threshold = models.FloatField(help_text='Move needed for BUY/SELL over the horizon')
    fwd_return = models.FloatField(null=True, help_text='Realised return over the horizon, once known')
    actual = models.CharField(max_length=4, choices=CHOICES, blank=True)
    in_sample = models.BooleanField(help_text='Label was used to train the model')

    class Meta:
        unique_together = [('stock', 'date')]
        ordering = ['-date', 'stock']

    @property
    def confidence(self):
        return {'SELL': self.p_sell, 'HOLD': self.p_hold, 'BUY': self.p_buy}[self.signal]

    @property
    def correct(self):
        return self.signal == self.actual if self.actual else None


class ModelRun(models.Model):
    """Metadata and evaluation numbers for the model behind the stored signals."""
    built_at = models.DateTimeField(auto_now_add=True)
    model_path = models.CharField(max_length=500)
    data_through = models.DateField()
    test_start = models.DateField()
    info = models.JSONField(default=dict)

    class Meta:
        get_latest_by = 'built_at'
