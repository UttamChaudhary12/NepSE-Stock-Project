from django.contrib import admin

from .models import ModelRun, Signal, Stock


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'name', 'sector', 'promoter')
    list_filter = ('sector', 'promoter')
    search_fields = ('symbol', 'name')


@admin.register(Signal)
class SignalAdmin(admin.ModelAdmin):
    list_display = ('stock', 'date', 'close', 'signal', 'p_buy', 'p_hold', 'p_sell', 'actual', 'in_sample')
    list_filter = ('signal', 'actual', 'in_sample', 'date')
    search_fields = ('stock__symbol',)


admin.site.register(ModelRun)
