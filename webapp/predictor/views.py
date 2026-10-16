from datetime import date as Date

from django.db.models import Count, Max
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render

from .models import ModelRun, Signal, Stock

SORTS = {
    'confidence': None,  # sorted in Python: probability of the predicted class
    'buy': '-p_buy', 'sell': '-p_sell', 'change': '-change', 'turnover': '-turnover', 'symbol': 'stock__symbol',
}


def _selected_date(request):
    latest = Signal.objects.aggregate(d=Max('date'))['d']
    if latest is None:
        return None, None
    try:
        wanted = Date.fromisoformat(request.GET.get('date', ''))
    except ValueError:
        return latest, latest
    # Non-trading days fall back to the closest earlier session (or the first one).
    day = (Signal.objects.filter(date__lte=wanted).aggregate(d=Max('date'))['d']
           or Signal.objects.order_by('date').values_list('date', flat=True).first())
    return day, latest


def dashboard(request):
    day, latest = _selected_date(request)
    if day is None:
        return render(request, 'predictor/empty.html')
    qs = Signal.objects.filter(date=day).select_related('stock')
    counts = dict(qs.values_list('signal').annotate(n=Count('id')))
    signal = request.GET.get('signal', '')
    sector = request.GET.get('sector', '')
    q = request.GET.get('q', '').strip()
    liquid = request.GET.get('liquid') == '1'
    sort = request.GET.get('sort', 'confidence')
    if signal in ('BUY', 'HOLD', 'SELL'):
        qs = qs.filter(signal=signal)
    if sector:
        qs = qs.filter(stock__sector=sector)
    if q:
        qs = qs.filter(stock__symbol__icontains=q) | qs.filter(stock__name__icontains=q)
    if liquid:
        qs = qs.filter(trade_count__gte=20)
    if SORTS.get(sort):
        rows = list(qs.order_by(SORTS[sort]))
    else:
        rows = sorted(qs, key=lambda s: s.confidence, reverse=True)
    dates = list(Signal.objects.values_list('date', flat=True).distinct().order_by('-date'))
    run = ModelRun.objects.order_by('-built_at').first()
    return render(request, 'predictor/dashboard.html', {
        'rows': rows, 'day': day, 'latest': latest, 'dates': dates, 'run': run,
        'counts': {k: counts.get(k, 0) for k in ('BUY', 'HOLD', 'SELL')},
        'total': sum(counts.values()),
        'sectors': Stock.objects.values_list('sector', flat=True).distinct().order_by('sector'),
        'f': {'signal': signal, 'sector': sector, 'q': q, 'liquid': liquid, 'sort': sort},
        'sorts': list(SORTS),
        'out_of_sample': run is None or day >= run.test_start,
    })


def stock_detail(request, symbol):
    stock = get_object_or_404(Stock, symbol=symbol.upper())
    signals = list(stock.signals.order_by('date'))
    if not signals:
        raise Http404('No signals for this stock')
    run = ModelRun.objects.order_by('-built_at').first()
    scored = [s for s in signals if s.actual and not s.in_sample]
    hits = sum(s.correct for s in scored)
    chart = {
        'dates': [s.date.isoformat() for s in signals],
        'close': [s.close for s in signals],
        'signal': [s.signal for s in signals],
        'p_buy': [round(s.p_buy, 4) for s in signals],
        'p_hold': [round(s.p_hold, 4) for s in signals],
        'p_sell': [round(s.p_sell, 4) for s in signals],
        'test_start': run.test_start.isoformat() if run else None,
    }
    return render(request, 'predictor/stock.html', {
        'stock': stock, 'latest': signals[-1], 'history': signals[::-1][:60], 'chart': chart,
        'scored': len(scored), 'hits': hits, 'hit_rate': hits/len(scored) if scored else None, 'run': run,
    })


def about(request):
    return render(request, 'predictor/about.html', {'run': ModelRun.objects.order_by('-built_at').first()})


def api_signals(request):
    day, _ = _selected_date(request)
    rows = Signal.objects.filter(date=day).select_related('stock').order_by('stock__symbol') if day else []
    return JsonResponse({'date': day.isoformat() if day else None, 'signals': [
        {'symbol': s.stock_id, 'name': s.stock.name, 'sector': s.stock.sector, 'close': s.close,
         'signal': s.signal, 'p_sell': round(s.p_sell, 4), 'p_hold': round(s.p_hold, 4),
         'p_buy': round(s.p_buy, 4), 'trade_count': s.trade_count} for s in rows]})
