from django import template

register = template.Library()


@register.filter
def pct(value, digits=1):
    """0.1234 -> '12.3%'."""
    if value is None or value == '':
        return '–'
    return f'{float(value)*100:.{int(digits)}f}%'


@register.filter
def spct(value, digits=1):
    """Signed percentage: 0.012 -> '+1.2%'."""
    if value is None or value == '':
        return '–'
    return f'{float(value)*100:+.{int(digits)}f}%'


@register.filter
def width(value):
    """Probability -> CSS width percentage."""
    return f'{float(value)*100:.1f}%'


@register.filter
def num(value, digits=4):
    return '–' if value is None or value == '' else f'{float(value):.{int(digits)}f}'
