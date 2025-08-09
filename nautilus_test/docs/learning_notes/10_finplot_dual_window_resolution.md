# Learning Note 10: Finplot Dual Window Issue Resolution

**Date**: 2025-07-15  
**Context**: Enhanced DSM Hybrid Integration - Production System  
**Issue**: Multiple finplot windows appearing (one normal, one small/empty)

## Problem Description

When running the enhanced DSM hybrid integration system, two finplot canvas windows were appearing:

1. **Main window**: Normal size with proper chart data
2. **Secondary window**: Small and empty, appearing alongside the main window

This created a confusing user experience and indicated improper window management in the finplot integration.

## Root Cause Analysis

### Primary Cause: Dual Window Creation

The issue was caused by **two separate finplot window creation calls**:

1. **FinplotActor window creation** (line 147 in enhanced_dsm_hybrid_integration.py):

   ```python
   # In FinplotActor.on_start()
   self._ax, self._ax2 = fplt.create_plot('Live NautilusTrader Data', rows=2, maximize=False)
   ```

2. **Post-backtest visualization window** (line 777):
   ```python
   # In create_enhanced_candlestick_chart()
   ax, ax2 = fplt.create_plot(title, rows=2)
   ```

### Secondary Factor: Theme Setup Side Effects

The `_setup_chart_theme()` method in FinplotActor was also setting finplot global properties:

```python
def _setup_chart_theme(self):
    fplt.foreground = '#f0f6fc'
    fplt.background = '#0d1117'
    pg.setConfigOptions(...)  # This might trigger window initialization
```

## Solution Implementation

### 1. Conditional Window Creation in FinplotActor

```python
def __init__(self, config=None):
    super().__init__(config)
    # ... other initialization ...
    self._backtest_mode = True  # Default to backtest mode

    # Skip chart styling in backtest mode to prevent window creation
    if not self._backtest_mode:
        self._setup_chart_theme()
```

### 2. Commented Out Plot Creation in Backtest Mode

```python
def on_start(self) -> None:
    # In backtest mode, skip creating the plot window to avoid duplicate windows
    # For live trading, uncomment the following lines:
    # self._ax, self._ax2 = fplt.create_plot('Live NautilusTrader Data', rows=2, maximize=False)
    # self._timer = pg.QtCore.QTimer()
    # self._timer.timeout.connect(self._refresh_chart)
    # self._timer.start(100)  # 100ms refresh rate for smooth updates

    self.log.info("FinplotActor started (backtest mode - chart creation skipped)")
```

### 3. Safety Check in Chart Refresh

```python
def _refresh_chart(self):
    # Skip if axes not created (backtest mode)
    if self._ax is None or self._ax2 is None:
        return
    # ... rest of refresh logic ...
```

## Key Learnings

### 1. Finplot Window Creation Behavior

- `fplt.create_plot()` immediately creates and displays a window
- Even without calling `fplt.show()`, the window appears
- Theme setup operations can trigger window initialization

### 2. Backtest vs Live Trading Distinction

- **Backtest mode**: Only post-analysis visualization needed
- **Live trading mode**: Real-time chart updates required
- Need clear separation of concerns between modes

### 3. Actor Lifecycle Management

- Actors in backtest mode don't need live chart capabilities
- Resource allocation should match the execution context
- Proper initialization flags prevent unnecessary resource usage

## Best Practices Established

### 1. Mode-Aware Initialization

```python
def __init__(self, config=None, backtest_mode=True):
    # Initialize based on execution context
    self._backtest_mode = backtest_mode

    if not self._backtest_mode:
        self._setup_live_charting()
```

### 2. Conditional Resource Allocation

```python
# Only allocate chart resources for live trading
if not self._backtest_mode:
    self._ax, self._ax2 = fplt.create_plot(...)
    self._timer = pg.QtCore.QTimer()
```

### 3. Clear Console Messaging

```python
# Inform user about the execution mode
console.print("[blue]🚀 FinplotActor started - backtest mode (post-backtest chart will be shown)[/blue]")
```

## Verification

### Before Fix

- **Symptoms**: Two windows appearing, one small/empty
- **Resource waste**: Unnecessary window creation and management
- **User confusion**: Multiple windows with unclear purpose

### After Fix

- **Result**: Single comprehensive chart window
- **Clean output**: Only post-backtest analysis chart
- **Clear messaging**: User informed about execution mode

## Implementation Impact

### Files Modified

- `enhanced_dsm_hybrid_integration.py`: FinplotActor class
  - Modified `__init__()` method
  - Updated `on_start()` method
  - Enhanced console messaging

### Performance Benefits

- **Reduced memory usage**: No unnecessary window creation
- **Faster startup**: Skip live chart initialization in backtest
- **Cleaner UX**: Single, purposeful visualization

## Future Considerations

### 1. Configuration-Driven Mode Selection

```python
# Future enhancement: Config-driven mode selection
class FinplotActor(Actor):
    def __init__(self, config=None):
        mode = config.get('execution_mode', 'backtest')
        self._live_mode = (mode == 'live')
```

### 2. Live Trading Integration

When implementing live trading mode:

- Uncomment the plot creation code
- Enable timer-based updates
- Add proper error handling for chart operations

### 3. Resource Management

- Implement proper cleanup in `on_stop()`
- Consider memory limits for long-running live sessions
- Add chart update throttling for high-frequency data

## Related Documentation

- `finplot_integration_guide.md`: Main integration patterns
- `08_backtesting_pnl_calculation_lessons.md`: Backtest execution context
- `09_native_integration_refactoring_lessons.md`: Actor lifecycle management

---

_This resolution demonstrates the importance of context-aware resource allocation and proper separation between backtest and live trading execution modes._
