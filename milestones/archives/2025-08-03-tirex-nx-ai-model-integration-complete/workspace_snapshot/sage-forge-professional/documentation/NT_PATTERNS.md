# 🎯 SAGE-Forge NautilusTrader Pattern Compliance

**Complete guide to NT-native pattern implementation in SAGE-Forge**

---

## 🏗️ **Core NT Architecture Patterns**

### **Actor Pattern Implementation**

```python
from nautilus_trader.common.actor import Actor

class SAGEActor(Actor):
    """NT-native actor following SAGE-Forge patterns."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

    def on_start(self):
        """Required: Initialize actor."""
        self.log.info("SAGE Actor started")

    def on_stop(self):
        """Required: Cleanup on shutdown."""
        self.log.info("SAGE Actor stopped")

    def on_event(self, event):
        """Optional: Handle custom events."""
        pass
```

### **Strategy Pattern Implementation**

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar

class SAGEStrategy(Strategy):
    """NT-native strategy following SAGE-Forge patterns."""

    def __init__(self, config):
        super().__init__(config)

    def on_start(self):
        """Required: Subscribe to data and initialize."""
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        """Required: Process bar data."""
        # Trading logic here
        pass

    def on_stop(self):
        """Required: Cleanup on shutdown."""
        pass
```

### **Indicator Pattern Implementation**

```python
from nautilus_trader.indicators.base.indicator import Indicator

class SAGEIndicator(Indicator):
    """NT-native indicator following SAGE-Forge patterns."""

    def __init__(self, period: int):
        super().__init__(params=[period])
        self.period = period

    def _compute(self, value: float) -> None:
        """Required: Compute indicator value."""
        # Indicator logic here
        pass
```

---

## 🔧 **SAGE-Forge Specific Patterns**

### **Model Integration Pattern**

```python
from sage_forge.models.base import BaseSAGEModel

class TiRexModel(BaseSAGEModel):
    """SAGE-Forge model wrapper for TiRex."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def predict(self, data):
        """Generate model prediction."""
        pass

    def get_uncertainty(self, data):
        """Get prediction uncertainty."""
        pass
```

### **Risk Management Pattern**

```python
from sage_forge.risk.position_sizer import RealisticPositionSizer

class SAGEStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        # Always use professional position sizing
        self.position_sizer = RealisticPositionSizer()

    def _calculate_position_size(self, signal_strength: float):
        """Use SAGE-Forge risk management."""
        return self.position_sizer.calculate_size(
            signal_strength=signal_strength,
            account_balance=self.account.balance()
        )
```

---

## 📊 **Data Flow Patterns**

### **Market Data Pipeline**

```python
from sage_forge.data.manager import ArrowDataManager
from sage_forge.data.enhanced_provider import EnhancedModernBarDataProvider

# 1. Initialize data manager
data_manager = ArrowDataManager()

# 2. Create enhanced provider
provider = EnhancedModernBarDataProvider(specs_manager)

# 3. Fetch and validate data
bars = provider.fetch_real_market_bars(
    instrument=instrument,
    bar_type=bar_type,
    limit=500
)
```

### **Event Flow Pattern**

```
Market Data → Enhanced Provider → Validation → NT Engine → Strategy → Actor → Orders
     ↓              ↓               ↓           ↓          ↓        ↓        ↓
   Real API    Specification    Quality     Backtesting  Trading  Funding  Execution
  Integration    Validation     Checks       Engine      Logic   Tracking   & Fills
```

---

## 🧪 **Testing Patterns**

### **Component Testing**

```python
import pytest
from sage_forge.strategies.tirex_strategy import TiRexStrategy

def test_tirex_strategy_compliance():
    """Test NT pattern compliance."""
    config = create_test_config()
    strategy = TiRexStrategy(config)

    # Test required methods exist
    assert hasattr(strategy, 'on_start')
    assert hasattr(strategy, 'on_bar')
    assert hasattr(strategy, 'on_stop')

    # Test inheritance
    assert isinstance(strategy, Strategy)
```

### **Integration Testing**

```python
def test_sage_forge_integration():
    """Test complete SAGE-Forge pipeline."""
    # 1. Create engine
    engine = create_test_engine()

    # 2. Add SAGE components
    engine.add_actor(FundingActor())
    engine.add_actor(FinplotActor())
    engine.add_strategy(SAGEStrategy(config))

    # 3. Run and validate
    engine.run()
    assert len(engine.cache.orders()) > 0
```

---

## 🔍 **Validation Checklist**

### **Strategy Validation**

- [ ] Inherits from `nautilus_trader.trading.strategy.Strategy`
- [ ] Implements `on_start()`, `on_bar()`, `on_stop()`
- [ ] Uses proper subscription patterns
- [ ] Handles orders through `self.order_factory`
- [ ] Uses SAGE-Forge risk management

### **Actor Validation**

- [ ] Inherits from `nautilus_trader.common.actor.Actor`
- [ ] Implements `on_start()`, `on_stop()`
- [ ] Handles events properly
- [ ] Uses NT logging system
- [ ] Integrates with message bus

### **Integration Validation**

- [ ] All imports work correctly
- [ ] Components communicate via NT message bus
- [ ] Data flows through NT cache system
- [ ] Orders execute through NT order management
- [ ] Performance tracking via NT reporting

---

## 🚀 **Best Practices**

### **Performance**

- Use NT cache system for data storage
- Minimize object creation in hot paths
- Leverage NT's built-in parallel processing
- Cache expensive calculations

### **Error Handling**

- Use NT logging system (`self.log`)
- Handle data quality issues gracefully
- Implement fallback mechanisms
- Validate inputs at boundaries

### **Testing**

- Test components in isolation
- Use NT test fixtures
- Validate against live data
- Performance test with realistic data sizes

---

**Last Updated**: 2025-08-03  
**Compliance**: 100% NautilusTrader native patterns  
**Usage**: Reference for all SAGE-Forge development
