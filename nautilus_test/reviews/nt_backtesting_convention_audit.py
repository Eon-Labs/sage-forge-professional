#!/usr/bin/env python3
"""
ADVERSARIAL REVIEW: NautilusTrader Backtesting Convention Compliance
PURPOSE: Critically examine whether SAGE model validation follows NT's native backtesting patterns
SCOPE: Architecture, data handling, execution flow, venue setup, strategy patterns
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class NautilusBacktestConventionAuditor:
    """
    Adversarial auditor examining NT backtesting convention compliance.
    Reviews architecture, patterns, and standards adherence.
    """
    
    def __init__(self):
        self.console = console
        self.violations = []
        self.recommendations = []
        self.architecture_issues = []
        self.data_handling_issues = []
        self.execution_issues = []
        
    def audit_sage_architecture_compliance(self) -> Dict:
        """Audit SAGE architecture against NT backtesting conventions"""
        console.print("🔍 [red]ADVERSARIAL AUDIT: Architecture Compliance[/red]")
        
        violations = []
        
        # CRITICAL VIOLATION 1: No BacktestEngine Integration
        violations.append({
            'severity': 'CRITICAL',
            'category': 'Architecture',
            'violation': 'SAGE models bypass BacktestEngine entirely',
            'details': 'SAGE validation runs standalone without NT BacktestEngine, BacktestNode, or backtesting infrastructure',
            'nt_convention': 'All backtesting should use BacktestEngine (low-level) or BacktestNode (high-level)',
            'current_approach': 'Direct model execution without NT backtesting framework',
            'impact': 'Cannot leverage NT execution simulation, venue modeling, portfolio tracking, or risk management'
        })
        
        # CRITICAL VIOLATION 2: No Strategy Pattern
        violations.append({
            'severity': 'CRITICAL', 
            'category': 'Architecture',
            'violation': 'SAGE models are not NT Strategy implementations',
            'details': 'Models inherit from custom wrappers, not nautilus_trader.trading.Strategy',
            'nt_convention': 'Trading logic should inherit from Strategy class with on_bar(), on_data() methods',
            'current_approach': 'Custom wrapper classes with generate_factors() methods',
            'impact': 'Cannot use NT strategy lifecycle, event handling, or portfolio integration'
        })
        
        # CRITICAL VIOLATION 3: No Venue Configuration
        violations.append({
            'severity': 'CRITICAL',
            'category': 'Architecture', 
            'violation': 'No NT venue setup or configuration',
            'details': 'Missing venue initialization with proper OmsType, AccountType, book_type',
            'nt_convention': 'engine.add_venue() with CASH/MARGIN account, NETTING/HEDGING OMS, L1/L2/L3 book_type',
            'current_approach': 'Direct data processing without venue modeling',
            'impact': 'No execution simulation, order management, or account modeling'
        })
        
        # HIGH VIOLATION 4: No Instrument Integration
        violations.append({
            'severity': 'HIGH',
            'category': 'Data',
            'violation': 'Raw pandas DataFrames instead of NT Instrument objects',
            'details': 'Using generic OHLCV data without NT instrument specifications',
            'nt_convention': 'engine.add_instrument() with proper InstrumentId, price_precision, size_precision',
            'current_approach': 'Pandas DataFrames with basic OHLCV columns',
            'impact': 'No proper price/size validation, tick handling, or instrument-specific logic'
        })
        
        # HIGH VIOLATION 5: No Bar Object Usage  
        violations.append({
            'severity': 'HIGH',
            'category': 'Data',
            'violation': 'Raw data arrays instead of NT Bar objects',
            'details': 'Processing pandas Series/arrays directly without NT data types',
            'nt_convention': 'Use BarDataWrangler to create Bar objects with proper BarType, timestamps',
            'current_approach': 'Direct pandas Series manipulation',
            'impact': 'Missing NT timestamp handling, bar aggregation, and data validation'
        })
        
        # MEDIUM VIOLATION 6: No Portfolio Integration
        violations.append({
            'severity': 'MEDIUM',
            'category': 'Execution',
            'violation': 'No portfolio or account tracking',
            'details': 'Model outputs not connected to portfolio performance or risk metrics',
            'nt_convention': 'Portfolio automatically tracks PnL, positions, margin, risk metrics',
            'current_approach': 'Isolated factor generation without portfolio context',
            'impact': 'Cannot evaluate trading performance or risk-adjusted returns'
        })
        
        # MEDIUM VIOLATION 7: No Message Bus Usage
        violations.append({
            'severity': 'MEDIUM',
            'category': 'Architecture',
            'violation': 'No NT message bus or event system usage',
            'details': 'Direct method calls instead of event-driven architecture',
            'nt_convention': 'MessageBus handles all data distribution and command routing',
            'current_approach': 'Direct function calls between components',
            'impact': 'Cannot leverage NT event logging, replay, or distributed processing'
        })
        
        return {
            'total_violations': len(violations),
            'critical_violations': len([v for v in violations if v['severity'] == 'CRITICAL']),
            'high_violations': len([v for v in violations if v['severity'] == 'HIGH']),
            'medium_violations': len([v for v in violations if v['severity'] == 'MEDIUM']),
            'violations': violations
        }
    
    def audit_data_handling_compliance(self) -> Dict:
        """Audit data handling against NT conventions"""
        console.print("🔍 [red]ADVERSARIAL AUDIT: Data Handling Compliance[/red]")
        
        violations = []
        
        # CRITICAL VIOLATION 1: No Data Wrangling
        violations.append({
            'severity': 'CRITICAL',
            'category': 'Data',
            'violation': 'Raw DSM data used without NT data wrangling',
            'details': 'Polars DataFrames converted to pandas without BarDataWrangler processing',
            'nt_convention': 'Use BarDataWrangler.process() to create properly formatted Bar objects',
            'current_approach': 'Direct polars.to_pandas() conversion',
            'impact': 'Missing NT timestamp validation, bar aggregation, and data quality checks'
        })
        
        # HIGH VIOLATION 2: Improper Timestamp Handling
        violations.append({
            'severity': 'HIGH',
            'category': 'Data',
            'violation': 'Timestamps not following NT ts_event/ts_init convention',
            'details': 'Missing proper ts_event (bar close time) and ts_init (data arrival time) handling',
            'nt_convention': 'Bar timestamps must represent close time, use ts_init_delta for arrival simulation',
            'current_approach': 'Generic timestamp column without NT timestamp semantics',
            'impact': 'Potential look-ahead bias and incorrect event sequencing'
        })
        
        # HIGH VIOLATION 3: No Bar Type Specification
        violations.append({
            'severity': 'HIGH',
            'category': 'Data',
            'violation': 'No BarType specification for data streams',
            'details': 'Missing BarType with proper step, aggregation, price_type specification',
            'nt_convention': 'BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL")',
            'current_approach': 'Generic OHLCV data without bar type metadata',
            'impact': 'Cannot properly route data or configure venue execution'
        })
        
        # MEDIUM VIOLATION 4: Missing Data Validation
        violations.append({
            'severity': 'MEDIUM', 
            'category': 'Data',
            'violation': 'No NT data validation or quality checks',
            'details': 'Missing price precision, size increment, and data consistency validation',
            'nt_convention': 'NT automatically validates data against instrument specifications',
            'current_approach': 'Basic pandas data quality checks only',
            'impact': 'May process invalid data that would be rejected by NT'
        })
        
        return {
            'total_violations': len(violations),
            'violations': violations
        }
    
    def audit_execution_compliance(self) -> Dict:
        """Audit execution patterns against NT conventions"""
        console.print("🔍 [red]ADVERSARIAL AUDIT: Execution Pattern Compliance[/red]")
        
        violations = []
        
        # CRITICAL VIOLATION 1: No Order Management
        violations.append({
            'severity': 'CRITICAL',
            'category': 'Execution',
            'violation': 'No order generation or management system',
            'details': 'Models generate factors but no orders, fills, or position tracking',
            'nt_convention': 'Strategies submit orders via self.submit_order(), get fills via on_order_filled()',
            'current_approach': 'Factor generation only, no trading simulation',
            'impact': 'Cannot evaluate actual trading performance or execution costs'
        })
        
        # CRITICAL VIOLATION 2: No Fill Simulation
        violations.append({
            'severity': 'CRITICAL',
            'category': 'Execution', 
            'violation': 'No fill model or execution simulation',
            'details': 'Missing FillModel for realistic order execution simulation',
            'nt_convention': 'FillModel with prob_fill_on_limit, prob_slippage for realistic fills',
            'current_approach': 'No execution simulation whatsoever',
            'impact': 'Cannot assess strategy under realistic market conditions'
        })
        
        # HIGH VIOLATION 3: No Risk Management
        violations.append({
            'severity': 'HIGH',
            'category': 'Execution',
            'violation': 'No risk engine or position sizing',
            'details': 'No risk limits, position sizing, or portfolio risk management',
            'nt_convention': 'RiskEngine enforces position limits, risk checks before order submission',
            'current_approach': 'No risk management framework',
            'impact': 'Cannot evaluate risk-adjusted performance or maximum drawdown'
        })
        
        return {
            'total_violations': len(violations),
            'violations': violations
        }
    
    def generate_compliance_recommendations(self, architecture_audit: Dict, data_audit: Dict, execution_audit: Dict) -> List[Dict]:
        """Generate actionable recommendations for NT compliance"""
        
        recommendations = [
            {
                'priority': 'CRITICAL',
                'title': 'Implement NT BacktestEngine Integration',
                'description': 'Refactor SAGE validation to use BacktestEngine with proper venue setup',
                'implementation': '''
# Create proper NT backtest setup
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.enums import OmsType, AccountType

engine = BacktestEngine(config=BacktestEngineConfig())
engine.add_venue(
    venue=Venue("BINANCE"),
    oms_type=OmsType.NETTING, 
    account_type=AccountType.CASH,
    starting_balances=[Money(10_000, USDT)]
)
''',
                'impact': 'Enables full NT backtesting capabilities and realistic execution simulation'
            },
            
            {
                'priority': 'CRITICAL',
                'title': 'Convert SAGE Models to NT Strategies',
                'description': 'Refactor AlphaForge, catch22, tsfresh wrappers to inherit from Strategy',
                'implementation': '''
from nautilus_trader.trading import Strategy

class AlphaForgeStrategy(Strategy):
    def __init__(self, config: AlphaForgeConfig):
        super().__init__(config)
        self.alphaforge = AlphaForgeWrapper()
    
    def on_bar(self, bar: Bar) -> None:
        factors = self.alphaforge.generate_factors(bar)
        # Generate orders based on factors
        if factors['signal'] > threshold:
            self.submit_order(self.order_factory.market(...))
''',
                'impact': 'Enables proper NT event handling and portfolio integration'
            },
            
            {
                'priority': 'HIGH', 
                'title': 'Implement NT Data Wrangling',
                'description': 'Use BarDataWrangler to properly format DSM data into NT Bar objects',
                'implementation': '''
from nautilus_trader.persistence.wranglers import BarDataWrangler

# Proper data wrangling
wrangler = BarDataWrangler(
    bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL"),
    instrument=instrument
)
bars = wrangler.process(df, ts_init_delta=60_000_000_000)  # 1 minute in nanoseconds
engine.add_data(bars)
''',
                'impact': 'Ensures proper timestamp handling and prevents look-ahead bias'
            },
            
            {
                'priority': 'HIGH',
                'title': 'Add Proper Instrument Configuration',
                'description': 'Configure instruments with proper specifications for BTCUSDT perpetual futures',
                'implementation': '''
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.identifiers import InstrumentId

instrument = CryptoPerpetual(
    instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
    native_symbol="BTCUSDT",
    base_currency=BTC,
    quote_currency=USDT,
    price_precision=2,
    size_precision=6,
    tick_size=Decimal("0.01"),
    lot_size=Decimal("0.000001"),
    # ... other perpetual futures specs
)
engine.add_instrument(instrument)
''',
                'impact': 'Enables proper price/size validation and instrument-specific behavior'
            },
            
            {
                'priority': 'MEDIUM',
                'title': 'Implement Portfolio Performance Tracking',
                'description': 'Use NT portfolio analyzer to track strategy performance metrics',
                'implementation': '''
# After backtest run
results = engine.portfolio.analyzer.get_performance_stats_pnls()
print(f"Total PnL: {results['PnL (total)']}")
print(f"Sharpe Ratio: {results['Sharpe Ratio']}")
print(f"Max Drawdown: {results['Max Drawdown']}")
''',
                'impact': 'Provides standard trading performance metrics and risk analysis'
            }
        ]
        
        return recommendations
    
    def run_comprehensive_audit(self) -> Dict:
        """Run complete adversarial audit of NT backtesting compliance"""
        console.print(Panel.fit(
            "⚔️ ADVERSARIAL NT BACKTESTING CONVENTION AUDIT ⚔️\n"
            "Critically examining SAGE model compliance with NautilusTrader patterns\n"
            "Identifying architecture violations and convention deviations",
            title="NT BACKTESTING COMPLIANCE REVIEW",
            border_style="red"
        ))
        
        # Run individual audits
        architecture_audit = self.audit_sage_architecture_compliance()
        data_audit = self.audit_data_handling_compliance() 
        execution_audit = self.audit_execution_compliance()
        
        # Generate recommendations
        recommendations = self.generate_compliance_recommendations(
            architecture_audit, data_audit, execution_audit
        )
        
        # Calculate overall compliance score
        total_violations = (
            architecture_audit['total_violations'] +
            data_audit['total_violations'] + 
            execution_audit['total_violations']
        )
        
        critical_violations = (
            architecture_audit.get('critical_violations', 0) +
            sum(1 for v in data_audit['violations'] if v['severity'] == 'CRITICAL') +
            sum(1 for v in execution_audit['violations'] if v['severity'] == 'CRITICAL')
        )
        
        # Generate comprehensive report
        self._generate_audit_report(
            architecture_audit, data_audit, execution_audit, 
            recommendations, total_violations, critical_violations
        )
        
        return {
            'total_violations': total_violations,
            'critical_violations': critical_violations,
            'architecture_audit': architecture_audit,
            'data_audit': data_audit,
            'execution_audit': execution_audit,
            'recommendations': recommendations,
            'compliance_score': max(0, 100 - (critical_violations * 25 + (total_violations - critical_violations) * 10))
        }
    
    def _generate_audit_report(self, architecture_audit: Dict, data_audit: Dict, 
                              execution_audit: Dict, recommendations: List[Dict],
                              total_violations: int, critical_violations: int):
        """Generate comprehensive audit report"""
        
        console.print("\n" + "="*80)
        console.print("⚔️ NT BACKTESTING CONVENTION AUDIT RESULTS")
        console.print("="*80)
        
        # Summary table
        summary_table = Table(title="🎯 Compliance Violation Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Critical", style="red")
        summary_table.add_column("High", style="yellow") 
        summary_table.add_column("Medium", style="blue")
        summary_table.add_column("Total", style="magenta")
        
        # Architecture violations
        arch_critical = architecture_audit.get('critical_violations', 0)
        arch_high = architecture_audit.get('high_violations', 0)
        arch_medium = architecture_audit.get('medium_violations', 0)
        arch_total = architecture_audit['total_violations']
        
        summary_table.add_row("Architecture", str(arch_critical), str(arch_high), str(arch_medium), str(arch_total))
        
        # Data violations
        data_critical = sum(1 for v in data_audit['violations'] if v['severity'] == 'CRITICAL')
        data_high = sum(1 for v in data_audit['violations'] if v['severity'] == 'HIGH')
        data_medium = sum(1 for v in data_audit['violations'] if v['severity'] == 'MEDIUM')
        
        summary_table.add_row("Data Handling", str(data_critical), str(data_high), str(data_medium), str(data_audit['total_violations']))
        
        # Execution violations
        exec_critical = sum(1 for v in execution_audit['violations'] if v['severity'] == 'CRITICAL')
        exec_high = sum(1 for v in execution_audit['violations'] if v['severity'] == 'HIGH')
        exec_medium = sum(1 for v in execution_audit['violations'] if v['severity'] == 'MEDIUM')
        
        summary_table.add_row("Execution", str(exec_critical), str(exec_high), str(exec_medium), str(execution_audit['total_violations']))
        
        # Totals
        total_high = arch_high + data_high + exec_high
        total_medium = arch_medium + data_medium + exec_medium
        summary_table.add_row("TOTAL", str(critical_violations), str(total_high), str(total_medium), str(total_violations))
        
        console.print(summary_table)
        
        # Detailed violations
        all_violations = []
        all_violations.extend(architecture_audit['violations'])
        all_violations.extend(data_audit['violations'])
        all_violations.extend(execution_audit['violations'])
        
        # Group by severity
        critical_viols = [v for v in all_violations if v['severity'] == 'CRITICAL']
        
        if critical_viols:
            console.print(f"\n💀 CRITICAL VIOLATIONS ({len(critical_viols)}):")
            for i, violation in enumerate(critical_viols, 1):
                console.print(f"\n{i}. [red]{violation['violation']}[/red]")
                console.print(f"   Category: {violation['category']}")
                console.print(f"   Details: {violation['details']}")
                console.print(f"   NT Convention: {violation['nt_convention']}")
                console.print(f"   Current Approach: {violation['current_approach']}")
                console.print(f"   Impact: {violation['impact']}")
        
        # Priority recommendations
        console.print(f"\n🔧 PRIORITY RECOMMENDATIONS:")
        critical_recs = [r for r in recommendations if r['priority'] == 'CRITICAL']
        for i, rec in enumerate(critical_recs, 1):
            console.print(f"\n{i}. [bright_red]{rec['title']}[/bright_red]")
            console.print(f"   {rec['description']}")
            console.print(f"   Implementation Preview:")
            console.print(f"   {rec['implementation'][:200]}...")
            console.print(f"   Impact: {rec['impact']}")
        
        # Final verdict
        console.print("\n" + "="*80)
        console.print("⚖️ FINAL NT COMPLIANCE VERDICT")
        console.print("="*80)
        
        compliance_score = max(0, 100 - (critical_violations * 25 + (total_violations - critical_violations) * 10))
        
        if compliance_score >= 90:
            verdict = "🎉 EXCELLENT - High NT backtesting compliance"
            color = "bright_green"
        elif compliance_score >= 70:
            verdict = "✅ GOOD - Minor NT convention violations"
            color = "green"
        elif compliance_score >= 50:
            verdict = "⚠️ POOR - Multiple critical NT violations"
            color = "yellow"
        else:
            verdict = "💀 CRITICAL - Fundamental NT architecture violations"
            color = "red"
        
        console.print(Panel.fit(
            f"NT Backtesting Compliance Score: {compliance_score:.1f}%\n"
            f"Total Violations: {total_violations} (Critical: {critical_violations})\n"
            f"Verdict: {verdict}\n\n"
            f"RECOMMENDATION: Complete architectural refactor required\n"
            f"to align with NautilusTrader backtesting conventions.",
            title="NT COMPLIANCE AUDIT VERDICT",
            border_style=color
        ))

def main():
    """Run NT backtesting convention compliance audit"""
    auditor = NautilusBacktestConventionAuditor()
    results = auditor.run_comprehensive_audit()
    return results

if __name__ == "__main__":
    main()