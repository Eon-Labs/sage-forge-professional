#!/usr/bin/env python3
"""
🔍 Comprehensive Data Quality Diagnostics for SAGE-Forge

Identifies exact locations, patterns, and root causes of NaN values in OHLCV data.
Implements feedback-driven iterative refinement to converge on valid solutions.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add SAGE-Forge to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sage_forge.data.manager import ArrowDataManager
from sage_forge.data.enhanced_provider import EnhancedModernBarDataProvider
from sage_forge.market.binance_specs import BinanceSpecificationManager

console = Console()


class DataQualityDiagnostics:
    """🔍 Comprehensive data quality analysis and diagnostics."""
    
    def __init__(self):
        self.console = Console()
        self.data_manager = ArrowDataManager()
        self.specs_manager = BinanceSpecificationManager()
        
    def run_comprehensive_diagnostics(self, symbol="BTCUSDT", limit=500):
        """Run complete diagnostic pipeline with detailed analysis."""
        self.console.print(Panel.fit(
            "🔍 SAGE-Forge Data Quality Diagnostics\n"
            "Identifying root causes of NaN values",
            style="bold blue"
        ))
        
        # Step 1: Raw data analysis
        raw_data = self._fetch_raw_data(symbol, limit)
        if raw_data is None:
            return None
            
        # Step 2: Pipeline stage analysis
        pipeline_results = self._analyze_pipeline_stages(raw_data)
        
        # Step 3: NaN pattern analysis
        nan_analysis = self._analyze_nan_patterns(pipeline_results)
        
        # Step 4: Root cause identification
        root_causes = self._identify_root_causes(nan_analysis)
        
        # Step 5: Generate fix recommendations
        fix_recommendations = self._generate_fix_recommendations(root_causes)
        
        return {
            "raw_data": raw_data,
            "pipeline_results": pipeline_results,
            "nan_analysis": nan_analysis,
            "root_causes": root_causes,
            "fix_recommendations": fix_recommendations
        }
        
    def _fetch_raw_data(self, symbol, limit):
        """Fetch raw data with detailed logging."""
        self.console.print(f"[blue]📡 Fetching raw data for {symbol} (limit: {limit})[/blue]")
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=2)
            
            raw_data = self.data_manager.fetch_real_market_data(
                symbol=symbol,
                timeframe="1m",
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            
            if raw_data is not None:
                self.console.print(f"[green]✅ Raw data fetched: {len(raw_data)} rows[/green]")
                self._analyze_raw_data_quality(raw_data)
                return raw_data
            else:
                self.console.print("[red]❌ No raw data returned[/red]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]❌ Raw data fetch error: {e}[/red]")
            return None
            
    def _analyze_raw_data_quality(self, raw_data):
        """Analyze quality of raw data before processing."""
        self.console.print("[yellow]🔍 Analyzing raw data quality...[/yellow]")
        
        # Determine data type
        if hasattr(raw_data, "null_count"):  # Polars
            data_type = "Polars"
            total_rows = len(raw_data)
            null_counts = raw_data.null_count()
            null_dict = dict(zip(raw_data.columns, null_counts.row(0)))
        elif hasattr(raw_data, "isna"):  # Pandas
            data_type = "Pandas"
            total_rows = len(raw_data)
            null_counts = raw_data.isna().sum()
            null_dict = null_counts.to_dict()
        else:
            self.console.print(f"[red]❌ Unknown data type: {type(raw_data)}[/red]")
            return
            
        # Create detailed table
        table = Table(title="Raw Data Quality Analysis")
        table.add_column("Column", style="cyan")
        table.add_column("NaN Count", style="red")
        table.add_column("Completeness %", style="green")
        table.add_column("Data Type", style="yellow")
        
        core_columns = ["open", "high", "low", "close", "volume"]
        total_nulls = 0
        
        for col in raw_data.columns:
            null_count = null_dict.get(col, 0)
            completeness = ((total_rows - null_count) / total_rows * 100) if total_rows > 0 else 0
            
            # Get data type
            if hasattr(raw_data, "dtypes"):  # Polars
                dtype = str(raw_data.dtypes[raw_data.columns.index(col)])
            else:  # Pandas
                dtype = str(raw_data[col].dtype)
                
            # Highlight core columns
            style = "bold red" if col in core_columns and null_count > 0 else "white"
            table.add_row(col, str(null_count), f"{completeness:.2f}%", dtype, style=style)
            
            if col in core_columns:
                total_nulls += null_count
                
        self.console.print(table)
        self.console.print(f"[blue]📊 Raw data summary: {data_type}, {total_rows} rows, {total_nulls} core NaNs[/blue]")
        
    def _analyze_pipeline_stages(self, raw_data):
        """Analyze data quality at each pipeline stage."""
        self.console.print("[yellow]🔄 Analyzing pipeline stages...[/yellow]")
        
        results = {}
        
        # Stage 1: Raw data
        results["raw"] = self._get_data_quality_snapshot(raw_data, "Raw Data")
        
        # Stage 2: After DSM processing
        try:
            processed_data = self.data_manager.process_ohlcv_data(raw_data)
            results["processed"] = self._get_data_quality_snapshot(processed_data, "After DSM Processing")
        except Exception as e:
            self.console.print(f"[red]❌ DSM processing failed: {e}[/red]")
            results["processed"] = {"error": str(e)}
            
        # Stage 3: After enhanced provider processing (without validation)
        try:
            provider = EnhancedModernBarDataProvider(self.specs_manager)
            # Temporarily bypass validation to see processing effects
            validated_data = provider._validate_data_against_specs(processed_data)
            results["validated"] = self._get_data_quality_snapshot(validated_data, "After Validation")
        except Exception as e:
            self.console.print(f"[red]❌ Validation failed: {e}[/red]")
            results["validated"] = {"error": str(e)}
            
        return results
        
    def _get_data_quality_snapshot(self, data, stage_name):
        """Get comprehensive quality snapshot for a data stage."""
        if data is None:
            return {"error": "No data"}
            
        try:
            # Basic info
            total_rows = len(data)
            
            # NaN analysis
            if hasattr(data, "null_count"):  # Polars
                null_counts = data.null_count()
                null_dict = dict(zip(data.columns, null_counts.row(0)))
            elif hasattr(data, "isna"):  # Pandas
                null_counts = data.isna().sum()
                null_dict = null_counts.to_dict()
            else:
                return {"error": f"Unknown data type: {type(data)}"}
                
            # Core column analysis
            core_columns = ["open", "high", "low", "close", "volume"]
            core_nulls = sum(null_dict.get(col, 0) for col in core_columns if col in null_dict)
            
            # Data type analysis
            dtypes = {}
            if hasattr(data, "dtypes"):  # Polars
                dtypes = {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
            else:  # Pandas
                dtypes = {col: str(data[col].dtype) for col in data.columns}
                
            snapshot = {
                "stage": stage_name,
                "total_rows": total_rows,
                "total_columns": len(data.columns),
                "core_nulls": core_nulls,
                "total_nulls": sum(null_dict.values()),
                "null_by_column": null_dict,
                "data_types": dtypes,
                "columns": list(data.columns),
                "completeness_pct": ((total_rows - core_nulls) / total_rows * 100) if total_rows > 0 else 0
            }
            
            # Display snapshot
            self._display_quality_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            return {"error": str(e)}
            
    def _display_quality_snapshot(self, snapshot):
        """Display quality snapshot in formatted table."""
        if "error" in snapshot:
            self.console.print(f"[red]❌ {snapshot['stage']}: {snapshot['error']}[/red]")
            return
            
        panel_content = (
            f"📊 **{snapshot['stage']}**\n"
            f"Rows: {snapshot['total_rows']:,} | Columns: {snapshot['total_columns']}\n"
            f"Core NaNs: {snapshot['core_nulls']} | Total NaNs: {snapshot['total_nulls']}\n"
            f"Core Completeness: {snapshot['completeness_pct']:.6f}%"
        )
        
        style = "green" if snapshot['core_nulls'] == 0 else "red"
        self.console.print(Panel(panel_content, style=style))
        
    def _analyze_nan_patterns(self, pipeline_results):
        """Analyze patterns in NaN locations and distributions."""
        self.console.print("[yellow]🔍 Analyzing NaN patterns...[/yellow]")
        
        pattern_analysis = {}
        
        for stage, results in pipeline_results.items():
            if "error" in results:
                continue
                
            null_by_column = results.get("null_by_column", {})
            core_columns = ["open", "high", "low", "close", "volume"]
            
            # Analyze patterns
            patterns = {
                "uniform_distribution": self._check_uniform_nan_distribution(null_by_column, core_columns),
                "column_specific": self._identify_column_specific_nans(null_by_column, core_columns),
                "progression": self._analyze_nan_progression(pipeline_results, stage)
            }
            
            pattern_analysis[stage] = patterns
            
        return pattern_analysis
        
    def _check_uniform_nan_distribution(self, null_by_column, core_columns):
        """Check if NaNs are uniformly distributed across core columns."""
        core_nulls = [null_by_column.get(col, 0) for col in core_columns if col in null_by_column]
        
        if not core_nulls or all(n == 0 for n in core_nulls):
            return {"uniform": True, "pattern": "no_nans"}
            
        # Check if all core columns have same number of NaNs
        unique_counts = set(core_nulls)
        if len(unique_counts) == 1:
            return {"uniform": True, "pattern": "same_rows", "count": core_nulls[0]}
        else:
            return {"uniform": False, "pattern": "mixed", "distribution": dict(zip(["open", "high", "low", "close", "volume"], core_nulls))}
            
    def _identify_column_specific_nans(self, null_by_column, core_columns):
        """Identify which specific columns have NaN issues."""
        problematic_columns = []
        
        for col in core_columns:
            null_count = null_by_column.get(col, 0)
            if null_count > 0:
                problematic_columns.append({"column": col, "nan_count": null_count})
                
        return problematic_columns
        
    def _analyze_nan_progression(self, pipeline_results, current_stage):
        """Analyze how NaNs change through pipeline stages."""
        stage_order = ["raw", "processed", "validated"]
        current_index = stage_order.index(current_stage) if current_stage in stage_order else -1
        
        if current_index <= 0:
            return {"progression": "initial"}
            
        previous_stage = stage_order[current_index - 1]
        current_nulls = pipeline_results[current_stage].get("core_nulls", 0)
        previous_nulls = pipeline_results[previous_stage].get("core_nulls", 0)
        
        if current_nulls > previous_nulls:
            return {"progression": "introduced", "delta": current_nulls - previous_nulls}
        elif current_nulls < previous_nulls:
            return {"progression": "fixed", "delta": previous_nulls - current_nulls}
        else:
            return {"progression": "unchanged", "delta": 0}
            
    def _identify_root_causes(self, nan_analysis):
        """Identify potential root causes based on NaN patterns."""
        self.console.print("[yellow]🎯 Identifying root causes...[/yellow]")
        
        root_causes = []
        
        # Analyze each stage
        for stage, patterns in nan_analysis.items():
            if patterns.get("uniform", {}).get("pattern") == "same_rows":
                root_causes.append({
                    "stage": stage,
                    "cause": "uniform_row_corruption",
                    "description": "Same rows are NaN across all core columns",
                    "severity": "high",
                    "likely_fix": "Remove or interpolate specific corrupt rows"
                })
                
            if patterns.get("progression", {}).get("progression") == "introduced":
                root_causes.append({
                    "stage": stage,
                    "cause": "processing_introduced_nans",
                    "description": f"Processing stage introduced {patterns['progression']['delta']} NaNs",
                    "severity": "critical",
                    "likely_fix": "Fix processing logic that corrupts data"
                })
                
            column_issues = patterns.get("column_specific", [])
            if column_issues:
                for issue in column_issues:
                    root_causes.append({
                        "stage": stage,
                        "cause": "column_specific_corruption",
                        "description": f"Column '{issue['column']}' has {issue['nan_count']} NaN values",
                        "severity": "medium",
                        "likely_fix": f"Fix data source or processing for {issue['column']}"
                    })
                    
        return root_causes
        
    def _generate_fix_recommendations(self, root_causes):
        """Generate specific fix recommendations based on root causes."""
        self.console.print("[yellow]🔧 Generating fix recommendations...[/yellow]")
        
        recommendations = []
        
        # Priority fixes
        critical_causes = [c for c in root_causes if c["severity"] == "critical"]
        high_causes = [c for c in root_causes if c["severity"] == "high"]
        
        if critical_causes:
            recommendations.append({
                "priority": "immediate",
                "type": "processing_fix",
                "description": "Fix data processing pipeline introducing NaNs",
                "implementation": "Review DSM processing logic for data corruption",
                "code_location": "sage_forge/data/manager.py:process_ohlcv_data()"
            })
            
        if high_causes:
            recommendations.append({
                "priority": "high",
                "type": "data_cleaning",
                "description": "Remove or interpolate corrupt rows",
                "implementation": "Add pre-processing step to clean corrupt rows",
                "code_location": "sage_forge/data/enhanced_provider.py:_enforce_100_percent_data_quality()"
            })
            
        # Always add validation enhancement
        recommendations.append({
            "priority": "medium",
            "type": "validation_enhancement", 
            "description": "Add detailed NaN location logging",
            "implementation": "Log exact row indices and values of NaN data",
            "code_location": "sage_forge/data/enhanced_provider.py:_enforce_100_percent_data_quality()"
        })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            self.console.print(Panel(
                f"**Fix #{i} - {rec['priority'].upper()} Priority**\n"
                f"Type: {rec['type']}\n"
                f"Description: {rec['description']}\n"
                f"Implementation: {rec['implementation']}\n"
                f"Location: {rec['code_location']}",
                style=f"{'red' if rec['priority'] == 'immediate' else 'yellow' if rec['priority'] == 'high' else 'blue'}"
            ))
            
        return recommendations


def main():
    """Run comprehensive data quality diagnostics."""
    console.print(Panel.fit(
        "🔍 SAGE-Forge Data Quality Diagnostics\n"
        "Feedback-driven iterative refinement for NaN analysis",
        style="bold blue"
    ))
    
    # Change to correct directory
    sage_dir = Path("/Users/terryli/eon/nt/sage-forge-professional")
    if sage_dir.exists():
        import os
        os.chdir(sage_dir)
        console.print(f"[green]✅ Changed to: {sage_dir}[/green]")
    else:
        console.print(f"[red]❌ Directory not found: {sage_dir}[/red]")
        return
        
    # Run diagnostics
    diagnostics = DataQualityDiagnostics()
    results = diagnostics.run_comprehensive_diagnostics()
    
    if results:
        console.print(Panel.fit(
            "🎯 Diagnostics Complete\n"
            "Check output above for detailed analysis and fix recommendations",
            style="bold green"
        ))
    else:
        console.print(Panel.fit(
            "❌ Diagnostics Failed\n"
            "Unable to complete analysis - check errors above",
            style="bold red"
        ))


if __name__ == "__main__":
    main()