"""
Test reporting utilities
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class TestReportGenerator:
    """Generate test reports"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def add_result(
        self,
        phase: str,
        test_name: str,
        status: str,  # 'passed', 'failed', 'skipped'
        metrics: Dict[str, Any],
        duration_ms: float,
        error_message: str = None
    ):
        """Add a test result"""
        self.results.append({
            "phase": phase,
            "test_name": test_name,
            "status": status,
            "metrics": metrics,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_json_report(self, filename: str = None) -> str:
        """Generate JSON report"""
        if filename is None:
            filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(),
            "results": self.results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def generate_markdown_report(self, filename: str = None) -> str:
        """Generate Markdown report"""
        if filename is None:
            filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        filepath = self.output_dir / filename
        summary = self._generate_summary()
        
        md = []
        md.append("# Live Talker Test Report")
        md.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("\n## Summary")
        md.append(f"\n- **Total Tests**: {summary['total']}")
        md.append(f"- **Passed**: {summary['passed']} ✅")
        md.append(f"- **Failed**: {summary['failed']} ❌")
        md.append(f"- **Skipped**: {summary['skipped']} ⏭️")
        md.append(f"- **Pass Rate**: {summary['pass_rate']:.1%}")
        md.append(f"- **Total Duration**: {summary['total_duration_ms']/1000:.2f}s")
        
        # Phase breakdown
        md.append("\n## Phase Results")
        for phase, stats in summary['by_phase'].items():
            md.append(f"\n### {phase}")
            md.append(f"- Tests: {stats['total']}")
            md.append(f"- Passed: {stats['passed']}")
            md.append(f"- Failed: {stats['failed']}")
            md.append(f"- Duration: {stats['duration_ms']/1000:.2f}s")
        
        # Detailed results
        md.append("\n## Detailed Results")
        for result in self.results:
            status_icon = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⏭️"
            md.append(f"\n### {result['test_name']} {status_icon}")
            md.append(f"- Phase: {result['phase']}")
            md.append(f"- Duration: {result['duration_ms']:.2f}ms")
            
            if result['metrics']:
                md.append("- Metrics:")
                for key, value in result['metrics'].items():
                    md.append(f"  - {key}: {value}")
            
            if result['error_message']:
                md.append(f"- Error: `{result['error_message']}`")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        
        return str(filepath)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['status'] == 'passed')
        failed = sum(1 for r in self.results if r['status'] == 'failed')
        skipped = sum(1 for r in self.results if r['status'] == 'skipped')
        
        # Group by phase
        by_phase = {}
        for result in self.results:
            phase = result['phase']
            if phase not in by_phase:
                by_phase[phase] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'duration_ms': 0
                }
            by_phase[phase]['total'] += 1
            by_phase[phase]['duration_ms'] += result['duration_ms']
            if result['status'] == 'passed':
                by_phase[phase]['passed'] += 1
            elif result['status'] == 'failed':
                by_phase[phase]['failed'] += 1
            else:
                by_phase[phase]['skipped'] += 1
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'pass_rate': passed / total if total > 0 else 0,
            'total_duration_ms': sum(r['duration_ms'] for r in self.results),
            'by_phase': by_phase
        }


def format_metric(name: str, value: float, unit: str = "", threshold: float = None) -> str:
    """Format a metric for display"""
    result = f"{name}: {value:.2f}{unit}"
    if threshold is not None:
        status = "✅" if value <= threshold else "❌"
        result += f" (threshold: {threshold}{unit}) {status}"
    return result


def generate_comparison_table(
    headers: List[str],
    rows: List[List[Any]],
    align: str = "left"
) -> str:
    """Generate a markdown comparison table"""
    lines = []
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    
    # Separator
    if align == "center":
        sep = "|" + "|".join([":---:"] * len(headers)) + "|"
    elif align == "right":
        sep = "|" + "|".join(["---:"] * len(headers)) + "|"
    else:
        sep = "|" + "|".join([":---"] * len(headers)) + "|"
    lines.append(sep)
    
    # Rows
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join(lines)
