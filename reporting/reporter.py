"""
Reporter
--------
Writes a single unified text report (`report.txt`) documenting every trial
in order. The report is updated after every trial, so even if the run is
interrupted, the report reflects everything done so far.

Per-trial block structure:

    Trial #N  (trial_id: TXXXX)
        Parent   : parent_id or "root"
        Rationale: why this trial was run
        Parameters    (grouped by config section)
        Results       (status, epochs, raw + smoothed best, runtime)
        Quality       (total + 4 components)
        Diagnosis     (verdict + observations)
        Actions       (what the Analyzer suggests for children)
        Conclusion
"""

from pathlib import Path


# Groups from YAML, in display order with readable labels
_PARAM_SECTIONS = {
    "parameters":    "Architecture parameters",
    "architectures": "Structural variants",
    "methods":       "Regularization / stability methods",
    "optimization":  "Optimization",
    "training":      "Training",
}


class Reporter:
    """Writes a cumulative text report."""

    def __init__(self, report_path, run_config: dict, config_manager):
        self.path = Path(report_path)
        self.run_config = run_config
        self.cm = config_manager
        self._trial_blocks: list[str] = []
        self._write_header()

    # ---------------------------------------------------------------- header

    def _write_header(self):
        lines = []
        lines.append("=" * 80)
        lines.append(" AutoTune-NN  -  Experiment Report ".center(80, "="))
        lines.append("=" * 80)
        lines.append("")
        lines.append("Run configuration:")
        lines.append("-" * 40)
        for k, v in self.run_config.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk:<24}: {vv}")
            else:
                lines.append(f"  {k:<28}: {v}")
        lines.append("")
        lines.append("=" * 80)
        lines.append(" Trial-by-trial log ".center(80, "="))
        lines.append("=" * 80)
        lines.append("")
        self._header = "\n".join(lines) + "\n"
        self._flush()

    # ------------------------------------------------------ add a trial

    def add_trial(self, trial_index: int, result, diagnosis,
                  quality_breakdown=None, rationale: str = "",
                  parent_id: str | None = None) -> None:
        """Append a fully-formatted trial block and flush to disk."""
        lines = []
        lines.append("#" * 80)
        lines.append(f"  Trial #{trial_index}   (trial_id: {result.trial_id})")
        lines.append("#" * 80)
        lines.append("")

        # Parent + rationale
        parent_display = parent_id or result.parent_trial_id
        lines.append(f"  [Parent]     {parent_display if parent_display else 'root (initial trial)'}")
        if rationale:
            lines.append(f"  [Rationale]  {rationale}")
        lines.append("")

        # Parameters grouped by section
        lines.extend(self._format_params(result.hyperparameters))
        lines.append("")

        # Results
        lines.append("  [Results]")
        lines.append(f"    status              : {result.status}")
        lines.append(f"    epochs_completed    : {result.epochs_completed}")
        lines.append(f"    raw best metric     : {result.raw_best_metric:.6f} "
                     f"(epoch {result.raw_best_epoch})")
        lines.append(f"    smoothed best       : {result.best_metric:.6f} "
                     f"(epoch {result.best_epoch})")
        if result.train_loss_curve:
            lines.append(f"    train_loss range    : {result.train_loss_curve[0]:.4f} "
                         f"-> {result.train_loss_curve[-1]:.4f}")
        if result.val_loss_curve:
            lines.append(f"    val_loss range      : {result.val_loss_curve[0]:.4f} "
                         f"-> {result.val_loss_curve[-1]:.4f} "
                         f"(min={min(result.val_loss_curve):.4f})")
        lines.append(f"    duration (seconds)  : {result.duration_seconds:.1f}")
        if result.failure_reason:
            lines.append(f"    failure_reason      : {result.failure_reason}")
        lines.append("")

        # Quality breakdown
        if quality_breakdown is not None:
            b = quality_breakdown
            lines.append("  [Quality]")
            lines.append(f"    total                  : {b.total:.4f}")
            lines.append(f"    best_metric component  : {b.best_metric_component:.4f}")
            lines.append(f"    stability component    : {b.stability_component:.4f}")
            lines.append(f"    convergence_speed comp : {b.convergence_speed_component:.4f}")
            lines.append(f"    generalization_gap comp: {b.generalization_gap_component:.4f}")
            lines.append("")

        # Diagnosis
        lines.append("  [Diagnosis]")
        lines.append(f"    verdict: {diagnosis.verdict}")
        for obs in diagnosis.observations:
            lines.append(f"      - {obs}")
        lines.append("")

        # Actions for next trial
        lines.append("  [Actions suggested for next trial]")
        if not diagnosis.actions:
            lines.append("    (none - trial considered terminal for this branch)")
        else:
            for i, action in enumerate(diagnosis.actions, 1):
                param_str = f" [{action.target_param}]" if action.target_param else ""
                val_str = (f", suggested={action.suggested_value}"
                           if action.suggested_value is not None else "")
                lines.append(f"    {i}. priority={action.priority:.2f} "
                             f"type={action.type}{param_str}{val_str}")
                lines.append(f"       reason: {action.reason}")
        lines.append("")

        if diagnosis.conclusion:
            lines.append(f"  [Conclusion]  {diagnosis.conclusion}")
        lines.append("")
        lines.append("")

        self._trial_blocks.append("\n".join(lines))
        self._flush()

    # ---------------------------------------------------------- finalize

    def finalize(self, summary: dict) -> None:
        """Write a final summary block and flush the report."""
        lines = []
        lines.append("=" * 80)
        lines.append(" Run summary ".center(80, "="))
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  Total trials        : {summary['total_trials']}")
        lines.append(f"  Best trial          : {summary['best_trial_id']}")
        lines.append(f"  Best quality score  : {summary['best_quality']:.6f}")
        lines.append(f"  Best metric (raw)   : {summary['best_metric_raw']:.6f}")
        lines.append(f"  Best metric (smooth): {summary['best_metric_smoothed']:.6f}")
        lines.append(f"  Stop reason         : {summary['stop_reason']}")
        lines.append("")
        lines.append("=" * 80)
        self._footer = "\n".join(lines) + "\n"
        self._flush(include_footer=True)

    # ------------------------------------------------------------ internal

    def _flush(self, include_footer: bool = False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parts = [self._header] + self._trial_blocks
        if include_footer and hasattr(self, "_footer"):
            parts.append(self._footer)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    def _format_params(self, hp: dict) -> list[str]:
        """Group hyperparameters by their config section for a tidy display."""
        group_of: dict[str, str] = {p["name"]: p["group"]
                                     for p in self.cm.parameters}
        buckets: dict[str, list] = {g: [] for g in _PARAM_SECTIONS}
        extras: list = []
        for name, val in hp.items():
            if "__" in name:
                extras.append((name, val))
                continue
            g = group_of.get(name)
            if g in buckets:
                buckets[g].append((name, val))
            else:
                extras.append((name, val))

        lines = []
        for g, label in _PARAM_SECTIONS.items():
            items = buckets.get(g, [])
            if not items:
                continue
            lines.append(f"  [{label}]")
            for name, val in sorted(items):
                lines.append(f"    {name:<24}: {_format_value(val)}")
        if extras:
            lines.append("  [Extras]")
            for name, val in sorted(extras):
                lines.append(f"    {name:<24}: {_format_value(val)}")
        return lines


def _format_value(v) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.6g}"
    return str(v)
