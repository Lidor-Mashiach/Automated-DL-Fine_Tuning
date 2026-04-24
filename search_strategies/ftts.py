"""
FTTS: Fine-Tuning Tree Search
-----------------------------
The main search strategy. Builds a tree of experiments where each node is
a trial and each edge is an Action that transformed a parent's hyperparameters
into a child's.

Algorithm:
  1. Start from a root trial using initial_value for every tunable parameter.
  2. After each trial: Analyzer produces prioritized Actions.
  3. For each Action, a queue entry is pushed with:
         child_score = parent_quality_score * action_priority
  4. For the next trial: pop the best queue entry, apply its Action to the
     parent's hyperparameters, run the resulting trial.
  5. Repeat until a stop condition fires.

Adaptive step sizes:
  * When an action succeeds (child beats parent), the step grows for future
    applications of that action type (multiplier *= successful_step_boost).
  * When an action fails, the step shrinks (multiplier *= failed_step_shrink).
  * Steps are clamped between min_step_factor and max_step_factor.
  * This gives "gentle but accelerating" search that doesn't get stuck.
"""

import math
from typing import Any

from core.analyzer import Action
from search_strategies.experiment_tree import ExperimentTree, Node


# =============================================================================
# Default step sizes per action type
# =============================================================================
# These are baseline multipliers. When the action involves a numeric parameter,
# the new value = current * step_factor (for "increase_*") or
#                  current / step_factor (for "decrease_*").
_DEFAULT_STEP_FACTORS = {
    # LR is multiplicative and spans orders of magnitude
    "increase_lr": 3.0,
    "decrease_lr": 3.0,
    # Weight decay also multiplicative
    "increase_weight_decay": 3.0,
    "decrease_weight_decay": 3.0,
    # Dropout is additive (0.1 at a time)
    "increase_dropout": 0.1,
    "decrease_dropout": 0.1,
    # Warmup additive
    "increase_warmup": 500,
}


class FTTS:
    """Fine-Tuning Tree Search strategy."""

    def __init__(self, config_manager, strategy_config: dict):
        """
        Args:
            config_manager: ConfigManager for the architecture.
            strategy_config: dict from configs/strategies/ftts.yaml.
        """
        self.cm = config_manager
        self.cfg = strategy_config
        self.tree = ExperimentTree()

        # Adaptive step control
        step_cfg = strategy_config.get("step_control", {}) or {}
        self.successful_step_boost = float(step_cfg.get("successful_step_boost", 1.5))
        self.failed_step_shrink = float(step_cfg.get("failed_step_shrink", 0.5))
        self.min_step_factor = float(step_cfg.get("min_step_factor", 1.2))
        self.max_step_factor = float(step_cfg.get("max_step_factor", 5.0))

        # Track per-action-type step multipliers (start at 1.0)
        self._step_multipliers: dict[str, float] = {}

        self._root_registered = False

    # ---------------------------------------------- public API

    def initial_hyperparameters(self) -> dict:
        """Build the root hyperparameters from initial_value fields of the YAML."""
        hp = {}
        for p in self.cm.active_parameters():
            hp[p["name"]] = self._resolve_initial(p)
            # Pass along extras (e.g., early_stopping's patience)
            for k, v in p["extras"].items():
                hp[f"{p['name']}__{k}"] = v
        return hp

    def mark_root(self, trial_id: str, hp: dict) -> None:
        """Register the root trial in the tree (called by Orchestrator)."""
        self.tree.register_root(trial_id, hp,
                                rationale="Root trial: starting from config initial values.")
        self._root_registered = True

    def propose_next(self) -> tuple[str, dict, Any, str] | None:
        """
        Pop the next (parent_id, action) pair from the tree and produce the
        hyperparameters for the child.

        Returns:
            (parent_id, child_hp, action_applied, rationale) or None if queue empty.
        """
        popped = self.tree.pop_best_pending()
        if popped is None:
            return None
        parent_id, action = popped
        parent = self.tree.get_node(parent_id)
        if parent is None:
            return None

        child_hp = dict(parent.hyperparameters)
        changes_made = self._apply_action(child_hp, action)

        if not changes_made:
            # Action could not be applied (e.g., param disabled). Try next one.
            return self.propose_next()

        rationale = (
            f"Based on {parent_id} (verdict={parent.verdict}, quality="
            f"{parent.quality_score:.3f}). Applied action '{action.type}' "
            f"[priority={action.priority:.2f}]: {action.reason} "
            f"-> {changes_made}"
        )
        return parent_id, child_hp, action, rationale

    def register_completed(self, trial_id: str, parent_id: str | None,
                           hp: dict, quality_breakdown, diagnosis,
                           status: str, rationale: str,
                           applied_action=None) -> None:
        """Register a completed trial in the tree."""
        self.tree.register_completed(
            trial_id=trial_id,
            parent_id=parent_id,
            hyperparameters=hp,
            quality_score=quality_breakdown.total if quality_breakdown else 0.0,
            verdict=diagnosis.verdict,
            actions=diagnosis.actions,
            raw_best=quality_breakdown.raw_best_smoothed if quality_breakdown else 0.0,
            smoothed_best=quality_breakdown.raw_best_smoothed if quality_breakdown else 0.0,
            status=status,
            rationale=rationale,
            applied_action=applied_action,
        )

        # Update step multipliers based on outcome
        if applied_action is not None and parent_id is not None:
            parent = self.tree.get_node(parent_id)
            if parent is not None:
                # Compare quality with parent
                current_q = quality_breakdown.total if quality_breakdown else 0.0
                improved = current_q > parent.quality_score
                self._update_step_multiplier(applied_action.type, improved)

    # ---------------------------------------------- private helpers

    def _resolve_initial(self, p: dict) -> Any:
        """Determine the root value for a parameter."""
        if p["initial_value"] is not None:
            return p["initial_value"]
        # null -> system picks middle
        if p["choices"] is not None:
            mid = p["choices"][len(p["choices"]) // 2]
            return mid
        if p["range"] is not None:
            lo, hi = p["range"]
            if p["log"]:
                return math.exp((math.log(max(lo, 1e-12)) +
                                 math.log(max(hi, 1e-12))) / 2)
            if isinstance(lo, int) and isinstance(hi, int):
                return (lo + hi) // 2
            return (lo + hi) / 2
        return None

    def _step_factor(self, action_type: str) -> float:
        """Current multiplier for this action type, based on adaptive history."""
        base = _DEFAULT_STEP_FACTORS.get(action_type, 1.5)
        mult = self._step_multipliers.get(action_type, 1.0)
        factor = base * mult
        return max(self.min_step_factor, min(self.max_step_factor, factor))

    def _update_step_multiplier(self, action_type: str, improved: bool):
        """If the action worked, grow its step. If it failed, shrink."""
        current = self._step_multipliers.get(action_type, 1.0)
        if improved:
            self._step_multipliers[action_type] = current * self.successful_step_boost
        else:
            self._step_multipliers[action_type] = current * self.failed_step_shrink

    def _apply_action(self, hp: dict, action: Action) -> str | None:
        """
        Apply a single Action to the hyperparameters dict in place.
        Returns a human-readable description of the change, or None if the
        action couldn't be applied.
        """
        t = action.type

        # ----- learning rate -----
        if t == "decrease_lr":
            return self._adjust_continuous(hp, "learning_rate",
                                           direction=-1, action_type=t)
        if t == "increase_lr":
            return self._adjust_continuous(hp, "learning_rate",
                                           direction=+1, action_type=t)

        # ----- dropout (additive) -----
        if t == "increase_dropout":
            return self._adjust_additive(hp, "dropout",
                                          delta_sign=+1, action_type=t)
        if t == "decrease_dropout":
            return self._adjust_additive(hp, "dropout",
                                          delta_sign=-1, action_type=t)

        # ----- weight decay -----
        if t == "increase_weight_decay":
            return self._adjust_continuous(hp, "weight_decay",
                                           direction=+1, action_type=t)
        if t == "decrease_weight_decay":
            return self._adjust_continuous(hp, "weight_decay",
                                           direction=-1, action_type=t)

        # ----- capacity: depth -----
        if t == "add_depth":
            return self._adjust_int_param(hp, ["num_hidden_layers", "num_layers",
                                                "num_encoder_layers", "num_conv_blocks"],
                                           direction=+1)
        if t == "reduce_depth":
            return self._adjust_int_param(hp, ["num_hidden_layers", "num_layers",
                                                "num_encoder_layers", "num_conv_blocks"],
                                           direction=-1)

        # ----- capacity: width -----
        if t == "add_width":
            return self._adjust_choice_param(hp, ["hidden_size", "d_model",
                                                   "fc_size", "base_filters"],
                                              direction=+1)
        if t == "reduce_width":
            return self._adjust_choice_param(hp, ["hidden_size", "d_model",
                                                   "fc_size", "base_filters"],
                                              direction=-1)

        # ----- layer shape -----
        if t == "change_layer_shape":
            return self._cycle_choice(hp, "layer_shape")

        # ----- activation -----
        if t == "change_activation":
            return self._cycle_choice(hp, "activation")

        # ----- optimizer -----
        if t == "change_optimizer":
            new_val = action.suggested_value or "adamw"
            return self._set_choice(hp, "name", new_val)

        # ----- gradient clipping -----
        if t == "add_gradient_clipping":
            return self._set_value(hp, "gradient_clipping",
                                    action.suggested_value or 1.0)

        # ----- lr scheduler -----
        if t == "add_lr_scheduler":
            new_val = action.suggested_value or "cosine"
            return self._set_choice(hp, "lr_scheduler", new_val)

        # ----- augmentation -----
        if t == "increase_augmentation":
            return self._advance_choice(hp, "data_augmentation",
                                         order=["none", "light", "medium"])

        # ----- batch size -----
        if t == "reduce_batch_size":
            return self._adjust_choice_param(hp, ["batch_size"], direction=-1)
        if t == "increase_batch_size":
            return self._adjust_choice_param(hp, ["batch_size"], direction=+1)

        # ----- label smoothing, mixup: enable by setting value > 0 -----
        if t == "enable_label_smoothing":
            return self._set_value(hp, "label_smoothing", 0.1)
        if t == "enable_mixup":
            return self._set_value(hp, "mixup", 0.2)

        # ----- warmup -----
        if t == "increase_warmup":
            return self._adjust_additive(hp, "lr_warmup", delta_sign=+1,
                                          action_type=t)

        # ----- attention dropout -----
        if t == "adjust_attention_dropout":
            return self._adjust_additive(hp, "attention_dropout",
                                          delta_sign=+1, action_type=t)

        # ----- bidirectional toggle -----
        if t == "toggle_bidirectional":
            current = hp.get("bidirectional", False)
            return self._set_value(hp, "bidirectional", not current)

        # ----- batch norm -----
        if t == "enable_batch_norm":
            return self._set_value(hp, "batch_norm", True)

        return None

    # -------------- parameter-level adjustment helpers --------------

    def _adjust_continuous(self, hp: dict, name: str, direction: int,
                           action_type: str) -> str | None:
        """For log-scale parameters (lr, weight_decay): multiply/divide by step factor."""
        p = self.cm.get_param(name)
        if p is None:
            return None
        current = hp.get(name)
        if current is None:
            return None
        factor = self._step_factor(action_type)
        new_val = current * factor if direction > 0 else current / factor
        # Clamp to range
        if p["range"]:
            lo, hi = p["range"]
            new_val = max(lo, min(hi, new_val))
        # Guard: no change means action didn't take
        if abs(new_val - current) / max(abs(current), 1e-12) < 1e-6:
            return None
        hp[name] = new_val
        return f"{name}: {current:.3e} -> {new_val:.3e}"

    def _adjust_additive(self, hp: dict, name: str, delta_sign: int,
                         action_type: str) -> str | None:
        """For parameters like dropout (0-0.5): add/subtract a fixed delta."""
        p = self.cm.get_param(name)
        if p is None:
            return None
        current = hp.get(name)
        if current is None:
            return None
        delta = _DEFAULT_STEP_FACTORS.get(action_type, 0.1) * delta_sign
        new_val = current + delta
        if p["range"]:
            lo, hi = p["range"]
            new_val = max(lo, min(hi, new_val))
        if abs(new_val - current) < 1e-9:
            return None
        hp[name] = new_val
        if isinstance(current, float):
            return f"{name}: {current:.3f} -> {new_val:.3f}"
        return f"{name}: {current} -> {new_val}"

    def _adjust_int_param(self, hp: dict, candidate_names: list,
                          direction: int) -> str | None:
        """For integer parameters (depth): try each candidate name in order."""
        for name in candidate_names:
            p = self.cm.get_param(name)
            if p is None or p["range"] is None:
                continue
            current = hp.get(name)
            if current is None:
                continue
            new_val = int(current) + direction
            lo, hi = p["range"]
            new_val = max(int(lo), min(int(hi), new_val))
            if new_val == current:
                continue
            hp[name] = new_val
            return f"{name}: {current} -> {new_val}"
        return None

    def _adjust_choice_param(self, hp: dict, candidate_names: list,
                             direction: int) -> str | None:
        """Move to the next/previous choice in the list."""
        for name in candidate_names:
            p = self.cm.get_param(name)
            if p is None or p["choices"] is None:
                continue
            choices = sorted(p["choices"])
            current = hp.get(name)
            if current is None:
                continue
            try:
                idx = choices.index(current)
            except ValueError:
                idx = len(choices) // 2
            new_idx = max(0, min(len(choices) - 1, idx + direction))
            if new_idx == idx:
                continue
            hp[name] = choices[new_idx]
            return f"{name}: {current} -> {hp[name]}"
        return None

    def _cycle_choice(self, hp: dict, name: str) -> str | None:
        """Pick the next choice different from the current value."""
        p = self.cm.get_param(name)
        if p is None or p["choices"] is None:
            return None
        current = hp.get(name)
        for c in p["choices"]:
            if c != current:
                hp[name] = c
                return f"{name}: {current} -> {c}"
        return None

    def _set_choice(self, hp: dict, name: str, value: Any) -> str | None:
        """Set a parameter to a specific choice, if the choice is allowed."""
        p = self.cm.get_param(name)
        if p is None:
            return None
        if p["choices"] is not None and value not in p["choices"]:
            return None
        current = hp.get(name)
        if current == value:
            return None
        hp[name] = value
        return f"{name}: {current} -> {value}"

    def _set_value(self, hp: dict, name: str, value: Any) -> str | None:
        """Set a parameter to a specific value (honoring range if applicable)."""
        p = self.cm.get_param(name)
        if p is None:
            return None
        if p["range"] is not None:
            lo, hi = p["range"]
            value = max(lo, min(hi, value))
        current = hp.get(name)
        if current == value:
            return None
        hp[name] = value
        return f"{name}: {current} -> {value}"

    def _advance_choice(self, hp: dict, name: str, order: list) -> str | None:
        """Move forward in an ordered list of choices (e.g. none -> light -> medium)."""
        p = self.cm.get_param(name)
        if p is None:
            return None
        current = hp.get(name)
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        new_idx = min(idx + 1, len(order) - 1)
        if order[new_idx] == current:
            return None
        hp[name] = order[new_idx]
        return f"{name}: {current} -> {order[new_idx]}"
