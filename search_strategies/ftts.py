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
    # Focal gamma additive (0.5 at a time)
    "increase_focal_gamma": 0.5,
    "decrease_focal_gamma": 0.5,
    # Text augmentation prob (0.05 at a time)
    "increase_text_augmentation": 0.05,
    # Embedding dropout (0.1 at a time)
    "increase_embedding_dropout": 0.1,
    # Stochastic depth (0.05 at a time)
    "increase_stochastic_depth": 0.05,
    # Adam betas (small adjustments)
    "adjust_adam_beta1": 0.02,
    "adjust_adam_beta2": 0.005,
}


# =============================================================================
# Discrete-step actions: actions whose target moves along an ordered list
# (choices) or an integer range. When DAG-dedup blocks one of these, instead
# of giving up we step further in the same direction (e.g. 128 -> 256 blocked,
# try 128 -> 512, then 128 -> 1024) until we find an unexplored value or hit
# the boundary.
#
# Maps action_type -> (candidate_param_names, direction).
# direction: +1 = increase, -1 = decrease.
# =============================================================================
_DISCRETE_STEP_ACTIONS = {
    "add_width":     (["hidden_size", "d_model", "fc_size", "base_filters"], +1),
    "reduce_width":  (["hidden_size", "d_model", "fc_size", "base_filters"], -1),
    "increase_sequence_length": (["sequence_length"], +1),
    "decrease_sequence_length": (["sequence_length"], -1),
    "increase_batch_size":      (["batch_size"], +1),
    "reduce_batch_size":        (["batch_size"], -1),
    "increase_teacher_forcing": (["teacher_forcing_ratio"], +1),
    "decrease_teacher_forcing": (["teacher_forcing_ratio"], -1),
    "add_depth":    (["num_hidden_layers", "num_layers",
                       "num_encoder_layers", "num_conv_blocks"], +1),
    "reduce_depth": (["num_hidden_layers", "num_layers",
                       "num_encoder_layers", "num_conv_blocks"], -1),
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

        # DAG dedup: track which HP signatures have already been explored
        # so different action paths don't re-evaluate the same configuration.
        self._seen_signatures: set[str] = set()

        # Value-coverage tracking (for diagnostics): for each tunable param,
        # which concrete values have already been visited across the tree.
        # Helps explain 'why didn't FTTS try param=X' - it might be because X
        # was already reached (in combination with other HPs) via another path.
        self._value_coverage: dict[str, set] = {}

    # ---------------------------------------------- public API

    def _hp_signature(self, hp: dict) -> str:
        """
        Produce a deterministic signature for a hyperparameter dict.
        Used to skip duplicates across different action paths (DAG-dedup).
        """
        import json
        # Filter to only the keys that affect training
        # (skip metadata-like keys with __ prefix if any)
        relevant = {k: v for k, v in hp.items() if not k.startswith("__")}
        return json.dumps(relevant, sort_keys=True, default=str)

    def _value_coverage_key(self, hp: dict, target_param: str | None) -> tuple | None:
        """
        Coverage key for value-aware diversity tracking.

        Different from _hp_signature: this looks at just `(target_param, value)`
        so we can tell whether a particular value of a tunable has already
        been explored (regardless of other HPs).

        Returns None if no target_param (action doesn't tune a specific value).
        """
        if not target_param or target_param not in hp:
            return None
        return (target_param, hp[target_param])

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
        self._seen_signatures.add(self._hp_signature(hp))

    def propose_next(self) -> tuple[str, dict, Any, str] | None:
        """
        Pop the next (parent_id, action) pair from the tree and produce the
        hyperparameters for the child.

        Skips duplicates: if applying the action produces a hyperparameter
        configuration that was already evaluated (or queued via another path),
        we transparently move on to the next action in the queue. This
        prevents wasteful re-exploration of the same node from different
        branches (DAG semantics).

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
            # Action could not be applied (e.g., param disabled, or already at
            # YAML range boundary). Skip and try next.
            print(f"[ftts] action '{action.type}' from {parent_id} produced "
                  f"no change (target may be at YAML boundary). Trying next.")
            return self.propose_next()

        # DAG-dedup: skip if this exact HP combination has been seen before.
        # This is correct DAG behavior - two paths converged to the same state.
        signature = self._hp_signature(child_hp)
        if signature in self._seen_signatures:
            tgt = action.target_param
            tgt_old = parent.hyperparameters.get(tgt) if tgt else None
            tgt_new = child_hp.get(tgt) if tgt else None

            # Stepped recovery: for discrete actions (increase_X / decrease_X
            # on a param with choices or an int range), don't just give up -
            # try stepping FURTHER in the same direction. e.g. if 128 -> 256
            # is taken, try 128 -> 512, then 128 -> 1024. This guarantees
            # large values stay reachable instead of being permanently
            # skipped because the intermediate step collided with another path.
            stepped = self._try_further_discrete_step(
                parent.hyperparameters, action, self._seen_signatures)
            if stepped is not None:
                child_hp, change_desc = stepped
                signature = self._hp_signature(child_hp)
                self._seen_signatures.add(signature)
                print(f"[ftts] dedup recovery: '{action.type}' from "
                      f"{parent_id} - {tgt}:{tgt_old}->{tgt_new} was already "
                      f"explored; stepped further to {change_desc}.")
                vk = self._value_coverage_key(child_hp, action.target_param)
                if vk is not None:
                    self._value_coverage.setdefault(vk[0], set()).add(vk[1])
                rationale = (
                    f"Based on {parent_id} (verdict={parent.verdict}, quality="
                    f"{parent.quality_score:.3f}). Applied action "
                    f"'{action.type}' [priority={action.priority:.2f}] with "
                    f"dedup-stepping: {change_desc}"
                )
                return parent_id, child_hp, action, rationale

            # No further step available - genuinely exhausted this direction.
            print(f"[ftts] dedup: skipping '{action.type}' from {parent_id} "
                  f"(target={tgt}: {tgt_old} -> {tgt_new}) - HP combination "
                  f"already explored and no further unexplored step exists.")
            return self.propose_next()
        self._seen_signatures.add(signature)

        # Track value-coverage for diagnostics: which values of each tunable
        # have already been visited. Useful for understanding 'why didn't FTTS
        # try seq_len=X?' - it might've been because X was already explored
        # in combination with all other reachable HP variants.
        vk = self._value_coverage_key(child_hp, action.target_param)
        if vk is not None:
            self._value_coverage.setdefault(vk[0], set()).add(vk[1])

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

        # ----- layer shape (one action per pattern) -----
        if t == "try_layer_shape_uniform":
            return self._set_choice(hp, "layer_shape", "uniform")
        if t == "try_layer_shape_funnel":
            return self._set_choice(hp, "layer_shape", "funnel")
        if t == "try_layer_shape_pyramid":
            return self._set_choice(hp, "layer_shape", "pyramid")
        if t == "try_layer_shape_hourglass":
            return self._set_choice(hp, "layer_shape", "hourglass")
        if t == "try_layer_shape_bottleneck":
            return self._set_choice(hp, "layer_shape", "bottleneck")

        # ----- activation -----
        if t == "change_activation":
            return self._cycle_choice(hp, "activation")

        # ----- optimizer -----
        if t == "change_optimizer":
            new_val = action.suggested_value or "adamw"
            return self._set_choice(hp, "optimizer_name", new_val)

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

        # ----- loss function: focal / cross_entropy switch -----
        if t == "try_focal_loss":
            return self._set_choice(hp, "loss_function", "focal")
        if t == "try_cross_entropy":
            return self._set_choice(hp, "loss_function", "cross_entropy")

        # ----- focal_gamma tuning (only relevant if focal is active) -----
        if t == "increase_focal_gamma":
            return self._adjust_additive(hp, "focal_gamma",
                                          delta_sign=+1, action_type=t)
        if t == "decrease_focal_gamma":
            return self._adjust_additive(hp, "focal_gamma",
                                          delta_sign=-1, action_type=t)

        # ----- normalization scheme switch -----
        if t == "change_normalization":
            return self._cycle_choice(hp, "normalization")

        # ----- text augmentation -----
        if t == "change_text_augmentation":
            return self._cycle_choice(hp, "text_augmentation")
        if t == "increase_text_augmentation":
            return self._adjust_additive(hp, "text_augmentation_prob",
                                          delta_sign=+1, action_type=t)

        # ----- gradient accumulation + mixed precision -----
        if t == "increase_grad_accumulation":
            current = int(hp.get("gradient_accumulation_steps", 1))
            return self._set_value(hp, "gradient_accumulation_steps",
                                    min(current * 2, 8))
        if t == "enable_mixed_precision":
            return self._set_value(hp, "mixed_precision", True)

        # ----- embedding dropout (NLP) -----
        if t == "increase_embedding_dropout":
            return self._adjust_additive(hp, "embedding_dropout",
                                          delta_sign=+1, action_type=t)

        # ----- advanced image augmentation -----
        if t == "enable_cutout":
            return self._set_value(hp, "cutout", 0.25)
        if t == "enable_cutmix":
            return self._set_value(hp, "cutmix", 1.0)

        # ----- stochastic depth (deep transformer regularization) -----
        if t == "increase_stochastic_depth":
            return self._adjust_additive(hp, "stochastic_depth",
                                          delta_sign=+1, action_type=t)

        # ----- adam betas (advanced) -----
        if t == "adjust_adam_beta1":
            return self._adjust_additive(hp, "adam_beta1",
                                          delta_sign=-1, action_type=t)
        if t == "adjust_adam_beta2":
            return self._adjust_additive(hp, "adam_beta2",
                                          delta_sign=-1, action_type=t)

        # ----- language modeling (NLP-specific) -----
        if t == "unfreeze_embeddings":
            return self._set_value(hp, "freeze_embeddings", False)
        if t == "change_fusion_method":
            return self._cycle_choice(hp, "fusion_method")
        if t == "increase_sequence_length":
            return self._adjust_choice_param(hp, ["sequence_length"], direction=+1)
        if t == "decrease_sequence_length":
            return self._adjust_choice_param(hp, ["sequence_length"], direction=-1)
        if t == "increase_teacher_forcing":
            return self._adjust_choice_param(hp, ["teacher_forcing_ratio"], direction=+1)
        if t == "decrease_teacher_forcing":
            return self._adjust_choice_param(hp, ["teacher_forcing_ratio"], direction=-1)
        if t == "disable_bidirectional":
            return self._set_value(hp, "bidirectional", False)

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

    def _try_further_discrete_step(self, parent_hp: dict, action: Action,
                                   seen: set) -> tuple[dict, str] | None:
        """
        DAG-dedup recovery for stepped (discrete) actions.

        When a normal one-step move (e.g. sequence_length 128 -> 256) lands on
        an already-explored HP signature, this tries progressively larger
        steps in the SAME direction along the parameter's choices/range:
            128 -> 256 (blocked) -> 128 -> 512 -> 128 -> 1024 ...
        It returns the first (child_hp, change_description) whose signature is
        unseen, or None if every further step is also seen or out of range.

        This guarantees that high values like sequence_length=512/1024 are
        actually reachable, instead of being permanently skipped because the
        single intermediate step happened to collide with another DAG path.
        """
        spec = _DISCRETE_STEP_ACTIONS.get(action.type)
        if spec is None:
            return None
        candidate_names, direction = spec

        # Find the FIRST candidate param that is actually tunable on this arch
        # and present in the parent hp. (e.g. for add_width on an LSTM, the
        # relevant param is hidden_size; d_model/fc_size won't exist.)
        for name in candidate_names:
            p = self.cm.get_param(name)
            if p is None:
                continue
            current = parent_hp.get(name)
            if current is None:
                continue

            # Build the ordered list of values we can step to.
            if p.get("choices"):
                ordered = sorted(p["choices"])
                try:
                    idx = ordered.index(current)
                except ValueError:
                    # current not in choices - snap to nearest by value
                    idx = min(range(len(ordered)),
                              key=lambda i: abs(ordered[i] - current))
                # Candidate target indices: every step further in `direction`.
                step = 1
                while True:
                    next_idx = idx + direction * step
                    if next_idx < 0 or next_idx >= len(ordered):
                        break  # hit the boundary of the choices list
                    candidate_hp = dict(parent_hp)
                    candidate_hp[name] = ordered[next_idx]
                    sig = self._hp_signature(candidate_hp)
                    if sig not in seen:
                        return candidate_hp, (f"{name}: {current} -> "
                                               f"{ordered[next_idx]} "
                                               f"(stepped past dedup)")
                    step += 1
                return None  # every further choice was also seen

            elif p.get("range") is not None:
                # Integer-range param (depth). Step by +/-1, +/-2, ...
                lo, hi = p["range"]
                step = 1
                while True:
                    next_val = int(current) + direction * step
                    if next_val < int(lo) or next_val > int(hi):
                        break
                    candidate_hp = dict(parent_hp)
                    candidate_hp[name] = next_val
                    sig = self._hp_signature(candidate_hp)
                    if sig not in seen:
                        return candidate_hp, (f"{name}: {current} -> "
                                               f"{next_val} (stepped past dedup)")
                    step += 1
                return None
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
