"""
ExperimentTree
--------------
The core data structure behind FTTS (Fine-Tuning Tree Search).

Each trial is a node in the tree. Edges represent "this child was derived from
this parent by applying this action". The priority queue picks the next trial
to run by finding the best-scoring pending action across all nodes.

Key properties:
  * thread-safe: a single Lock protects all mutations.
  * priority-ordered: a heap keeps pending actions sorted by child_score.
  * inspectable: the full tree is serializable for reports and resume.

Node life-cycle:
  1. Created with a set of pending actions (from the Analyzer).
  2. Each pending action generates a queue entry (child_score=parent_quality*priority).
  3. When an entry is popped, one pending action is consumed and the resulting
     trial becomes a new child node.
  4. When all pending actions have been consumed, the node is "exhausted".
     It stays in the tree for reporting, but no more queue entries come from it.
"""

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A single experiment in the tree."""
    trial_id: str
    parent_id: str | None                 # None = root
    hyperparameters: dict
    quality_score: float = 0.0            # set after the trial runs
    verdict: str = "pending"              # from Analyzer
    pending_actions: list = field(default_factory=list)  # list of Action
    consumed_actions: list = field(default_factory=list) # Actions already spawned children
    children_ids: list[str] = field(default_factory=list)
    status: str = "pending"               # pending / running / done / failed / diverged
    raw_best_metric: float = 0.0
    smoothed_best_metric: float = 0.0
    rationale: str = ""                   # why this trial's parameters were chosen


@dataclass(order=True)
class QueueEntry:
    """An entry in the priority queue. Sorted by neg_score so heapq pops highest first."""
    neg_score: float                      # negative of (parent_quality * action_priority)
    counter: int                          # tie-breaker so heap is stable
    parent_id: str = field(compare=False)
    action: Any = field(compare=False)    # Action object


class ExperimentTree:
    """
    Thread-safe tree of experiment nodes with a priority queue of pending actions.

    Usage pattern (from Orchestrator):
        tree = ExperimentTree()
        tree.add_root_placeholder()

        while not should_stop():
            # Get next proposal
            parent_id, action = tree.pop_best_pending()
            trial_id = generate_id()
            hp = apply_action(parent_hp, action)
            result = train(...)

            # Record
            diagnosis = analyze(result)
            tree.register_completed(trial_id, parent_id, result, diagnosis, action)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._nodes: dict[str, Node] = {}
        self._heap: list[QueueEntry] = []
        self._counter = 0              # tie-breaker for heap stability

    # ---------------------------------------------------------------- utilities

    @property
    def nodes(self) -> dict[str, Node]:
        """Read-only view of all nodes. Caller should not mutate."""
        return self._nodes

    def get_node(self, trial_id: str) -> Node | None:
        with self._lock:
            return self._nodes.get(trial_id)

    def size(self) -> int:
        with self._lock:
            return len(self._nodes)

    def queue_size(self) -> int:
        """Number of pending actions across all nodes."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """True if no more pending actions."""
        with self._lock:
            return len(self._heap) == 0

    # --------------------------------------------------------- root management

    def register_root(self, trial_id: str, hyperparameters: dict,
                      rationale: str = "Root trial: starting point from config.") -> None:
        """Register the first trial (root of the tree)."""
        with self._lock:
            if self._nodes:
                raise RuntimeError("Root already exists.")
            node = Node(
                trial_id=trial_id,
                parent_id=None,
                hyperparameters=hyperparameters,
                status="running",
                rationale=rationale,
            )
            self._nodes[trial_id] = node

    # ----------------------------------------------------- completing a trial

    def register_completed(self, trial_id: str, parent_id: str | None,
                           hyperparameters: dict, quality_score: float,
                           verdict: str, actions: list,
                           raw_best: float, smoothed_best: float,
                           status: str, rationale: str,
                           applied_action=None) -> None:
        """
        Register a trial that has just finished, with its diagnosis and
        the prioritized actions for potential children.

        Args:
            trial_id: this trial's id.
            parent_id: the parent node's id (None for root).
            hyperparameters: what was used for training.
            quality_score: QualityBreakdown.total from quality_scorer.
            verdict: diagnosis verdict.
            actions: list of Action from Analyzer (prioritized).
            raw_best / smoothed_best: metric values for the report.
            status: "done" / "failed" / "diverged".
            rationale: human-readable explanation of why this trial was run.
            applied_action: the Action that created this trial (from parent).
                            None for root.
        """
        with self._lock:
            # Create or update node
            if trial_id in self._nodes:
                node = self._nodes[trial_id]
                node.hyperparameters = hyperparameters
            else:
                node = Node(
                    trial_id=trial_id,
                    parent_id=parent_id,
                    hyperparameters=hyperparameters,
                    rationale=rationale,
                )
                self._nodes[trial_id] = node

            node.quality_score = quality_score
            node.verdict = verdict
            node.raw_best_metric = raw_best
            node.smoothed_best_metric = smoothed_best
            node.status = status
            node.pending_actions = list(actions)
            node.consumed_actions = []

            # Mark parent-child linkage
            if parent_id is not None and parent_id in self._nodes:
                parent = self._nodes[parent_id]
                if trial_id not in parent.children_ids:
                    parent.children_ids.append(trial_id)
                if applied_action is not None:
                    parent.consumed_actions.append(applied_action)

            # Add this node's actions to the priority queue
            # Only if the trial was successful enough to be a useful parent
            if status in ("done",) and actions:
                for action in actions:
                    score = quality_score * action.priority
                    self._counter += 1
                    entry = QueueEntry(
                        neg_score=-score,
                        counter=self._counter,
                        parent_id=trial_id,
                        action=action,
                    )
                    heapq.heappush(self._heap, entry)

    # -------------------------------------------------- popping next proposal

    def pop_best_pending(self) -> tuple[str, Any] | None:
        """
        Pop the best (parent_id, action) pair from the priority queue.

        Returns None if the queue is empty.

        The popped action is NOT consumed automatically — the caller must call
        register_completed for the resulting trial, which ties the action to
        the parent's consumed_actions list.
        """
        with self._lock:
            while self._heap:
                entry = heapq.heappop(self._heap)
                parent = self._nodes.get(entry.parent_id)
                if parent is None:
                    continue  # orphaned, skip
                # Check action is still pending (not consumed earlier)
                if entry.action in parent.pending_actions:
                    parent.pending_actions.remove(entry.action)
                    return entry.parent_id, entry.action
            return None

    # ---------------------------------------------------- resume-level access

    def best_node(self) -> Node | None:
        """The node with the highest quality_score. None if tree empty."""
        with self._lock:
            done = [n for n in self._nodes.values() if n.status == "done"]
            if not done:
                return None
            return max(done, key=lambda n: n.quality_score)

    def to_summary_dict(self) -> dict:
        """Summary for reports: node counts by status, depth, etc."""
        with self._lock:
            by_status: dict[str, int] = {}
            by_verdict: dict[str, int] = {}
            for n in self._nodes.values():
                by_status[n.status] = by_status.get(n.status, 0) + 1
                by_verdict[n.verdict] = by_verdict.get(n.verdict, 0) + 1
            return {
                "total_nodes": len(self._nodes),
                "pending_in_queue": len(self._heap),
                "by_status": by_status,
                "by_verdict": by_verdict,
            }

    def lineage(self, trial_id: str) -> list[str]:
        """Return the chain of ancestors for a trial, from root to it (inclusive)."""
        with self._lock:
            chain = []
            cur = self._nodes.get(trial_id)
            while cur is not None:
                chain.append(cur.trial_id)
                if cur.parent_id is None:
                    break
                cur = self._nodes.get(cur.parent_id)
            return list(reversed(chain))
