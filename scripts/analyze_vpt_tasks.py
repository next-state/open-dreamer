"""Analyze Dreamer-4 task annotations available in the Minecraft VPT arrayrecords.

Read-only analysis: scans a sample of shards, reads the per-frame `stats` counters
in each episode (without decoding the mp4 video), maps them to all 20 Dreamer-4 tasks,
and reports per-task coverage, a per-task matched-item audit (to spot rules that are too
loose / too restrictive), and dataset heterogeneity.

All 20 tasks are derivable from `stats` counters. The 8 craft_* tasks come from the
`minecraft.craft_item:<item>` namespace (an earlier note that there was "no crafted
counter" was wrong). Smelting iron is counted under craft_item:iron_ingot too.

Usage:
    .venv/bin/python scripts/analyze_vpt_tasks.py --shards 30 --episodes-per-shard 100
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import statistics
from collections import Counter, defaultdict

from array_record.python.array_record_data_source import ArrayRecordDataSource

DATA_GLOB = "/mnt/data/minecraft-vpt/arrayrecords-mp4/shard-*.array_record"

# All 20 Dreamer-4 tasks (Table 4). Each maps to a predicate over (namespace_short,
# item), where namespace_short is the last dotted component of the stat key namespace
# (mine_block / craft_item / custom / use_item) and item is the suffix after ":".
# Kept in sync with data_vis/visualize_vpt_rerun.py::TASK_SIGNALS.
TASK_SIGNALS = {
    # mine_* : minecraft.mine_block:minecraft.<block>
    "mine_log": lambda ns, it: ns == "mine_block" and it.endswith("_log"),
    "mine_cobblestone": lambda ns, it: ns == "mine_block" and it in ("stone", "cobblestone"),
    "mine_iron_ore": lambda ns, it: ns == "mine_block" and "iron_ore" in it,
    "mine_coal": lambda ns, it: ns == "mine_block" and "coal_ore" in it,
    "mine_diamond": lambda ns, it: ns == "mine_block" and "diamond_ore" in it,
    # craft_* : minecraft.craft_item:minecraft.<item>  (smelting iron counts here too)
    "craft_planks": lambda ns, it: ns == "craft_item" and it.endswith("_planks"),
    "craft_stick": lambda ns, it: ns == "craft_item" and it == "stick",
    "craft_crafting_table": lambda ns, it: ns == "craft_item" and it == "crafting_table",
    "craft_furnace": lambda ns, it: ns == "craft_item" and it == "furnace",
    "craft_iron_ingot": lambda ns, it: ns == "craft_item" and it == "iron_ingot",
    "craft_wooden_pickaxe": lambda ns, it: ns == "craft_item" and it == "wooden_pickaxe",
    "craft_stone_pickaxe": lambda ns, it: ns == "craft_item" and it == "stone_pickaxe",
    "craft_iron_pickaxe": lambda ns, it: ns == "craft_item" and it == "iron_pickaxe",
    # open_* : minecraft.custom:minecraft.interact_with_*
    "open_crafting_table": lambda ns, it: ns == "custom" and it == "interact_with_crafting_table",
    "open_furnace": lambda ns, it: ns == "custom" and it == "interact_with_furnace",
    # place_* : minecraft.use_item:minecraft.<placeable>
    "place_crafting_table": lambda ns, it: ns == "use_item" and it == "crafting_table",
    "place_furnace": lambda ns, it: ns == "use_item" and it == "furnace",
    # use_*_pickaxe : minecraft.use_item:minecraft.<pickaxe>
    "use_wooden_pickaxe": lambda ns, it: ns == "use_item" and it == "wooden_pickaxe",
    "use_stone_pickaxe": lambda ns, it: ns == "use_item" and it == "stone_pickaxe",
    "use_iron_pickaxe": lambda ns, it: ns == "use_item" and it == "iron_pickaxe",
}

ORE_TASKS = ("mine_iron_ore", "mine_coal", "mine_diamond")


def parse_stat_key(key: str) -> tuple[str, str]:
    """'minecraft.mine_block:minecraft.oak_log' -> ('mine_block', 'oak_log')."""
    ns, _, item = key.partition(":")
    return ns.split(".")[-1], item.replace("minecraft.", "")


def episode_events(action_dicts: list[dict]) -> dict:
    """Return per-task in-segment events and raw namespace tallies for one episode.

    VPT `stats` are cumulative over the whole play *session* and are NOT reset per
    5-minute segment, so the values present at the first stats-bearing frame are a
    carried-over BASELINE. A real in-segment event = a counter strictly exceeds its
    previously seen value, OR a counter newly appears AFTER the baseline frame. Baseline
    values are never counted (counting them inflates coverage).

    The baseline anchor is the first frame whose `stats` dict is non-empty, NOT frame 0:
    in some segments stats only start transmitting a few frames in, dumping the whole
    session baseline at once. Anchoring to frame 0 would flag that dump as phantom events.
    """
    t_base = next((t for t, a in enumerate(action_dicts) if a.get("stats")), 0)
    last_value: dict[str, int] = {}
    task_event_frames: dict[str, list[int]] = defaultdict(list)
    task_items: dict[str, Counter] = defaultdict(Counter)  # task -> concrete item -> count
    raw = {ns: Counter() for ns in ("mine_block", "craft_item", "use_item", "custom", "pickup")}

    for t, action in enumerate(action_dicts):
        for key, val in action.get("stats", {}).items():
            ns, item = parse_stat_key(key)
            prev = last_value.get(key)
            incremented = (prev is None and t > t_base) or (
                prev is not None and isinstance(val, (int, float)) and val > prev)
            last_value[key] = val
            if not incremented:
                continue
            if ns in raw:
                raw[ns][item] += 1
            for task, pred in TASK_SIGNALS.items():
                if pred(ns, item):
                    task_event_frames[task].append(t)
                    task_items[task][item] += 1
    return {"task_event_frames": dict(task_event_frames), "raw": raw,
            "task_items": {k: dict(v) for k, v in task_items.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=30, help="number of shards to sample (evenly spaced)")
    ap.add_argument("--episodes-per-shard", type=int, default=100)
    ap.add_argument("--out", type=str, default=None, help="optional path to write JSON report")
    args = ap.parse_args()

    all_shards = sorted(glob.glob(DATA_GLOB))
    if not all_shards:
        raise SystemExit(f"no shards found at {DATA_GLOB}")
    n = min(args.shards, len(all_shards))
    step = max(1, len(all_shards) // n)
    sampled = all_shards[::step][:n]
    print(f"Total shards available: {len(all_shards)}; sampling {len(sampled)} (every {step}th)\n")

    # Per-task aggregates
    task_episodes = Counter()          # episodes with >=1 occurrence
    task_total_events = Counter()      # total increments
    task_events_per_ep = defaultdict(list)
    task_item_total = defaultdict(Counter)  # task -> concrete item -> total count (rule audit)
    raw_total = {ns: Counter() for ns in ("mine_block", "craft_item", "use_item", "custom", "pickup")}
    shard_has_ore = Counter()          # shard -> n episodes with any ore mining

    n_episodes = 0
    n_relevant = 0                     # episodes hitting >=1 of the 20 tasks
    n_ore = 0                          # episodes with any ore-mining task

    for shard in sampled:
        ds = ArrayRecordDataSource([shard])
        k = min(args.episodes_per_shard, len(ds))
        shard_name = shard.split("/")[-1]
        for i in range(k):
            rec = pickle.loads(ds[i])
            n_episodes += 1
            res = episode_events(rec["actions"])
            tf = res["task_event_frames"]
            for ns, c in res["raw"].items():
                raw_total[ns].update(c)
            for task, items in res["task_items"].items():
                task_item_total[task].update(items)
            hit_any = False
            hit_ore = False
            for task, frames in tf.items():
                if frames:
                    task_episodes[task] += 1
                    task_total_events[task] += len(frames)
                    task_events_per_ep[task].append(len(frames))
                    hit_any = True
                    if task in ORE_TASKS:
                        hit_ore = True
            if hit_any:
                n_relevant += 1
            if hit_ore:
                n_ore += 1
                shard_has_ore[shard_name] += 1
        print(f"  scanned {shard_name}: {k} episodes (running total {n_episodes})")

    # ---- Report ----
    print("\n" + "=" * 78)
    print(f"SCANNED {n_episodes} episodes across {len(sampled)} shards")
    print("=" * 78)

    print("\nPER-TASK COVERAGE (all 20 tasks):")
    print(f"  {'task':22s} {'eps':>6s} {'%eps':>7s} {'events':>8s} {'med/ep':>7s}")
    for task in TASK_SIGNALS:
        eps = task_episodes[task]
        pct = 100.0 * eps / n_episodes if n_episodes else 0.0
        tot = task_total_events[task]
        med = statistics.median(task_events_per_ep[task]) if task_events_per_ep[task] else 0
        print(f"  {task:22s} {eps:6d} {pct:6.2f}% {tot:8d} {med:7.0f}")

    # Rule audit: the concrete stat items each rule matched. For exact-match rules this
    # is just the rule's literal; for substring/suffix rules (mine_log, mine_*_ore,
    # craft_planks) it reveals every variant collapsed into the task -> spot rules that
    # are too loose (catching the wrong item) or too restrictive (missing a variant).
    print("\nRULE AUDIT (distinct stat items matched per task; scan the multi-item rules):")
    for task in TASK_SIGNALS:
        items = task_item_total[task].most_common(12)
        shown = ", ".join(f"{it}({c})" for it, c in items) or "(no events in sample)"
        print(f"  {task:22s} {shown}")

    print(f"\nCANDIDATE 'RELEVANT' FRACTION (episodes hitting >=1 of the 20 tasks): "
          f"{n_relevant}/{n_episodes} = {100.0*n_relevant/max(1,n_episodes):.1f}%")
    print(f"ORE-MINING episodes (iron/coal/diamond): {n_ore}/{n_episodes} = "
          f"{100.0*n_ore/max(1,n_episodes):.2f}%")
    if shard_has_ore:
        print("  shards containing ore-mining episodes:")
        for s, c in shard_has_ore.most_common():
            print(f"    {s}: {c} episodes")
    else:
        print("  NO ore-mining episodes found in the sample (building-heavy subset).")

    print("\nRAW NAMESPACE TOP ITEMS (increment counts; sanity-check the task mapping):")
    for ns in ("mine_block", "craft_item", "use_item", "custom", "pickup"):
        print(f"  [{ns}] " + ", ".join(f"{it}:{c}" for it, c in raw_total[ns].most_common(15)))

    if args.out:
        report = {
            "n_episodes": n_episodes,
            "n_shards": len(sampled),
            "task_episodes": dict(task_episodes),
            "task_total_events": dict(task_total_events),
            "n_relevant": n_relevant,
            "n_ore": n_ore,
            "shard_has_ore": dict(shard_has_ore),
            "raw_top": {ns: dict(raw_total[ns].most_common(40)) for ns in raw_total},
            "task_items": {t: dict(task_item_total[t].most_common(40)) for t in TASK_SIGNALS},
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {args.out}")


if __name__ == "__main__":
    main()
