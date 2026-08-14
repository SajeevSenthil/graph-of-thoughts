"""
Phase 2 -- Evidence graph constructor + reduction filter + hard gate.

INPUT:  a scenario's anomaly.json (raw host log, JSONL, one event per line;
        can be ~1M-2M lines / hundreds of MB -- always streamed, never
        loaded whole into memory).
OUTPUT: data/evidence_graph/{scenario}.json -- a small candidate graph
        (nodes = processes; edges = spawned / connected_to / wrote), plus
        reduction statistics and a hard-gate pass/fail report against
        data/ground_truth/{scenario}.json.

Construction and reduction happen in a single streaming pass: a node is
only ever created for a Process/Start event, and connected_to/wrote edges
are only ever created when the structural keep-rule already matches (see
_is_external_ip / _is_suspicious_path). Every other event type
(FileIO/Read, Image/Load, internal-only network traffic, ...) is read and
discarded without ever being inspected for content -- so "construct" and
"reduce" are the same pass here, not two separate stages that would need
reconciling.

CRITICAL: this file must NEVER read or forward the raw log's own
`is_warn` field. Spot-checking WS12's anomaly.json shows `is_warn` is
already "True" on the malicious cmd.exe/certutil.exe processes and
"False" on benign ones -- it is the dataset's own ground-truth label
sitting inside the evidence. If it ever leaked into a node's attributes
or the serialized text, the LLM would be handed the answer directly and
the entire experiment would be invalid. It is deliberately absent from
every dict this module builds.
"""

import argparse
import html
import ipaddress
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "ground_truth"
OUTPUT_DIR = REPO_ROOT / "data" / "evidence_graph"

# Purely structural rules -- no content/malice judgment anywhere in this file.
SUSPICIOUS_EXTENSIONS = {".exe", ".dll", ".bat", ".ps1", ".vbs", ".hta", ".py", ".sh", ".jar", ".jsp"}
# Linux binaries are frequently extension-less (e.g. this dataset's own
# "sandcat-linux" dropper), so extension alone isn't enough there -- also
# treat a write to a classic attacker staging directory as suspicious,
# regardless of extension.
LINUX_STAGING_DIRS = ("/tmp/", "/var/tmp/", "/dev/shm/")

# scenario -> (path to its anomaly.json, OS/schema family). APT29/Sidewinder/
# FIN6 share ONE Windows 10 host log (three attacks run against the same
# captured machine at different times) -- confirmed by inspecting the W10
# zip, which contains a single anomaly.json, not three. Ubuntu uses a
# completely different schema (sysdig/Falco-style syscall trace, not
# Windows ETW) -- confirmed by inspecting its raw events directly; see
# _construct_and_reduce_linux.
SIMULATED_DATA = Path.home() / "sajeev" / "Simulated-Data"
SCENARIO_LOGS = {
    "WS12": (SIMULATED_DATA / "SimulatedWS12" / "anomaly.json", "windows"),
    "Ubuntu": (SIMULATED_DATA / "realAPTlinux" / "hw17" / "anomaly.json", "linux"),
    "APT29": (SIMULATED_DATA / "realAPTWin10" / "win10" / "anomaly.json", "windows"),
    "Sidewinder": (SIMULATED_DATA / "realAPTWin10" / "win10" / "anomaly.json", "windows"),
    "FIN6": (SIMULATED_DATA / "realAPTWin10" / "win10" / "anomaly.json", "windows"),
}


def _is_external_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def _is_suspicious_path(path: str, os_family: str) -> bool:
    ext_hit = Path(path.replace("\\", "/")).suffix.lower() in SUSPICIOUS_EXTENSIONS
    if os_family == "linux":
        return ext_hit or path.startswith(LINUX_STAGING_DIRS)
    return ext_hit


def _stream_events(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _new_stats():
    return {
        "total_events": 0,
        "process_start_seen": 0,
        "tcpip_seen": 0,
        "tcpip_kept_external": 0,
        "fileio_write_seen": 0,
        "fileio_write_kept_suspicious": 0,
    }


def _construct_windows(anomaly_json_path: Path):
    """
    Windows ETW-style schema (WS12, APT29, Sidewinder, FIN6): every event
    carries a top-level PID, and Process/Start carries ParentID + ImageFileName
    + CommandLine directly.

    Nodes are keyed by (PID, MSec) -- the start time makes each Process/Start
    event's key unique even when its PID is reused later (confirmed on WS12:
    3,139 Process/Start events but only 593 distinct PIDs, i.e. the average
    PID was reused >5 times -- and the raw log's own `UniqueProcessKey`
    turned out to be a recycled kernel-object address, NOT actually unique
    either: one value repeated 802 times). Keying by PID alone would let a
    later process silently overwrite an earlier one in the node dict -- if
    that earlier process happened to be part of the attack, it would vanish
    without any error. `pid_to_current_node` tracks, at each point while
    streaming forward through the (chronologically ordered) log, which
    process instance currently "owns" a given PID, so that TcpIp/FileIO
    events and parent-child linkage always attach to the correct instance.
    """
    nodes = {}  # unique_key -> node dict
    pid_to_current_node = {}  # pid -> unique_key of whichever process currently owns it
    # (owner_key, target) -> first-seen msec. A long-lived connection or a file
    # written in many small chunks fires one raw TcpIp/FileIO event PER
    # PACKET/CHUNK -- deduplicating to one edge per (process, target) keeps
    # the candidate set to "this process touched this endpoint", which is all
    # the evidence actually needs, without exploding edge count on repetition.
    connected_to = {}
    wrote = {}
    stats = _new_stats()

    for event in _stream_events(anomaly_json_path):
        stats["total_events"] += 1
        name = event.get("EventName")
        pid = event.get("PID")

        if name == "Process/Start":
            stats["process_start_seen"] += 1
            unique_key = f"{pid}_{event.get('MSec')}"
            parent_pid = event.get("ParentID")
            nodes[unique_key] = {
                "pid": pid,
                "parent_id": parent_pid,
                "parent_key": pid_to_current_node.get(parent_pid),
                "image": event.get("ImageFileName"),
                "command_line": html.unescape(event.get("CommandLine", "")),
                "msec": event.get("MSec"),
            }
            pid_to_current_node[pid] = unique_key
        elif name in ("TcpIp/Send", "TcpIp/Recv"):
            stats["tcpip_seen"] += 1
            daddr = event.get("daddr")
            owner_key = pid_to_current_node.get(pid)
            if daddr and owner_key and _is_external_ip(daddr):
                stats["tcpip_kept_external"] += 1
                dport = str(event.get("dport", "")).replace(",", "")
                key = (owner_key, f"{daddr}:{dport}")
                if key not in connected_to:
                    connected_to[key] = event.get("MSec")
        elif name == "FileIO/Write":
            stats["fileio_write_seen"] += 1
            fname = event.get("FileName", "")
            owner_key = pid_to_current_node.get(pid)
            if fname and owner_key and _is_suspicious_path(fname, "windows"):
                stats["fileio_write_kept_suspicious"] += 1
                key = (owner_key, fname)
                if key not in wrote:
                    wrote[key] = event.get("MSec")
        # FileIO/Read, Image/Load, Process/Stop, internal-only network traffic,
        # and every other event type: read, never inspected further, dropped.

    return nodes, connected_to, wrote, stats


_LINUX_PID_RE = re.compile(r"\bpid=(\d+)\(")
_LINUX_PTID_RE = re.compile(r"\bptid=(\d+)\(")
_LINUX_REMOTE_ADDR_RE = re.compile(r"->([\d.]+):(\d+)\)?$")
_LINUX_NETWORK_EVENTS = {"sendto", "recvfrom", "sendmsg", "recvmsg", "accept", "connect"}


def _construct_linux(anomaly_json_path: Path):
    """
    sysdig/Falco-style syscall trace (Ubuntu) -- a structurally different
    schema from the Windows ETW logs, confirmed by inspecting the raw events
    directly: no top-level PID field exists on most event types (only
    `execve`/`clone` embed pid=/ptid= inside a free-text `evt.args` string;
    network and write events carry only `proc.cmdline`, no numeric PID at
    all). So process ownership here is resolved by matching `proc.cmdline`
    text to the most recent `execve` that produced that exact command line,
    not by numeric PID lookups as on Windows.

    `execve` (not `clone`) is treated as this dataset's Process/Start
    equivalent: `clone` is the fork() that creates a new PID still running
    the parent's program image, while `execve` is the moment that PID's
    image (and command line) becomes the thing we actually care about.
    """
    nodes = {}
    cmdline_to_current_node = {}  # proc.cmdline -> unique_key of the execve that produced it
    connected_to = {}
    wrote = {}
    stats = _new_stats()

    for event in _stream_events(anomaly_json_path):
        stats["total_events"] += 1
        etype = event.get("evt.type")
        cmdline = event.get("proc.cmdline") or ""

        if etype == "execve":
            stats["process_start_seen"] += 1
            args = event.get("evt.args") or ""
            m_pid = _LINUX_PID_RE.search(args)
            if not m_pid:
                continue  # can't identify this process instance -- skip rather than guess
            pid = m_pid.group(1)
            m_ptid = _LINUX_PTID_RE.search(args)
            parent_cmdline = event.get("proc.pcmdline") or ""
            unique_key = f"{pid}_{event.get('evt.time')}"
            nodes[unique_key] = {
                "pid": pid,
                "parent_id": m_ptid.group(1) if m_ptid else None,
                "parent_key": cmdline_to_current_node.get(parent_cmdline),
                "image": event.get("proc.name"),
                "command_line": cmdline,
                "msec": event.get("evt.time"),
            }
            cmdline_to_current_node[cmdline] = unique_key
        elif etype in _LINUX_NETWORK_EVENTS:
            stats["tcpip_seen"] += 1
            fd_name = event.get("fd.name") or ""
            m = _LINUX_REMOTE_ADDR_RE.search(fd_name)
            owner_key = cmdline_to_current_node.get(cmdline)
            if m and owner_key and _is_external_ip(m.group(1)):
                stats["tcpip_kept_external"] += 1
                key = (owner_key, f"{m.group(1)}:{m.group(2)}")
                if key not in connected_to:
                    connected_to[key] = event.get("evt.time")
        elif etype in ("write", "writev"):
            stats["fileio_write_seen"] += 1
            fd_name = event.get("fd.name") or ""
            owner_key = cmdline_to_current_node.get(cmdline)
            # fd.name is only a real filesystem path for actual file writes --
            # sysdig reuses the same "write" syscall event for socket/pipe fds
            # too, where fd.name holds a "local->remote" tuple or "pipe:[N]".
            is_real_path = fd_name.startswith("/") and "->" not in fd_name
            if is_real_path and owner_key and _is_suspicious_path(fd_name, "linux"):
                stats["fileio_write_kept_suspicious"] += 1
                key = (owner_key, fd_name)
                if key not in wrote:
                    wrote[key] = event.get("evt.time")
        # everything else (read, fstat, open, close, ...): read, dropped.

    return nodes, connected_to, wrote, stats


def construct_and_reduce(anomaly_json_path: Path, os_family: str):
    """Dispatch to the right schema adapter, then apply the shared reduction."""
    if os_family == "linux":
        nodes, connected_to, wrote, stats = _construct_linux(anomaly_json_path)
    else:
        nodes, connected_to, wrote, stats = _construct_windows(anomaly_json_path)
    return _finalize(nodes, connected_to, wrote, stats)


_VOLATILE_PARAM_RE = re.compile(r"\d{3,}")


def _normalize_for_grouping(command_line: str) -> str:
    """
    Grouping key ONLY -- never used for hard-gate matching or for what
    actually gets displayed/stored. Collapses runs of 3+ digits (PIDs,
    handles, GUID fragments, millisecond timestamps) to a placeholder so
    near-identical repeated launches of the same program group together even
    when a volatile numeric parameter differs each time. Confirmed necessary
    on WS12: Chrome alone spawns dozens of renderer/gpu/utility subprocesses
    whose command lines are identical except for a --renderer-client-id or
    --launch-time-ticks value, so exact-text grouping alone left hundreds of
    near-duplicates uncollapsed. Checked against all 89 ground-truth
    match_substrings before adopting this -- none depend on a 3+ digit run
    being preserved verbatim for a DIFFERENT command to stay distinguishable
    within the same scenario.
    """
    return _VOLATILE_PARAM_RE.sub("#", command_line)


def _finalize(nodes, connected_to, wrote, stats):
    stats["raw_process_nodes"] = len(nodes)

    # Collapse repeats: the *exact same program run with the exact same
    # command line* (or the same command differing only in a volatile numeric
    # parameter -- see _normalize_for_grouping) re-launched many times (a
    # helper/service respawning) adds no new investigative fact beyond "this
    # happened N times" -- confirmed on WS12, where the correctly
    # PID-reuse-safe count above (3,139 distinct process-start events) turned
    # out to be almost entirely one handful of commands repeating hundreds of
    # times each. Collapsing to one representative node per group, keeping a
    # repeat_count and the earliest occurrence, is still purely structural
    # (grouping by literal content pattern, not by a malice judgment) and is
    # what actually gets the candidate set down to something serializable.
    groups = {}
    for key, node in nodes.items():
        group_key = (node["image"], _normalize_for_grouping(node["command_line"]))
        groups.setdefault(group_key, []).append(key)

    canonical = {}  # raw key -> representative key
    collapsed_nodes = {}
    for (_image, _cmd), keys in groups.items():
        keys.sort(key=lambda k: float(nodes[k]["msec"]))
        rep_key = keys[0]
        rep_node = dict(nodes[rep_key])
        rep_node["repeat_count"] = len(keys)
        collapsed_nodes[rep_key] = rep_node
        for k in keys:
            canonical[k] = rep_key

    for node in collapsed_nodes.values():
        if node["parent_key"]:
            node["parent_key"] = canonical.get(node["parent_key"])

    # Re-key and re-deduplicate connected_to/wrote through the canonical
    # mapping -- two repeats of the same command may each have independently
    # connected to the same external target, which is now a true duplicate
    # once both repeats collapse onto one representative node.
    def _remap_and_dedupe(edge_dict):
        remapped = {}
        for (owner_key, target), msec in edge_dict.items():
            new_key = (canonical.get(owner_key, owner_key), target)
            if new_key not in remapped:
                remapped[new_key] = msec
        return remapped

    connected_to = _remap_and_dedupe(connected_to)
    wrote = _remap_and_dedupe(wrote)

    edges = [
        {"type": "connected_to", "from_node": owner_key, "target": target, "msec": msec}
        for (owner_key, target), msec in connected_to.items()
    ] + [
        {"type": "wrote", "from_node": owner_key, "target": target, "msec": msec}
        for (owner_key, target), msec in wrote.items()
    ]

    for key, node in collapsed_nodes.items():
        parent_key = node["parent_key"]
        if parent_key and parent_key in collapsed_nodes:
            edges.append({"type": "spawned", "from_node": parent_key, "to_node": key, "msec": node["msec"]})

    stats["candidate_nodes"] = len(collapsed_nodes)
    stats["candidate_edges"] = len(edges)
    stats["pid_reuse_collisions"] = stats["process_start_seen"] - stats["raw_process_nodes"]
    stats["repeat_collapsed"] = stats["raw_process_nodes"] - len(collapsed_nodes)
    stats["raw_connected_to_events"] = stats["tcpip_kept_external"]
    stats["deduped_connected_to_edges"] = len(connected_to)
    stats["raw_wrote_events"] = stats["fileio_write_kept_suspicious"]
    stats["deduped_wrote_edges"] = len(wrote)
    return collapsed_nodes, edges, stats


# Known, explained hard-gate exceptions -- a node the reduction filter can
# never make survive, for a documented structural reason, not a bug. Kept
# separate from the parser's own match_substring=None ("unscoreable") case,
# which is a different failure mode (non-distinctive command vs. evidence
# that isn't in this log at all).
EXPECTED_GATE_EXCEPTIONS = {
    "Ubuntu": {
        "A8": (
            "hostIp for this node is 192.168.0.244, not 192.168.0.155 like every "
            "other Ubuntu node -- it's evidence from a SECOND host reached via the "
            "A6/A7 lateral-movement step (running JuicyPotato_x64.exe, a Windows "
            "privilege-escalation tool). We only have a log for the originating "
            "Linux host; confirmed via `grep -c JuicyPotato` on the raw log = 0."
        ),
    },
}


def hard_gate(nodes: dict, ground_truth_nodes: list[dict], scenario: str | None = None) -> dict:
    """Check every ground-truth match_substring against the reduced node set."""
    exceptions = EXPECTED_GATE_EXCEPTIONS.get(scenario, {})
    all_commands = [n["command_line"] for n in nodes.values()]
    results = []
    for gt in ground_truth_nodes:
        ms = gt["match_substring"]
        if ms is None:
            results.append({"id": gt["id"], "match_substring": None, "found": None, "expected_exception": None})
            continue
        found = any(ms in cmd for cmd in all_commands)
        results.append(
            {"id": gt["id"], "match_substring": ms, "found": found, "expected_exception": exceptions.get(gt["id"])}
        )

    checkable = [r for r in results if r["found"] is not None]
    genuinely_missing = [r for r in checkable if not r["found"] and not r["expected_exception"]]
    explained_missing = [r for r in checkable if not r["found"] and r["expected_exception"]]
    return {
        "passed": len(genuinely_missing) == 0,
        "checked": len(checkable),
        "survived": sum(1 for r in checkable if r["found"]),
        "missing": [r["id"] for r in genuinely_missing],
        "explained_missing": {r["id"]: r["expected_exception"] for r in explained_missing},
        "unscoreable": [r["id"] for r in results if r["found"] is None],
        "results": results,
    }


def run_scenario(scenario: str, graph_cache: dict) -> dict:
    log_path, os_family = SCENARIO_LOGS[scenario]
    ground_truth = json.loads((GROUND_TRUTH_DIR / f"{scenario}.json").read_text())

    # APT29/Sidewinder/FIN6 share one W10 log -- construct it once, reuse for
    # all three scenarios' hard-gate checks instead of re-streaming ~650MB
    # three times for identical output.
    if log_path not in graph_cache:
        graph_cache[log_path] = construct_and_reduce(log_path, os_family)
    nodes, edges, stats = graph_cache[log_path]

    gate = hard_gate(nodes, ground_truth, scenario)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{scenario}.json"
    out_path.write_text(
        json.dumps({"scenario": scenario, "nodes": nodes, "edges": edges, "stats": stats, "hard_gate": gate}, indent=2),
        encoding="utf-8",
    )
    return {"scenario": scenario, "stats": stats, "gate": gate, "out_path": out_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIO_LOGS.keys()))
    args = parser.parse_args()

    graph_cache = {}
    seen_logs = set()
    for scenario in args.scenarios:
        log_path, _os_family = SCENARIO_LOGS[scenario]
        cache_note = " (shared W10 log)" if log_path in seen_logs else ""
        seen_logs.add(log_path)

        result = run_scenario(scenario, graph_cache)
        s, g = result["stats"], result["gate"]
        gate_str = "PASS" if g["passed"] else f"FAIL -- missing: {g['missing']}"
        if g["explained_missing"]:
            gate_str += f"  [explained exception: {list(g['explained_missing'].keys())}]"
        print(
            f"{scenario:12s}{cache_note:16s} {s['total_events']:>9,} events -> "
            f"{s['raw_process_nodes']:>4d} process starts "
            f"({s['pid_reuse_collisions']} PID-collisions avoided, {s['repeat_collapsed']} repeats collapsed) -> "
            f"{s['candidate_nodes']:>4d} nodes / {s['candidate_edges']:>4d} edges   "
            f"hard gate: {gate_str}"
            + (f"  [{len(g['unscoreable'])} unscoreable: {g['unscoreable']}]" if g["unscoreable"] else "")
        )


if __name__ == "__main__":
    main()
