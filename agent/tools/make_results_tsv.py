"""Rebuild agent/results.tsv (autoresearch-style log) from saved run JSONs and commit subjects."""
import glob, json, os, subprocess
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
runs = []
for path in glob.glob(os.path.join(root, "agent", "results", "*.json")):
    if path.endswith(".logs.json") or "deployment" in path: continue
    run = json.load(open(path))
    if run.get("startedAt"): runs.append(run)
best, lines = 0.0, ["commit\tscore\tpublic0_tps\tpublic1_tps\tpublic2_tps\tstatus\tdescription"]
for run in sorted(runs, key=lambda r: r["startedAt"]):
    result = run.get("result") or {}; sha = (run.get("commitSha") or "")[:7]; score = result.get("score") or 0.0
    tps = [round(s.get("tokensPerSecond") or 0, 1) for s in result.get("shapes", [])] + [0, 0, 0]
    status = "crash" if run.get("state") != "succeeded" else ("keep" if score > best else "discard")
    best = max(best, score)
    subject = subprocess.run(["git", "-C", root, "log", "-1", "--format=%s", sha], capture_output=True, text=True).stdout.strip()
    lines.append(f"{sha}\t{score:.1f}\t{tps[0]}\t{tps[1]}\t{tps[2]}\t{status}\t{subject}")
open(os.path.join(root, "agent", "results.tsv"), "w").write("\n".join(lines) + "\n")
print(len(lines) - 1, "runs; best", round(best, 1))
