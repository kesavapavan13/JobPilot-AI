"""Evaluate the pipeline on a folder of job descriptions.

Usage:
    python evaluate.py --resume my_resume.txt --jds eval_data/jds --out eval_results.csv

Each *.txt file in --jds is one job description. The MySQL tracker is NOT used.
Prints the numbers for your resume bullet ([N] jobs, [X]% fabrication-free).
"""
import argparse
import csv
import statistics
from pathlib import Path

from agent import PipelineError, run_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="Plain-text resume file")
    p.add_argument("--jds", default="eval_data/jds", help="Folder of .txt job descriptions")
    p.add_argument("--out", default="eval_results.csv")
    args = p.parse_args()

    resume = Path(args.resume).read_text(encoding="utf-8")
    files = sorted(Path(args.jds).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {args.jds}")

    rows = []
    for f in files:
        print(f"Running {f.name} ...", flush=True)
        try:
            s = run_pipeline(f.read_text(encoding="utf-8"), resume, persist=False)
            rows.append({
                "file": f.name, "role": s["jd_info"]["role"], "company": s["jd_info"]["company"],
                "score": s["match"]["score"], "attempts": s["attempts"],
                "first_pass_clean": int(s.get("first_pass_approved", False)),
                "final_clean": int(s["review"]["approved"]),
                "unsupported_claims_left": len(s["review"]["fabricated_claims"]),
                "error": "",
            })
        except (PipelineError, ValueError) as e:
            rows.append({"file": f.name, "role": "", "company": "", "score": "", "attempts": "",
                         "first_pass_clean": 0, "final_clean": 0, "unsupported_claims_left": "",
                         "error": str(e)})

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    n = len(ok)
    if not n:
        raise SystemExit("Every run failed. See the error column in " + args.out)
    first = 100 * sum(r["first_pass_clean"] for r in ok) / n
    final = 100 * sum(r["final_clean"] for r in ok) / n
    print("\n===== Results =====")
    print(f"Job descriptions evaluated : {n} ({len(rows) - n} failed)")
    print(f"Clean on first attempt     : {first:.0f}%  (no self-correction needed)")
    print(f"Clean after review loop    : {final:.0f}%  <- use this as [X]")
    print(f"Average attempts           : {statistics.mean(r['attempts'] for r in ok):.2f}")
    print(f"Average match score        : {statistics.mean(r['score'] for r in ok):.0f}")
    print(f"Details saved to {args.out}")
    print("\nNote: the reviewer is an LLM, so 'clean' means 'passed the reviewer'. "
          "Spot-check 5-10 outputs by hand before quoting the number.")


if __name__ == "__main__":
    main()
