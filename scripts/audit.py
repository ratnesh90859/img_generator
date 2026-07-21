"""
audit.py — Standalone admin script
────────────────────────────────────
Run this anytime to check the health of all generated models:

  python scripts/audit.py

Outputs a table of all models with issues found.
"""

import sys
import os
import io

# Force UTF-8 output on Windows terminals that default to cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.validator import audit_all_models, full_validation_report
from agent.pipeline import get_vehicle_status
from google.cloud import firestore
from config.settings import FIRESTORE_COLLECTION


def print_audit():
    print("\n" + "=" * 60)
    print("  Kotak 360° Vehicle POC — Full Audit Report")
    print("=" * 60)

    _fs   = firestore.Client()
    docs  = list(_fs.collection(FIRESTORE_COLLECTION).stream())
    total = len(docs)

    print(f"\nTotal models in Firestore: {total}")
    print()

    issues = []

    for doc in docs:
        d      = doc.to_dict()
        mid    = d.get("model_id", "?")
        name   = d.get("display_name", mid)
        status = d.get("status", "unknown")
        frames = d.get("generated_frames", 0)
        total_f = d.get("total_frames", 36)
        failed  = len(d.get("failed_frames", []))

        icon = "[OK] " if status == "completed" and failed == 0 else "[PART]" if status in ("partial",) else "[FAIL]"
        print(f"  {icon}  {name:<30} status={status:<12} frames={frames}/{total_f}  failures={failed}")

        if status == "completed":
            # Spot-check frame count
            from agent.gcs_uploader import list_existing_frames
            existing = len(list_existing_frames(mid))
            if existing < total_f:
                issues.append({
                    "model": name,
                    "issue": f"GCS has {existing} frames but Firestore says {total_f} completed"
                })
                print(f"       ⚠️  GCS mismatch: {existing} frames on disk vs {total_f} expected!")

    print()
    if issues:
        print(f"[!] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"   - {issue['model']}: {issue['issue']}")
    else:
        print("[OK] All models look healthy.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_audit()
