#!/usr/bin/env python3
"""
Quick use case lookup for hackathon.

Usage:
    python scripts/usecase.py                    # List all use cases
    python scripts/usecase.py gene               # Show gene expression prediction
    python scripts/usecase.py multimodal         # Show multimodal integration
    python scripts/usecase.py --queries gene     # Just show the queries
    python scripts/usecase.py --code gene        # Just show code template
    python scripts/usecase.py --decisions        # Show decision trees
    python scripts/usecase.py --emergency        # Show emergency fixes
"""

import argparse
import json
from pathlib import Path

USE_CASES_PATH = Path(__file__).parent.parent / "data" / "hackathon_use_cases.json"


def load_use_cases():
    with open(USE_CASES_PATH) as f:
        return json.load(f)


def list_use_cases(data):
    print("\n=== HACKATHON USE CASES ===\n")
    for key, uc in data["use_cases"].items():
        likelihood = uc.get("likelihood", "unknown")
        icon = "🔴" if likelihood == "very_high" else "🟡" if likelihood == "high" else "🟢"
        print(f"{icon} {key}")
        print(f"   {uc['description']}")
    print()


def show_use_case(data, name):
    # Find matching use case
    matches = [k for k in data["use_cases"].keys() if name.lower() in k.lower()]

    if not matches:
        print(f"No use case matching '{name}'")
        print("Available:", ", ".join(data["use_cases"].keys()))
        return

    key = matches[0]
    uc = data["use_cases"][key]

    print(f"\n{'='*60}")
    print(f"USE CASE: {key}")
    print(f"{'='*60}")
    print(f"\n{uc['description']}")
    print(f"Likelihood: {uc.get('likelihood', 'unknown')}")

    if "quick_queries" in uc:
        print(f"\n--- QUICK QUERIES ---")
        for q in uc["quick_queries"]:
            print(f"  {q}")

    if "sota_methods" in uc:
        print(f"\n--- SOTA METHODS ---")
        for method, info in uc["sota_methods"].items():
            repo = info.get("repo", "")
            approach = info.get("approach", "")
            print(f"  • {method}: {approach} {'→ '+repo if repo else ''}")

    if "architecture_choices" in uc:
        print(f"\n--- ARCHITECTURE CHOICES ---")
        for category, options in uc["architecture_choices"].items():
            if isinstance(options, list):
                print(f"  {category}: {', '.join(options)}")
            else:
                print(f"  {category}: {options}")

    if "key_decisions" in uc:
        print(f"\n--- KEY DECISIONS ---")
        for d in uc["key_decisions"]:
            print(f"  ❓ {d}")

    if "polymathic_angles" in uc:
        print(f"\n--- POLYMATHIC ANGLES ---")
        for p in uc["polymathic_angles"]:
            print(f"  💡 {p}")

    if "code_template" in uc:
        print(f"\n--- CODE TEMPLATE ---")
        print(uc["code_template"])

    print()


def show_queries_only(data, name):
    matches = [k for k in data["use_cases"].keys() if name.lower() in k.lower()]
    if not matches:
        return

    uc = data["use_cases"][matches[0]]
    if "quick_queries" in uc:
        print(f"\n# Queries for {matches[0]}:")
        for q in uc["quick_queries"]:
            print(q)


def show_code_only(data, name):
    matches = [k for k in data["use_cases"].keys() if name.lower() in k.lower()]
    if not matches:
        return

    uc = data["use_cases"][matches[0]]
    if "code_template" in uc:
        print(uc["code_template"])


def show_decisions(data):
    print("\n=== QUICK DECISION TREES ===\n")
    for name, tree in data["quick_decision_trees"].items():
        print(f"❓ {tree['question']}")
        for k, v in tree.items():
            if k != "question":
                print(f"   {k}: {v}")
        print()


def show_emergency(data):
    print("\n=== EMERGENCY FIXES ===\n")
    for problem, fixes in data["emergency_fixes"].items():
        print(f"🚨 {problem.upper()}")
        for fix in fixes:
            print(f"   → {fix}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Hackathon use case lookup")
    parser.add_argument("name", nargs="?", help="Use case name (partial match)")
    parser.add_argument("--queries", metavar="NAME", help="Show only queries for use case")
    parser.add_argument("--code", metavar="NAME", help="Show only code for use case")
    parser.add_argument("--decisions", action="store_true", help="Show decision trees")
    parser.add_argument("--emergency", action="store_true", help="Show emergency fixes")

    args = parser.parse_args()
    data = load_use_cases()

    if args.decisions:
        show_decisions(data)
    elif args.emergency:
        show_emergency(data)
    elif args.queries:
        show_queries_only(data, args.queries)
    elif args.code:
        show_code_only(data, args.code)
    elif args.name:
        show_use_case(data, args.name)
    else:
        list_use_cases(data)


if __name__ == "__main__":
    main()
