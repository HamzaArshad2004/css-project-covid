"""
Master Pipeline Script
Runs the complete analysis from data collection to visualization
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def check_directories():
    """Ensure all required directories exist"""
    dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "results" / "fca",
        PROJECT_ROOT / "results" / "visualizations",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def run_step(step_num, step_name, script_name=None, required=True, extra_args=None, module=None):
    """Run a pipeline step, either as a script or a Python module (-m)."""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 70)

    if module is not None:
        cmd = [sys.executable, "-m", module]
        if extra_args:
            cmd.extend(extra_args)
        # Run from project root so src.* imports resolve correctly
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elif script_name is not None:
        cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(cmd)
    else:
        print(f"  WARNING: no script or module specified for step {step_num}; skipping.")
        return True

    if result.returncode != 0 and required:
        print(f"\nError in {step_name}")
        print(f"Pipeline stopped. Please fix errors in {script_name or module}")
        sys.exit(1)

    return result.returncode == 0


def build_lagged_matrix(step_num):
    """Build the predictive (lagged/lead) feature matrix.

    IMPORTANT: this MUST run before fca_analysis.py, because fca_analysis.py
    mines predictive rules from data/processed/fca_predictive_matrix.csv only
    if that file already exists. If the matrix is built afterward, predictive
    mining is silently skipped on a clean run and no predictive rules are
    produced. It also depends on fca_binary_matrix.csv, so it must run after
    feature engineering.
    """
    return run_step(
        step_num, "Build Lagged Feature Matrix",
        script_name=None,
        required=False,
        module="src.features.lagged_features",
    )


def main():
    print("=" * 70)
    print("CRISIS BEHAVIORAL ANALYSIS - FULL PIPELINE")
    print("UAE COVID-19 Analysis")
    print("=" * 70)
    
    # Check environment
    print("\n🔍 Checking environment...")
    check_directories()
    print("  ✓ Directories OK")
    
    # Ask user what to run
    print("\n" + "=" * 70)
    print("PIPELINE OPTIONS")
    print("=" * 70)
    print("1. Run full pipeline (all steps, including quality gates)")
    print("2. Skip data collection (use existing data)")
    print("3. Run only analysis (FCA + post-processing + visualization)")
    print("4. Run post-processing only (prune → score → backtest → briefs)")
    print("5. Custom (choose steps)")

    choice = input("\nSelect option (1-5): ").strip()

    # Ask about LLM scoring for options that include the evaluation step
    use_llm = False
    if choice in ("1", "2", "3", "5"):
        llm_choice = input(
            "\nEnable LLM scoring for rule evaluation? "
            "(requires OPENAI_API_KEY in .env) [y/N]: "
        ).strip().lower()
        use_llm = llm_choice in ("y", "yes")

    llm_extra = ["--llm"] if use_llm else []

    def run_postprocessing(step_offset: int = 0) -> None:
        """Run the new quality-gate post-processing chain.

        Note: the lagged feature matrix is built earlier in the pipeline (before
        FCA), so it is NOT part of this chain. This chain assumes both the
        same-day rules and the predictive rules have already been mined.
        """
        # Prune rules
        run_step(
            step_offset + 1, "Prune Rules (quality gates + redundancy)",
            script_name=None,
            required=False,
            module="src.rules.prune_rules",
        )
        # Score rules (composite metric)
        run_step(
            step_offset + 2, "Score Rules (composite metric)",
            script_name=None,
            required=False,
            module="src.scoring.score_rules",
        )
        # Temporal backtest
        run_step(
            step_offset + 3, "Temporal Backtest (stability check)",
            script_name=None,
            required=False,
            module="src.validation.temporal_backtest",
        )
        # Policy briefs
        run_step(
            step_offset + 4, "Generate Policy Briefs",
            script_name=None,
            required=False,
            module="src.reporting.policy_briefs",
        )

    if choice == "1":
        # Full pipeline
        run_step(1, "Reddit Data Collection", "collect_reddit_data.py")
        run_step(2, "Mobility Data Processing", "collect_mobility_data.py")
        run_step(3, "Feature Engineering", "preprocess_and_features.py")
        # Build the predictive matrix BEFORE FCA so predictive rules get mined.
        build_lagged_matrix(4)
        run_step(5, "Formal Concept Analysis", "fca_analysis.py")
        run_step(6, "Visualization", "visualize_results.py")
        run_step(7, "Rule Evaluation (LLM + Pruning)", "evaluate_rules_llm.py",
                 required=False, extra_args=llm_extra)
        run_postprocessing(step_offset=7)

    elif choice == "2":
        # Skip collection
        print("\n  Skipping data collection...")
        run_step(2, "Mobility Data Processing", "collect_mobility_data.py")
        run_step(3, "Feature Engineering", "preprocess_and_features.py")
        build_lagged_matrix(4)
        run_step(5, "Formal Concept Analysis", "fca_analysis.py")
        run_step(6, "Visualization", "visualize_results.py")
        run_step(7, "Rule Evaluation (LLM + Pruning)", "evaluate_rules_llm.py",
                 required=False, extra_args=llm_extra)
        run_postprocessing(step_offset=7)

    elif choice == "3":
        # Only analysis — still rebuild the lagged matrix first so FCA can mine
        # predictive rules from it. Assumes feature engineering already ran and
        # fca_binary_matrix.csv exists.
        print("\n  Skipping data collection and preprocessing...")
        build_lagged_matrix(4)
        run_step(5, "Formal Concept Analysis", "fca_analysis.py")
        run_step(6, "Visualization", "visualize_results.py")
        run_step(7, "Rule Evaluation (LLM + Pruning)", "evaluate_rules_llm.py",
                 required=False, extra_args=llm_extra)
        run_postprocessing(step_offset=7)

    elif choice == "4":
        # Post-processing chain only
        run_postprocessing(step_offset=0)

    elif choice == "5":
        # Custom
        print("\nSelect steps to run (space-separated, e.g., '1 3 4'):")
        print("  1: Reddit Collection")
        print("  2: Mobility Processing")
        print("  3: Feature Engineering")
        print("  4: Build Lagged Feature Matrix  (run BEFORE FCA for predictive rules)")
        print("  5: FCA Analysis  (mines same-day AND predictive rules)")
        print("  6: Visualization")
        print("  7: Rule Evaluation (LLM + Pruning)")
        print("  8: Prune Rules")
        print("  9: Score Rules")
        print(" 10: Temporal Backtest")
        print(" 11: Policy Briefs")
        print("\n  NOTE: step 4 must run before step 5 for predictive rules to be mined.")

        steps = input("\nSteps: ").strip().split()

        step_map = {
            '1':  ("Reddit Data Collection",          None, "collect_reddit_data.py",       True,  []),
            '2':  ("Mobility Data Processing",         None, "collect_mobility_data.py",     True,  []),
            '3':  ("Feature Engineering",              None, "preprocess_and_features.py",   True,  []),
            '4':  ("Build Lagged Matrix",  "src.features.lagged_features",       None, False, []),
            '5':  ("Formal Concept Analysis",          None, "fca_analysis.py",              True,  []),
            '6':  ("Visualization",                    None, "visualize_results.py",         True,  []),
            '7':  ("Rule Evaluation",                  None, "evaluate_rules_llm.py",        False, llm_extra),
            '8':  ("Prune Rules",          "src.rules.prune_rules",              None, False, []),
            '9':  ("Score Rules",          "src.scoring.score_rules",            None, False, []),
            '10': ("Temporal Backtest",    "src.validation.temporal_backtest",   None, False, []),
            '11': ("Policy Briefs",        "src.reporting.policy_briefs",        None, False, []),
        }

        # Warn if the user asked for FCA without building the lagged matrix first
        if '5' in steps and '4' not in steps:
            predictive_matrix = PROJECT_ROOT / "data" / "processed" / "fca_predictive_matrix.csv"
            if not predictive_matrix.exists():
                print(
                    "\n  WARNING: You selected FCA (5) without building the lagged matrix (4),\n"
                    "  and no fca_predictive_matrix.csv exists yet. Predictive rules will NOT\n"
                    "  be mined. Add step 4 before step 5 to mine predictive rules."
                )

        for i, step_id in enumerate(steps, start=1):
            if step_id in step_map:
                entry = step_map[step_id]
                name, module, script, required, extra = entry
                run_step(i, name, script_name=script, required=required,
                         extra_args=extra, module=module)

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  data/processed/")
    print("     - fca_binary_matrix.csv")
    print("     - fca_predictive_matrix.csv  (built before FCA so predictive rules are mined)")
    print("\n  results/fca/")
    print("     - association_rules.csv              (raw, tightened thresholds)")
    print("     - association_rules_predictive.csv   (lagged/lead rules)")
    print("     - association_rules_predictive_pruned.csv  (clean predictive set)")
    print("     - association_rules_pruned.csv        (after quality gates)")
    print("     - association_rules_scored.csv        (composite-scored)")
    print("     - association_rules_stable.csv        (backtest-validated)")
    print("     - backtest_report.csv")
    print("     - association_rules_evaluated.csv    (LLM-ranked, if --llm used)")
    print("     - top_cross_domain_rules.txt")
    print("\n  results/")
    print("     - policy_briefs.txt                  (operational policy output)")
    print("     - sentiment_timeline.png")
    print("     - features_heatmap.png")
    print("     - combined_analysis.png")
    print("\n  📄 results/summary_report.txt")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()