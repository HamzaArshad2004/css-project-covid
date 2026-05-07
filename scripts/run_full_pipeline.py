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


def run_step(step_num, step_name, script_name, required=True, extra_args=None):
    """Run a pipeline step"""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 70)

    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd)

    if result.returncode != 0 and required:
        print(f"\n❌ Error in {step_name}")
        print(f"Pipeline stopped. Please fix errors in {script_name}")
        sys.exit(1)

    return result.returncode == 0

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
    print("1. Run full pipeline (all steps)")
    print("2. Skip data collection (use existing data)")
    print("3. Run only analysis (FCA + visualization)")
    print("4. Custom (choose steps)")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    # Ask about LLM scoring for options that include the evaluation step
    use_llm = False
    if choice in ("1", "2", "3", "4"):
        llm_choice = input(
            "\nEnable LLM scoring for rule evaluation? "
            "(requires OPENAI_API_KEY in .env) [y/N]: "
        ).strip().lower()
        use_llm = llm_choice in ("y", "yes")

    llm_extra = ["--llm"] if use_llm else []

    if choice == "1":
        # Full pipeline
        run_step(1, "Reddit Data Collection", "collect_reddit_data.py")
        run_step(2, "Mobility Data Processing", "collect_mobility_data.py")
        run_step(3, "Feature Engineering", "preprocess_and_features.py")
        run_step(4, "Formal Concept Analysis", "fca_analysis.py")
        run_step(5, "Visualization", "visualize_results.py")
        run_step(6, "Rule Evaluation (LLM)", "evaluate_rules_llm.py", required=False, extra_args=llm_extra)

    elif choice == "2":
        # Skip collection
        print("\n⏭️  Skipping data collection...")
        run_step(2, "Mobility Data Processing", "collect_mobility_data.py")
        run_step(3, "Feature Engineering", "preprocess_and_features.py")
        run_step(4, "Formal Concept Analysis", "fca_analysis.py")
        run_step(5, "Visualization", "visualize_results.py")
        run_step(6, "Rule Evaluation (LLM)", "evaluate_rules_llm.py", required=False, extra_args=llm_extra)

    elif choice == "3":
        # Only analysis
        print("\n⏭️  Skipping data collection and preprocessing...")
        run_step(4, "Formal Concept Analysis", "fca_analysis.py")
        run_step(5, "Visualization", "visualize_results.py")
        run_step(6, "Rule Evaluation (LLM)", "evaluate_rules_llm.py", required=False, extra_args=llm_extra)

    elif choice == "4":
        # Custom
        print("\nSelect steps to run (space-separated, e.g., '1 3 4'):")
        print("  1: Reddit Collection")
        print("  2: Mobility Processing")
        print("  3: Feature Engineering")
        print("  4: FCA Analysis")
        print("  5: Visualization")
        print("  6: Rule Evaluation (LLM)")

        steps = input("\nSteps: ").strip().split()

        step_map = {
            '1': ("Reddit Data Collection", "collect_reddit_data.py", True, []),
            '2': ("Mobility Data Processing", "collect_mobility_data.py", True, []),
            '3': ("Feature Engineering", "preprocess_and_features.py", True, []),
            '4': ("Formal Concept Analysis", "fca_analysis.py", True, []),
            '5': ("Visualization", "visualize_results.py", True, []),
            '6': ("Rule Evaluation (LLM)", "evaluate_rules_llm.py", False, llm_extra),
        }

        for i, step_id in enumerate(steps, start=1):
            if step_id in step_map:
                name, script, required, extra = step_map[step_id]
                run_step(i, name, script, required=required, extra_args=extra)
    
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  📁 data/processed/")
    print("     - mobility_data_processed.csv")
    print("     - reddit_sentiment_features.csv")
    print("     - fca_binary_matrix.csv")
    print("\n  📁 results/fca/")
    print("     - formal_concepts.csv")
    print("     - association_rules.csv")
    print("     - association_rules_evaluated.csv")
    print("     - top_cross_domain_rules.txt")
    print("     - galois_lattice.png")
    print("     - crisis_context.cxt (for Galicia)")
    print("\n  📁 results/visualizations/")
    print("     - mobility_trends.png")
    print("     - sentiment_timeline.png")
    print("     - features_heatmap.png")
    print("     - combined_analysis.png")
    print("\n  📄 results/summary_report.txt")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()