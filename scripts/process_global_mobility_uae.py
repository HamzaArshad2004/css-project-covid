"""Process Google Community Mobility Reports for UAE COVID analysis."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Global_Mobility_Report.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_data_processed.csv"
BASELINES_FILE = PROJECT_ROOT / "data" / "processed" / "mobility_baselines.json"

START_DATE = "2020-03-01"
END_DATE = "2021-12-31"


def process_uae_mobility(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    baselines_file: Path = BASELINES_FILE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[pd.DataFrame, dict]:
    """Extract and transform UAE mobility data for the selected COVID period."""
    print("=" * 70)
    print("PROCESSING GOOGLE MOBILITY DATA - UAE COVID")
    print("=" * 70)

    print("\nLoading global mobility data...")
    print("(This may take a minute because the file is large)")

    cols_to_load = [
        "country_region",
        "sub_region_1",
        "sub_region_2",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]

    df = pd.read_csv(input_file, usecols=cols_to_load, low_memory=False)
    print(f"Loaded {len(df):,} rows of global data")

    print("\nFiltering for United Arab Emirates...")
    uae_data = df[df["country_region"] == "United Arab Emirates"].copy()
    print(f"Found {len(uae_data):,} rows for UAE")

    national_data = uae_data[uae_data["sub_region_1"].isna()].copy()
    if national_data.empty:
        print("No national-level rows found, aggregating emirate-level rows by date...")
        national_data = uae_data.groupby("date").mean(numeric_only=True).reset_index()
        national_data["country_region"] = "United Arab Emirates"

    national_data["date"] = pd.to_datetime(national_data["date"])
    national_data = national_data[
        (national_data["date"] >= pd.to_datetime(start_date))
        & (national_data["date"] <= pd.to_datetime(end_date))
    ].copy()

    national_data = national_data.rename(
        columns={
            "retail_and_recreation_percent_change_from_baseline": "retail_recreation_change",
            "grocery_and_pharmacy_percent_change_from_baseline": "grocery_pharmacy_change",
            "parks_percent_change_from_baseline": "parks_change",
            "transit_stations_percent_change_from_baseline": "transit_change",
            "workplaces_percent_change_from_baseline": "workplaces_change",
            "residential_percent_change_from_baseline": "residential_change",
        }
    )

    mobility_cols = [
        "retail_recreation_change",
        "grocery_pharmacy_change",
        "parks_change",
        "transit_change",
        "workplaces_change",
        "residential_change",
    ]

    for col in mobility_cols:
        national_data[col] = national_data[col].fillna(0)

    national_data = national_data[["date", "country_region", *mobility_cols]].copy()
    national_data = national_data.rename(columns={"date": "Date"})
    national_data = national_data.sort_values("Date")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    national_data.to_csv(output_file, index=False)

    baselines = {}
    for col in mobility_cols:
        baselines[col] = {
            "mean": float(national_data[col].mean()),
            "std": float(national_data[col].std()),
            "median": float(national_data[col].median()),
            "q25": float(national_data[col].quantile(0.25)),
            "q75": float(national_data[col].quantile(0.75)),
        }

    baselines_file.parent.mkdir(parents=True, exist_ok=True)
    with open(baselines_file, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    print(
        f"\nDate range: {national_data['Date'].min().date()} to {national_data['Date'].max().date()}"
    )
    print(f"Total days: {len(national_data)}")
    print(f"Saved processed data to {output_file}")
    print(f"Saved baselines to {baselines_file}")
    print("\n" + "=" * 70)
    print("UAE COVID MOBILITY PROCESSING COMPLETE")
    print("=" * 70)

    return national_data, baselines


def main() -> None:
    process_uae_mobility()


if __name__ == "__main__":
    main()