import sys
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator

def validate_data(path: str):
    df = pd.read_csv(path)

    # 1. Basic sanity on date & zip formatting
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    assert df["date"].notna().all(), "Invalid or missing dates"
    assert df["date"].between("2010-01-01", "2025-12-31").all(), "Dates out of expected range"

    df["zipcode_str"] = df["zipcode"].astype(str).str.zfill(5)

    # 2. Create an Ephemeral DataContext (no config files needed)
    context = gx.get_context(mode="ephemeral")

    # 3. Wrap DataFrame in Validator via a Batch
    batch = Batch(data=df)
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[batch],
        data_context=context,
    )

    # 4. Define your expectations
    validator.expect_column_values_to_not_be_null("price")
    validator.expect_column_values_to_be_between("price", min_value=1_000, max_value=12_000_000)
    
    # Allow 0 values (missing data indicators) or realistic price ranges
    validator.expect_column_values_to_be_between("median_sale_price", min_value=0, max_value=19_000_000)
    validator.expect_column_values_to_be_between("median_list_price", min_value=0, max_value=19_000_000)  # Allow for high-end markets but exclude obvious data errors
    
    validator.expect_column_values_to_be_between("homes_sold", min_value=0)
    validator.expect_column_values_to_be_between("pending_sales", min_value=0)
    
    # Allow for longer days on market - some properties take years to sell
    validator.expect_column_values_to_be_between("median_dom", min_value=0, max_value=10_000)
    
    # Allow wider range for sale-to-list ratio (0 for missing data, up to 2.0 for competitive markets)
    validator.expect_column_values_to_be_between("avg_sale_to_list", min_value=0, max_value=2.0)
    
    validator.expect_column_values_to_not_be_null("city_full")
    validator.expect_column_value_lengths_to_equal("zipcode_str", 5)
    
    # Allow 0 for missing population data
    validator.expect_column_values_to_be_between("Total Population", min_value=0)
    validator.expect_column_values_to_be_between("Median Age", min_value=0, max_value=120)
    
    # Allow 0 for missing home value data
    validator.expect_column_values_to_be_between("Median Home Value", min_value=0)

    # 5. Run validation
    results = validator.validate()
    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])
    failed = total - passed

    print(f"\n{path}: {passed}/{total} checks passed")
    if failed:
        print("❌ Failed expectations:")
        for r in results["results"]:
            if not r["success"]:
                config = r["expectation_config"]
                column = config.kwargs.get("column", "N/A")
                expectation_type = config.type
                kwargs = {k: v for k, v in config.kwargs.items() if k != "column"}
                print(f"  - {expectation_type} on column '{column}' with params: {kwargs}")
                
                # Show some details about the failure
                result = r.get("result", {})
                if "observed_value" in result:
                    print(f"    Observed: {result['observed_value']}")
                if "element_count" in result and "unexpected_count" in result:
                    print(f"    Unexpected count: {result['unexpected_count']}/{result['element_count']}")
        sys.exit(1)
    else:
        print("✅ All checks passed!")

if __name__ == "__main__":
    for split in ["data/raw/train.csv", "data/raw/eval.csv", "data/raw/holdout.csv"]:
        validate_data(split)
