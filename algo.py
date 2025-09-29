import os
import random
import time
import pandas as pd
import matplotlib.pyplot as plt


# CONFIGURATION
INPUT_DIR = "."  # folder containing CSV files
OUTPUT_CSV = "benchmark_results.csv"
OUTPUT_TABLE_CSV = "benchmark_results_table.csv"
OUTPUT_PLOT = "benchmark_plot.png"

BRUTE_FORCE_LIMIT = 5000  # skip brute force on larger files


# ALGORITHMS

def two_sum_bruteforce(arr, target):
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (arr[i], arr[j])
    return None

def two_sum_sorting(arr, target):
    arr_sorted = sorted(arr)
    left, right = 0, len(arr_sorted) - 1
    while left < right:
        s = arr_sorted[left] + arr_sorted[right]
        if s == target:
            return (arr_sorted[left], arr_sorted[right])
        elif s < target:
            left += 1
        else:
            right -= 1
    return None

def two_sum_hashing(arr, target):
    seen = set()
    for num in arr:
        complement = target - num
        if complement in seen:
            return (num, complement)
        seen.add(num)
    return None


# HELPER FUNCTIONS

def time_function(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start

def load_csv_files(input_dir):
    data_files = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv") and file not in [OUTPUT_CSV, OUTPUT_TABLE_CSV]:
            path = os.path.join(input_dir, file)
            try:
                df = pd.read_csv(path, header=0)
                first_col = df.iloc[:, 0]
                arr = pd.to_numeric(first_col, errors="coerce").dropna().astype(int).tolist()
                if len(arr) > 1:  # need at least 2 numbers
                    data_files.append((file, arr))
                    print(f"Loaded {file} -> {len(arr)} integers.")
                else:
                    print(f"{file} has insufficient valid integers, skipping.")
            except Exception as e:
                print(f"Failed to read {path}: {e}")
    return data_files

def run_benchmarks(data_files):
    results = []
    for filename, arr in data_files:
        n = len(arr)
        # choose a target guaranteed to exist
        i, j = random.sample(range(n), 2)
        target = arr[i] + arr[j]
        print(f"Running {filename} (n={n}), target={target}")

        row = {"file": filename, "n": n, "target": target}

        # Brute-force
        if n <= BRUTE_FORCE_LIMIT:
            res, t = time_function(two_sum_bruteforce, arr, target)
            row["brute_force_time"] = t
            row["brute_force_found"] = bool(res)
        else:
            row["brute_force_time"] = None
            row["brute_force_found"] = None

        # Sorting + two pointers
        res, t = time_function(two_sum_sorting, arr, target)
        row["sorting_time"] = t
        row["sorting_found"] = bool(res)

        # Hashing
        res, t = time_function(two_sum_hashing, arr, target)
        row["hashing_time"] = t
        row["hashing_found"] = bool(res)

        results.append(row)

    return pd.DataFrame(results)

def print_and_save_table(df):
    # Format runtimes for display
    df_display = df.copy()
    df_display["brute_force_time"] = df_display["brute_force_time"].apply(
        lambda x: f"{x:.5f}" if pd.notnull(x) else "skipped"
    )
    df_display["sorting_time"] = df_display["sorting_time"].apply(
        lambda x: f"{x:.5f}" if pd.notnull(x) else "-"
    )
    df_display["hashing_time"] = df_display["hashing_time"].apply(
        lambda x: f"{x:.5f}" if pd.notnull(x) else "-"
    )

    # Convert found columns to Yes/No
    for col in ["brute_force_found", "sorting_found", "hashing_found"]:
        df_display[col] = df_display[col].apply(lambda x: "Yes" if x else ("No" if x == False else "-"))

    columns_to_show = ["file", "n", "brute_force_time", "sorting_time", "hashing_time",
                       "brute_force_found", "sorting_found", "hashing_found"]

    print("\n=== Two-Sum Algorithm Benchmark Table ===")
    print(df_display[columns_to_show].to_string(index=False))

    # Save CSV
    df_display[columns_to_show].to_csv(OUTPUT_TABLE_CSV, index=False)
    print(f"\nSaved formatted table to {OUTPUT_TABLE_CSV}")

def plot_results(df, output_file):
    if df.empty:
        print("No results to plot.")
        return
    plt.figure(figsize=(10,6))
    grouped = df.groupby("n").mean(numeric_only=True)

    if "brute_force_time" in grouped:
        plt.plot(grouped.index, grouped["brute_force_time"], label="Brute Force O(n²)", marker="o")
    plt.plot(grouped.index, grouped["sorting_time"], label="Sorting + Two Pointers O(n log n)", marker="o")
    plt.plot(grouped.index, grouped["hashing_time"], label="Hashing O(n)", marker="o")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size (n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Two-Sum Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


# MAIN

if __name__ == "__main__":
    print("Reading data files...")
    data_files = load_csv_files(INPUT_DIR)
    if not data_files:
        print("No valid CSV files found.")
        exit(1)

    print("Running benchmarks...")
    results = run_benchmarks(data_files)

    # Save raw benchmark results
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved raw benchmark results to {OUTPUT_CSV}")

    # Print and save formatted table
    print_and_save_table(results)

    # Plot runtime comparison
    plot_results(results, OUTPUT_PLOT)

    
    # SUMMARY STATEMENT
    
    print("\n=== Summary ===")
    print("1. Brute Force: O(n²) time, very slow for large n, but simple and guaranteed to find a pair if it exists.")
    print("2. Sorting + Two Pointers: O(n log n) time, faster than brute-force, uses moderate memory, suitable for medium datasets.")
    print("3. Hash Table: O(n) time, fastest approach, uses extra memory for storage, ideal for large datasets and real-time solutions.")
    print("This comparison illustrates the trade-off between time complexity and memory usage, and explains why hash table is preferred for large-scale or real-time energy management applications.")

    print("\nDone.")
