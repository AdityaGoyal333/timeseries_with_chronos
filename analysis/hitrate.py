import numpy as np
import scipy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import traceback
import pickle as pl
import glob    
from pathlib import Path


CURRENT_FILE_PATH =  Path(__file__).resolve()



def analyze_coverage_vectorized_recorded(all_timestamps, results, wsize, final_arr):
    """
    Vectorized version - faster for large datasets
    """

    # these are the actual valid timestamps
    valid_timestamps = all_timestamps[final_arr]
    
    # this is total duration we recroded for 
    total_duration_recorded = all_timestamps[-1] - all_timestamps[0]
    
    total_windows = int(np.floor(total_duration_recorded / wsize))  + 1
    
    # record_timestamps = []
    if len(valid_timestamps) == 0:
        results[wsize] = {
            'hits': 0, 
            'misses': total_windows, 
            'total_windows': total_windows,
            'coverage_rate': 0
        }
        return
    
    # first_overall = all_timestamps[0].item()
    # the first_overall will change to the timestamp which is just before the firstoverall whose difference with birthtime is divisible by window size
    # first_overall = all_timestamps[0].item() - (all_timestamps[0].item()-birthtime)%wsize
    
    # the vt timestamps are already wrt to the birth time 
    first_overall = all_timestamps[0].item() - all_timestamps[0].item()%wsize
    
    # Vectorized approach: assign each timestamp to a window
    window_indices = np.floor((valid_timestamps - first_overall) / wsize).astype(int)
    
    # Count unique windows that have at least one timestamp
    unique_windows = np.unique(window_indices)
    hits = len(unique_windows)    
    misses = total_windows - hits

    # print(f"first overall: {first_overall}")
    # print("valid recorded timestamps")
    print(all_timestamps[0].item())
    print(all_timestamps[-1].item())
    
    results[wsize] = {
        'hits': hits,
        'misses': misses,
        'total_windows': total_windows,
        'coverage_rate': hits / total_windows if total_windows > 0 else 0
    }



def analyze_baby_window_coverage(parquet_file_path, window_sizes):
    """
    Analyze window coverage for a single baby across different window sizes
    """
    
    # Load mat file
    parquet_data = pd.read_parquet(parquet_file_path)
    
    valid_timestamps = parquet_data['age_us'].values
    hr_cols = parquet_data['HR'].values
    spo2_cols = parquet_data['SPO2'].values
    
    # Step 1: Filter for non-null HR and SPO2 values
    final_arr = ~np.isnan(hr_cols) & ~np.isnan(spo2_cols)
    # print(f"sum: {sum(final_arr)}")

    # Step 2: Analyze for each window size
    results = {}
    for wsize in window_sizes:
        # print(f"Analyzing for {mat_file_path}")
        # print(f" - analyzing for window size: {wsize}")
        
        analyze_coverage_vectorized_recorded(valid_timestamps, results, wsize, final_arr)

    print(results)
    return results



# here workers means threads
def analyze_all_babies_windows(window_sizes, parquet_files, file_path, max_workers=30):
    """
    Analyze window coverage for all babies concurrently.

    Each worker analyzes one baby and returns its result.
    Shared result structures are updated only by the main thread.
    """

    all_results = []
    baby_results = {}

    print(
        f"Analyzing {len(parquet_files)} babies "
        f"with window sizes: {window_sizes}"
    )

    def analyze_one_baby(index, file_path):
        """Run in a worker thread."""
        baby_result = analyze_baby_window_coverage(
            file_path,
            window_sizes
        )

        baby_id = file_path.split("/")[-1].split(".")[0]

        row = OrderedDict()
        row["baby_id"] = baby_id

        for wsize in window_sizes:
            row[f"coverage_{wsize}s"] = baby_result[wsize]["coverage_rate"]
            row[f"hits_{wsize}s"] = baby_result[wsize]["hits"]
            row[f"total_windows_{wsize}s"] = (
                baby_result[wsize]["total_windows"]
            )

        return index, baby_id, baby_result, row

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(analyze_one_baby, i, file_path): (i, file_path)
            for i, file_path in enumerate(parquet_files)
        }

        for completed_count, future in enumerate(
            as_completed(future_to_file),
            start=1,
        ):
            index, file_path = future_to_file[future]

            try:
                index, baby_id, baby_result, row = future.result()
                print(f"processed {baby_id}",flush=True)

            except Exception:
                print(
                    f"\nFailed baby {index + 1}/{len(parquet_files)}: "
                    f"{file_path}"
                )
                traceback.print_exc()
                continue

            # These writes occur in the main thread, so no lock is needed.
            baby_results[baby_id] = baby_result
            all_results.append(row)
            completed +=1
            
            if(completed%10==0):
                save_results_to_pickle(pd.DataFrame(all_results), baby_results,window_sizes, file_path=file_path)

                print(
                    f"Completed {completed_count}/{len(parquet_files)}: "
                    # f"{file_path}"
                )

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)    
    return summary_df, baby_results




def save_results_to_pickle(summary_df, baby_results, window_sizes,file_path):
    """Save results to pickle file"""
    results_dict = {
        'summary_df': summary_df,
        'baby_results': baby_results,
        'window_sizes': window_sizes
    }
    
    with open(file_path, 'wb') as f:
        pl.dump(results_dict, f)
    
    print(f"Results saved to {file_path}")
    
    
def run_vectorized_analysis(window_sizes,parquet_files):
    """Thread function for vectorized analysis"""
    file_path = 'results_vectorized_new_data.pkl'
    file_path = CURRENT_FILE_PATH.parent.parent / "data" / file_path

    print("Starting vectorized analysis...")
    summary_df, baby_results = analyze_all_babies_windows(window_sizes, parquet_files, file_path)
    save_results_to_pickle(summary_df, baby_results, window_sizes, file_path=file_path)
    print("Vectorized analysis completed!")
    


if __name__ == "__main__":
    
    """
    for running for a group of mat files and window sizes
    """
    
    parquet_files_pattern = str(CURRENT_FILE_PATH.parent.parent / "vitals"/ "*.parquet")
    parquet_files = list(set(sorted(glob.glob(parquet_files_pattern))))[:30]
    window_sizes_seconds = [1, 2, 3, 4, 5, 10, 20, 24, 32, 44, 58, 78, 105, 141, 188, 252, 300, 1083, 1940, 3000] 
    window_sizes_microseconds = [window_size*1000000 for window_size in window_sizes_seconds]
    run_vectorized_analysis(window_sizes_microseconds,parquet_files)
    