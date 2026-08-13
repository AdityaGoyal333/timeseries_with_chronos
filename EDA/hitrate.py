import numpy as np
import scipy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import traceback
import pickle as pl
import glob


def analyze_coverage_non_optimal( all_timestamps, results, wsize, final_arr, birthtime, deathtime, debug_output=False):
    """
    Non-vectorized implementation matching the vectorized logic.
    """
    valid_timestamps = all_timestamps[final_arr]

    # Denominator covers the entire birth-to-death duration.
    total_duration = deathtime - birthtime
    total_windows = int(np.floor(total_duration / wsize)) + 1

    if len(valid_timestamps) == 0:
        results[wsize] = {
            "hits": 0,
            "misses": total_windows,
            "total_windows": total_windows,
            "coverage_rate": 0
        }
        return

    # Align the beginning to a multiple of wsize before the first timestamp.
    first_timestamp = all_timestamps[0].item()
    first_overall = first_timestamp - (first_timestamp % wsize)

    hits = 0

    for window_idx in range(total_windows):
        window_start = first_overall + window_idx * wsize
        window_end = window_start + wsize

        timestamps_in_window = valid_timestamps[
            (valid_timestamps >= window_start) &
            (valid_timestamps < window_end)
        ]

        if len(timestamps_in_window) > 0:
            hits += 1

    misses = total_windows - hits

    results[wsize] = {
        "hits": hits,
        "misses": misses,
        "total_windows": total_windows,
        "coverage_rate": hits / total_windows if total_windows > 0 else 0
    }
    
    
def analyze_coverage_vectorized(all_timestamps, results, wsize, final_arr, birthtime, deathtime, debug_output=False):
    """
    Vectorized version - faster for large datasets
    """

    # these are the actual valid timestamps
    valid_timestamps = all_timestamps[final_arr]
    
    # this is total duration we recroded for 
    total_duration_recorded = all_timestamps[-1] - all_timestamps[0]
    
    # this is the total duration from birth to death
    total_duration = deathtime - birthtime
    
    total_windows = int(np.floor(total_duration / wsize))  + 1
    
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


    print(f"Birthtime: {birthtime}" )
    print(f"Deathtime: {deathtime}")
    print(f"first overall: {first_overall}")
    print("valid recorded timestamps")
    print(all_timestamps[0].item())
    print(all_timestamps[-1].item())
    
    results[wsize] = {
        'hits': hits,
        'misses': misses,
        'total_windows': total_windows,
        'coverage_rate': hits / total_windows if total_windows > 0 else 0
    }

    # return list(record_timestamps)
    

    
    
def analyze_baby_window_coverage(mat_file_path, window_sizes,optimized,debug_output=False):
    """
    Analyze window coverage for a single baby across different window sizes
    """
    
    # Load mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    vname = list(mat_data['vname'])
    pname = list(mat_data['pname'])
    valid_timestamps = mat_data['vt']
        
    if 'BirthTime' in pname and 'Age at Death' in pname:
        if (pname[1][0][0] == 'BirthTime' ) and (pname[-1][0][0] == 'Age at Death'): 
            birthtime = mat_data['pdata'][0][1]
            age_at_death = mat_data['pdata'][0][-1]
            
        else:
            birthtime = mat_data['pdata'][0][pname.index('BirthTime')]
            age_at_death = mat_data['pdata'][0][pname.index('Age at Death')]
    else:
        raise Exception(f"{mat_file_path}: File does not have birth or age at death")
  
    # for label = 0 datapoints , its no actually the deathtime
    if(np.isnan(age_at_death)):
        deathtime = valid_timestamps[-1].item()
    else:
        deathtime = age_at_death*86400 + birthtime
        
    if(debug_output):
        print(f"age at death: {age_at_death}",flush=True)
    
    if 'HR' in vname and 'SPO2-%' in vname:
        if (vname[0][0][0] == 'HR') and (vname[2][0][0] == 'SPO2-%'): 
            hr_cols = mat_data['vdata'][:,0]
            spo2_cols = mat_data['vdata'][:,2]
        else:
            hr_cols = mat_data['vdata'][:,vname.index('HR')]
            spo2_cols = mat_data['vdata'][:,vname.index('SPO2-%')]
    else:
        raise Exception(f"{mat_file_path}: File does not have HR or SPO2")
    
    # Step 1: Filter for non-null HR and SPO2 values
    final_arr = ~np.isnan(hr_cols) & ~np.isnan(spo2_cols)
    # print(f"sum: {sum(final_arr)}")

    # Step 2: Analyze for each window size
    results = {}
    for wsize in window_sizes:
        # print(f"Analyzing for {mat_file_path}")
        # print(f" - analyzing for window size: {wsize}")
        
        if(optimized):
            analyze_coverage_vectorized(valid_timestamps, results, wsize, final_arr,birthtime,deathtime,debug_output=True)
            # return analyze_coverage_vectorized(valid_timestamps, results, wsize, final_arr)
            
        else:
            analyze_coverage_non_optimal(valid_timestamps, results, wsize, final_arr,birthtime,deathtime,debug_output=True)
            # analyze_coverage(valid_timestamps,results,wsize,final_arr)
            # return analyze_coverage_non_optimal(valid_timestamps, results, wsize, final_arr)
    
    return results




def save_results_to_pickle(summary_df, baby_results, window_sizes,filename):
    """Save results to pickle file"""
    results_dict = {
        'summary_df': summary_df,
        'baby_results': baby_results,
        'window_sizes': window_sizes
    }
    
    with open(filename, 'wb') as f:
        pl.dump(results_dict, f)
    
    print(f"Results saved to {filename}")
    
    

# here workers means threads
def analyze_all_babies_windows(window_sizes, mat_files, optimized, filename, max_workers=30):
    """
    Analyze window coverage for all babies concurrently.

    Each worker analyzes one baby and returns its result.
    Shared result structures are updated only by the main thread.
    """

    all_results = []
    baby_results = {}

    print(
        f"Analyzing {len(mat_files)} babies "
        f"with window sizes: {window_sizes}"
    )

    def analyze_one_baby(index, file_path):
        """Run in a worker thread."""
        baby_result = analyze_baby_window_coverage(
            file_path,
            window_sizes,
            optimized,
        )

        baby_id = file_path.split("/")[-1]

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
            for i, file_path in enumerate(mat_files)
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
                    f"\nFailed baby {index + 1}/{len(mat_files)}: "
                    f"{file_path}"
                )
                traceback.print_exc()
                continue

            # These writes occur in the main thread, so no lock is needed.
            baby_results[baby_id] = baby_result
            all_results.append(row)
            completed +=1
            
            if(completed%100==0):
                save_results_to_pickle(pd.DataFrame(all_results), baby_results,window_sizes, filename=filename)

                print(
                    f"Completed {completed_count}/{len(mat_files)}: "
                    # f"{file_path}"
                )

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)    
    return summary_df, baby_results



def run_non_optimal_analysis(window_sizes,mat_files):
    """Thread function for non-optimal analysis"""
    filename = 'results_non_optimal.pkl'
    print("Starting non-optimal analysis...")
    summary_df, baby_results = analyze_all_babies_windows(window_sizes, mat_files, False, filename)
    save_results_to_pickle(summary_df, baby_results, window_sizes, filename=filename)
    print("Non-optimal analysis completed!")


def run_vectorized_analysis(window_sizes,mat_files):
    """Thread function for vectorized analysis"""
    filename = 'results_vectorized.pkl'
    print("Starting vectorized analysis...")
    summary_df, baby_results = analyze_all_babies_windows(window_sizes, mat_files, True, filename)
    save_results_to_pickle(summary_df, baby_results, window_sizes, filename=filename)
    print("Vectorized analysis completed!")
    
    

if __name__ == "__main__":
    
    
    """
    for validating both the optimal and vectorized implementations (already done)
    """
    # Create threads
    # thread_non_optimal = threading.Thread(target=run_non_optimal_analysis)
    # thread_vectorized = threading.Thread(target=run_vectorized_analysis)

    # Start both threads
    # print("Starting both analyses concurrently...")
    # thread_non_optimal.start()
    # thread_vectorized.start()

    # Wait for both to complete
    # thread_non_optimal.join()
    # thread_vectorized.join()

    # print("Both analyses completed!")
    
    # arr = [int(math.pow(1.338,x)) for x in range(20,30)]
    
    
    """
    for running for a group of mat files and window sizes
    """
    # mat_files = set(sorted(glob.glob('./data/download/Vitals/*.mat')))
    # window_sizes = [1, 2, 3, 4, 5, 10, 20, 24, 32, 44, 58, 78, 105, 141, 188, 252, 300, 1083, 1940, 3000] 
    # run_vectorized_analysis(window_sizes,mat_files)
    
    
    """
    for testing
    """
    # print(analyze_baby_window_coverage('/Users/adityagoyal/Desktop/Research - yin li/timeseries_project/timeseries_with_chronos/data/download/Vitals/NICU_5480_vitals.mat', [50],True,True))
    
    