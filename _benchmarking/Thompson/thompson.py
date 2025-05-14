#!/usr/bin/env python
"""
Comparison of ts_main.py implementations:

This docstring highlights the differences between the two versions of the `ts_main.py` script.

### Key Differences:

1. **Module Imports**:
   - **Old Version**:
     - Imports `ThompsonSampler` from `thompson_sampling`.
     - Imports `evaluators` in the `read_input` and `parse_input_dict` functions.
   - **New Version**:
     - Imports `ThompsonSampler` from `thompson_sampling_iwmspy`.
     - Imports `evaluators_iwmspy` in the `parse_input_dict` function.

   **Reason**: The new version uses a modified `ThompsonSampler` implementation (`thompson_sampling_iwmspy`) and a custom evaluator module (`evaluators_iwmspy`) for additional functionality or customizations.

2. **`parse_input_dict` Function**:
   - **Old Version**:
     - Imports the `evaluator_class_name` from the `evaluators` module.
   - **New Version**:
     - Imports the `evaluator_class_name` from the `evaluators_iwmspy` module.

   **Reason**: The new version integrates with the custom evaluator module (`evaluators_iwmspy`) instead of the standard `evaluators` module.

3. **ThompsonSampler Integration**:
   - **Old Version**:
     - Uses the `ThompsonSampler` class from the `thompson_sampling` module.
   - **New Version**:
     - Uses the `ThompsonSampler` class from the `thompson_sampling_iwmspy` module.

   **Reason**: The new version likely includes enhancements or modifications to the `ThompsonSampler` class to support additional features or custom behavior.
"""

import importlib
import json
import sys, os
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TS"))

from thompson_sampling_iwmspy import ThompsonSampler
from ts_logger import get_logger
import evaluators_iwmspy


def read_input(json_filename: str) -> dict:
    """
    Read input parameters from a json file
    :param json_filename: input json file
    :return: a dictionary with the input parameters
    """
    input_data = None
    with open(json_filename, 'r') as ifs:
        input_data = json.load(ifs)
        module = importlib.import_module("evaluators_iwmspy")
        evaluator_class_name = input_data["evaluator_class_name"]
        class_ = getattr(module, evaluator_class_name)
        evaluator_arg = input_data["evaluator_arg"]
        evaluator = class_(evaluator_arg)
        input_data['evaluator_class'] = evaluator
    return input_data


def parse_input_dict(input_data: dict) -> None:
    """
    Parse the input dictionary and add the necessary information
    :param input_data:
    """
    evaluator_class_name = input_data["evaluator_class_name"]
    class_ = getattr(evaluators_iwmspy, evaluator_class_name)
    evaluator_arg = input_data["evaluator_arg"]
    evaluator = class_(evaluator_arg)
    input_data['evaluator_class'] = evaluator


def run_ts(input_dict: dict, hide_progress: bool = False) -> None:
    """
    Perform a Thompson sampling run
    :param hide_progress: hide the progress bar
    :param input_dict: dictionary with input parameters
    """
    evaluator = input_dict["evaluator_class"]
    reaction_smarts = input_dict["reaction_smarts"]
    num_ts_iterations = input_dict["num_ts_iterations"]
    reagent_file_list = input_dict["reagent_file_list"]
    num_warmup_trials = input_dict["num_warmup_trials"]
    result_filename = input_dict.get("results_filename")
    ts_mode = input_dict["ts_mode"]
    log_filename = input_dict.get("log_filename")
    logger = get_logger(__name__, filename=log_filename)
    ts = ThompsonSampler(mode=ts_mode)
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)
    ts.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    ts.set_reaction(reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    ts.warm_up(num_warmup_trials=num_warmup_trials)
    # run the search with TS
    out_list = ts.search(num_cycles=num_ts_iterations)
    total_evaluations = evaluator.counter
    percent_searched = total_evaluations / ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    # write the results to disk
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
    if result_filename is not None:
        out_df.to_csv(result_filename, index=False)
        logger.info(f"Saved results to: {result_filename}")
    if not hide_progress:
        if ts_mode == "maximize":
            print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(10))
        else:
            print(out_df.sort_values("score", ascending=True).drop_duplicates(subset="SMILES").head(10))
    return out_df


def run_10_cycles():
    """ A testing function for the paper
    :return: None
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in range(0, 10):
        input_dict['results_filename'] = f"ts_result_{i:03d}.csv"
        run_ts(input_dict, hide_progress=False)


def compare_iterations():
    """ A testing function for the paper
    :return:
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in (2, 5, 10, 50, 100):
        num_ts_iterations = i * 1000
        input_dict["num_ts_iterations"] = num_ts_iterations
        input_dict["results_filename"] = f"iteration_test_{i}K.csv"
        run_ts(input_dict)


def main():
    start = timer()
    json_filename = sys.argv[1]
    input_dict = read_input(json_filename)
    run_ts(input_dict)
    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()
