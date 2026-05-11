from evaluation import get_evaluation_run_output_folder


def test_single_evaluation_run_uses_run_1_subfolder():
    assert get_evaluation_run_output_folder("miplib_nl_eval", 1) == "miplib_nl_eval/run_1"


def test_multiple_evaluation_runs_use_same_subfolder_convention():
    assert get_evaluation_run_output_folder("miplib_nl_eval", 1) == "miplib_nl_eval/run_1"
    assert get_evaluation_run_output_folder("miplib_nl_eval", 3) == "miplib_nl_eval/run_3"
