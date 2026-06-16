from src.evaluate_zones import occupancy_at, score_zone


def test_occupancy_counts_overlapping_visits():
    intervals = [(0.0, 10.0), (5.0, 15.0)]
    assert occupancy_at(intervals, 2.0) == 1
    assert occupancy_at(intervals, 7.0) == 2
    assert occupancy_at(intervals, 20.0) == 0


def test_errors_that_cancel_out_still_show_in_mean_absolute_error():
    truth = {0.0: {"shelf": 1}, 2.0: {"shelf": 0}}
    intervals = [(2.0, 2.0)]  # wrong at both moments: misses the person at 0 s, invents one at 2 s
    score = score_zone("test", "shelf", truth, intervals, step_seconds=2.0)
    assert score.bias == 0
    assert score.mean_absolute_error == 1.0
    assert score.true_person_seconds == score.predicted_person_seconds
