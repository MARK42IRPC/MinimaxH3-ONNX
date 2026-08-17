import numpy as np

from h3_workbench.jobs import _segment_video_condition
from h3_workbench.profiles import PROFILE_360P_17F


def test_segment_condition_aligns_end_anchor_to_last_retained_frame() -> None:
    start_anchor = np.asarray([7.0], dtype=np.float32).reshape(1, 1, 1, 1, 1)
    end_anchor = np.asarray([11.0], dtype=np.float32).reshape(1, 1, 1, 1, 1)

    condition = _segment_video_condition(
        PROFILE_360P_17F,
        target_frames=17,
        segment=0,
        segment_count=1,
        temporal_mode="segmented",
        frame_anchors={"start": start_anchor, "end": end_anchor},
    )

    assert condition is not None
    assert condition.indices == (0, 4)
    np.testing.assert_array_equal(condition.clean.reshape(-1), [7.0, 11.0])


def test_last_short_segment_anchors_its_first_retained_frame() -> None:
    end_anchor = np.asarray([10.0], dtype=np.float32).reshape(1, 1, 1, 1, 1)

    condition = _segment_video_condition(
        PROFILE_360P_17F,
        target_frames=23,
        segment=1,
        segment_count=2,
        temporal_mode="segmented",
        frame_anchors={"end": end_anchor},
    )

    assert condition is not None
    assert condition.indices == (0,)
    assert condition.clean.item() == 10.0
