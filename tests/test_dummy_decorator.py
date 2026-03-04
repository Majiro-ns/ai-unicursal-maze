from __future__ import annotations

import numpy as np

from backend.core.graph_builder import skeleton_to_graph
from backend.core.decorator import generate_dummy_branches, DummyOptions
from backend.core.path_finder import PathPoint


def test_generate_dummy_branches_does_not_crash_on_simple_line() -> None:
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[2, 1] = True
    skeleton[2, 2] = True
    skeleton[2, 3] = True

    graph = skeleton_to_graph(skeleton)
    main_path = [PathPoint(x=float(x), y=2.0) for x in (1, 2, 3)]

    branches = generate_dummy_branches(graph, main_path, DummyOptions(max_branches=3, max_branch_length=3))

    assert isinstance(branches, list)
    for branch in branches:
        assert isinstance(branch, list)
