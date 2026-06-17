"""Validate the LangGraph wiring compiles and exposes the expected nodes,
without launching a browser or hitting the network."""

from __future__ import annotations

import asyncio
from pathlib import Path

from iag_sim.orchestration.graph import build_graph
from iag_sim.orchestration.resources import Resources


def _fake_resources(tmp_path: Path) -> Resources:
    return Resources(
        settings=None,  # not touched at compile time
        client=None,
        run_dir=tmp_path,
        semaphores={"before": asyncio.Semaphore(1), "after": asyncio.Semaphore(1)},
        harnesses={},
    )


def test_graph_compiles_with_expected_nodes(tmp_path):
    graph = build_graph(_fake_resources(tmp_path), settings=None, run_dir=tmp_path)
    nodes = set(graph.get_graph().nodes)
    assert "worker" in nodes
    assert "aggregate" in nodes
