import pytest

from edgemind.harness.orchestration.routing import should_orchestrate


def test_explicit_route_modes_override_auto_detection():
    assert should_orchestrate("single-step", "先搜索再总结", mode="single") is False
    assert should_orchestrate("single-step", "hello", mode="multi") is True
    with pytest.raises(ValueError, match="mode"):
        should_orchestrate("task", "input", mode="anything")


def test_auto_route_keeps_simple_request_on_single_agent_path():
    assert should_orchestrate("Check current weather", "杭州天气怎么样？") is False
    assert should_orchestrate("Summarize a note", "请总结这段笔记。") is False


def test_auto_route_selects_explicit_sequential_work():
    assert should_orchestrate("", "先搜索资料，然后总结三条结论") is True
    assert should_orchestrate("", "依次核验来源并整理结论") is True


def test_auto_route_requires_multiple_action_verbs_around_conjunction():
    assert should_orchestrate("", "搜索资料并总结核心观点") is True
    assert should_orchestrate("", "我喜欢并收藏这篇文章") is False


def test_auto_route_considers_objective_and_input_together():
    assert should_orchestrate("比较两种方案", "给我分析一下") is True
