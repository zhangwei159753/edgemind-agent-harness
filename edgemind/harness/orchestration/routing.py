import re


_ACTIONS = ("搜索", "查询", "检索", "比较", "核验", "总结", "分析", "整理", "生成")
_SEQUENCE_PATTERNS = (
    re.compile(r"先.{1,100}(?:再|然后|接着).{1,100}"),
    re.compile(r"(?:然后|接着|依次)"),
)
_CONNECTOR = re.compile(r"并且|以及|并")


def should_orchestrate(task_objective: str, user_input: str, mode: str = "auto") -> bool:
    """Select a multi-agent route only for explicit multi-stage requests."""
    if mode not in {"auto", "single", "multi"}:
        raise ValueError("mode must be auto, single, or multi")
    if mode == "single":
        return False
    if mode == "multi":
        return True
    if not isinstance(task_objective, str) or not isinstance(user_input, str):
        raise ValueError("task objective and user input must be text")
    text = f"{task_objective}\n{user_input}"[:40000]
    if any(pattern.search(text) for pattern in _SEQUENCE_PATTERNS):
        return True
    segments = _CONNECTOR.split(text)
    for left, right in zip(segments, segments[1:]):
        if any(action in left for action in _ACTIONS) and any(action in right for action in _ACTIONS):
            return True
    objective_actions = {action for action in _ACTIONS if action in task_objective}
    input_actions = {action for action in _ACTIONS if action in user_input}
    if objective_actions and input_actions and objective_actions != input_actions:
        return True
    return False
