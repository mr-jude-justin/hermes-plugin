from contracts import PlanItems
from harness import collect


def test_unattended(monkeypatch):
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    assert collect.unattended("webhook") and collect.unattended("API_SERVER")
    assert not collect.unattended("discord") and not collect.unattended("")
    monkeypatch.setenv("HERMES_CRON_SESSION", "true")
    assert collect.unattended("discord")


def test_task_features_counts_prior_turns_and_spots_paths():
    history = [{"role": "user"}, {"role": "assistant"}, {"role": "user"}]
    f = collect.task_features(message="fix src/app/main.py\n```x```", model="m", platform="cli",
                              session_id="s", conversation_history=history,
                              recent_tools=[f"t{i}" for i in range(15)])
    assert f.session_turns_so_far == 1 and f.has_paths_or_urls and f.has_code_block
    assert f.recent_tools == [f"t{i}" for i in range(5, 15)]
    plain = collect.task_features(message="hello there", model="m", platform="cli", session_id="s")
    assert plain.session_turns_so_far == 0 and not plain.has_paths_or_urls


def test_tool_path():
    assert collect.tool_path("read_file", {"path": "a.py"}) == "a.py"
    assert collect.tool_path("patch", {"file_path": "b.py"}) == "b.py"
    assert collect.tool_path("terminal", {"path": "a.py"}) is None
    assert collect.tool_path("read_file", "a.py") is None


def test_plan_items_full_and_merge():
    todos = [{"status": "completed"}, {"status": "pending"}, {"status": "pending"}]
    first = collect.plan_items({"todos": todos}, None)
    assert first == PlanItems(total=3, done=1)
    merged = collect.plan_items({"merge": True, "todos": [{"status": "completed"}, {"status": "completed"}]}, first)
    assert merged == PlanItems(total=3, done=2)
    assert collect.plan_items({"other": 1}, first) is first


def test_tool_failed():
    assert collect.tool_failed("Error: no such file")
    assert collect.tool_failed({"error": "boom"})
    assert collect.tool_failed("[cost-forecast] blocked")
    assert not collect.tool_failed("ok, wrote 3 lines")
