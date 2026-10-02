from insight_copilot.graph import executor_prompt, planner_prompt, synthesizer_prompt, _time_context


def test_followup_filters_are_inherited():
    assert "carry forward" in planner_prompt()
    assert "inherit the previous relevant question's metric" in executor_prompt()
    assert "must not reset the date range" in executor_prompt()


def test_anomalies_are_not_described_as_proof_of_structural_change():
    prompt = synthesizer_prompt().lower()
    assert "not proof of a structural shift" in prompt
    assert "do not infer a" in prompt


def test_forecasts_are_anchored_to_dataset_cutoff():
    prompt = synthesizer_prompt().lower()
    assert "historical dataset cutoff" in prompt
    assert "not imply it describes current or future real-world sales" in prompt
    assert "latest order date in data" in _time_context().lower()
