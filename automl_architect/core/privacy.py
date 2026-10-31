"""Minimize raw dataset values sent to an external reasoning provider."""

from copy import copy


def prompt_profile(profile, include_values=False):
    if profile is None or include_values:
        return profile
    result = profile.model_copy(deep=True)
    for column in result.columns:
        column.top_values = []
    if result.target:
        result.target.class_counts = [
            item.model_copy(update={"value": f"class_{i}"})
            for i, item in enumerate(result.target.class_counts)
        ]
    return result


def prompt_state(state):
    view = copy(state)
    view.profile = prompt_profile(state.profile, state.settings.include_sample_values_in_prompts)
    return view
