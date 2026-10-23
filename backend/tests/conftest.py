import pytest

from app.services import answer_service


@pytest.fixture(autouse=True)
def reset_llm_cooldown():
    """Keep the process-wide provider cooldown from leaking between tests."""
    answer_service._reset_llm_cooldown()
    yield
    answer_service._reset_llm_cooldown()
