"""Tests for the SituationStore."""

import pytest

from enigma_reason.domain.signal import Signal
from enigma_reason.store.situation_store import SituationStore

from tests.test_signal import _valid_signal


@pytest.fixture
def store() -> SituationStore:
    return SituationStore()


def _signal(**kw) -> Signal:
    return Signal.model_validate(_valid_signal(**kw))


class TestSituationStore:
    @pytest.mark.asyncio
    async def test_ingest_creates_situation(self, store: SituationStore) -> None:
        sig = _signal()
        situation = await store.ingest(sig)
        assert situation.evidence_count == 1

    @pytest.mark.asyncio
    async def test_same_type_and_entity_groups_into_one_situation(self, store: SituationStore) -> None:
        s1 = _signal(signal_type="intrusion", entity={"kind": "user", "identifier": "alice"})
        s2 = _signal(signal_type="intrusion", entity={"kind": "user", "identifier": "alice"})
        sit1 = await store.ingest(s1)
        sit2 = await store.ingest(s2)
        assert sit1.situation_id == sit2.situation_id
        assert sit1.evidence_count == 2

    @pytest.mark.asyncio
    async def test_different_entity_creates_separate_situations(self, store: SituationStore) -> None:
        s1 = _signal(entity={"kind": "user", "identifier": "alice"})
        s2 = _signal(entity={"kind": "user", "identifier": "bob"})
        sit1 = await store.ingest(s1)
        sit2 = await store.ingest(s2)
        assert sit1.situation_id != sit2.situation_id

    @pytest.mark.asyncio
    async def test_active_count(self, store: SituationStore) -> None:
        await store.ingest(_signal(entity={"kind": "user", "identifier": "a"}))
        await store.ingest(_signal(entity={"kind": "user", "identifier": "b"}))
        assert await store.active_count() == 2

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown(self, store: SituationStore) -> None:
        from uuid import uuid4
        assert await store.get(uuid4()) is None
