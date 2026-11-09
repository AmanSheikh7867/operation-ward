"""
Tests for database resilience, retry buffering, and local fallback.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch

from backend.database.supabase_client import persist, _db_write
from backend.state import S


@pytest.mark.asyncio
class TestDatabaseFallback:
    async def test_persist_buffers_on_db_down(self):
        # Force db_client to None
        original_client = S.db_client
        S.db_client = None
        S.retry_buffer.clear()

        row = {"patient_id": "bed01", "hr": 78, "spo2": 98.0, "risk": 0.1}
        await persist("vitals", row, op="insert")

        # Must park row in retry buffer
        assert len(S.retry_buffer) == 1
        table, buffered_row, op = S.retry_buffer[0]
        assert table == "vitals"
        assert buffered_row == row
        assert op == "insert"
        assert S.db_connected is False

        S.db_client = original_client

    async def test_db_write_raises_when_no_client(self):
        original_client = S.db_client
        S.db_client = None

        with pytest.raises(ConnectionError):
            _db_write("vitals", {"dummy": 1})

        S.db_client = original_client

    async def test_buffer_preserves_order(self):
        S.db_client = None
        S.retry_buffer.clear()

        for i in range(5):
            await persist("vitals", {"val": i}, op="insert")

        assert len(S.retry_buffer) == 5
        values = [row["val"] for table, row, op in S.retry_buffer]
        assert values == [0, 1, 2, 3, 4]

    async def test_successful_write_clears_error(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = MagicMock()
        mock_client.table.return_value = mock_table

        original_client = S.db_client
        S.db_client = mock_client
        S.last_error = "Old error"

        await persist("vitals", {"patient_id": "test"}, op="insert")
        assert S.last_error == ""
        assert S.db_connected is True

        S.db_client = original_client
