import pytest
from pathlib import Path
import tempfile
from khors.tools.registry import ToolContext
from khors.tools.files import _read_file

def test_read_file_wrapper():
    """Убеждаемся, что чтение файла оборачивает содержимое в теги <file_content> для изоляции."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_wrap.txt"
        test_file.write_text("Secret Token: 123", encoding="utf-8")
        
        ctx = ToolContext(repo_dir=temp_path, drive_root=temp_path)
        
        # Test full read
        res_full = _read_file(ctx, "test_wrap.txt")
        assert "<file_content path=\"test_wrap.txt\">" in res_full
        assert "Secret Token: 123" in res_full
        assert "</file_content>" in res_full
        
        # Test line range read
        test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")
        res_lines = _read_file(ctx, "test_wrap.txt", start_line=2, end_line=3)
        assert "<file_content path=\"test_wrap.txt\">" in res_lines
        assert "Line 2\nLine 3" in res_lines
        assert "</file_content>" in res_lines
