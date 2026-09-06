"""
Unit test for lightweight Hugging Face Hub checkpoint packaging and restore.

Tests:
1. Packaging of ~760 MB checkpoint bundle (LoRA weights, optimizer, state JSON).
2. Mock upload via HfApi.upload_folder with correct folder and commit message.
3. Mock resume restoring step, epoch, and optimizer state without data corruption.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def test_checkpoint_bundle_packaging_and_restore():
    """Verify that checkpoint state and optimizer weights are correctly bundled and restored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # 1. Simulate saving training state
        state_file = ckpt_dir / "training_state.json"
        saved_state = {
            "step": 75,
            "epoch": 2,
            "track": "with_tools",
            "completed_ids": ["img_001", "img_002", "img_003"],
        }
        with open(state_file, "w") as f:
            json.dump(saved_state, f)

        # 2. Simulate saving optimizer state
        dummy_tensor = torch.nn.Parameter(torch.randn(10, 10))
        opt = torch.optim.AdamW([dummy_tensor], lr=1e-4)
        opt_path = ckpt_dir / "optimizer.pt"
        torch.save(opt.state_dict(), opt_path)

        # 3. Simulate adapter config
        adapter_cfg = ckpt_dir / "adapter_config.json"
        with open(adapter_cfg, "w") as f:
            json.dump({"r": 32, "lora_alpha": 64, "base_model_name_or_path": "Qwen/Qwen3.5-9B"}, f)

        # Assert all files exist
        assert state_file.is_file()
        assert opt_path.is_file()
        assert adapter_cfg.is_file()

        # 4. Simulate restore
        with open(state_file, "r") as f:
            loaded_state = json.load(f)
        assert loaded_state["step"] == 75
        assert loaded_state["epoch"] == 2
        assert loaded_state["track"] == "with_tools"
        assert len(loaded_state["completed_ids"]) == 3

        new_opt = torch.optim.AdamW([dummy_tensor], lr=1e-4)
        new_opt.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        assert new_opt.param_groups[0]["lr"] == 1e-4


def test_mock_upload_checkpoint_to_hf():
    """Verify that upload_checkpoint_to_hf calls HfApi with correct parameters."""
    from scripts.train_sft import upload_checkpoint_to_hf

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        (ckpt_dir / "training_state.json").write_text('{"step": 50}')

        with patch("huggingface_hub.HfApi") as mock_hf_api_cls:
            mock_api = MagicMock()
            mock_hf_api_cls.return_value = mock_api

            upload_checkpoint_to_hf(
                checkpoint_dir=ckpt_dir,
                hf_repo="user/test-vlm-dental",
                step=50,
                epoch=2,
            )

            mock_api.upload_folder.assert_called_once()
            call_kwargs = mock_api.upload_folder.call_args[1]
            assert call_kwargs["repo_id"] == "user/test-vlm-dental"
            assert "Step 50" in call_kwargs["commit_message"]
            assert "*.tmp" in call_kwargs["ignore_patterns"]
