from __future__ import annotations

from h3_workbench import device_profile
from h3_workbench.device_profile import DeviceProfile, select_device_profile
from h3_workbench.memory_planner import probe_gpu_memory


def _gpu(index: int, name: str, free_gib: int, uuid: str, capability: str = "8.6") -> DeviceProfile:
    return DeviceProfile(
        index=index,
        name=name,
        uuid=uuid,
        provider="cuda",
        total_bytes=24 * 1024**3,
        free_bytes=free_gib * 1024**3,
        compute_capability=capability,
        supports_fp16=True,
        supports_bf16=True,
        supports_fp8=capability != "8.6",
        supports_cuda_graph=True,
        tier="enthusiast",
    )


def test_device_selection_supports_index_uuid_and_auto() -> None:
    profiles = (_gpu(0, "RTX A", 8, "GPU-a"), _gpu(1, "RTX B", 20, "GPU-b"))

    assert select_device_profile(profiles, "0").uuid == "GPU-a"
    assert select_device_profile(profiles, "GPU-b").index == 1
    assert select_device_profile(profiles, "auto").index == 1


def test_device_selection_rejects_unknown_selector() -> None:
    profiles = (_gpu(0, "RTX A", 8, "GPU-a"),)

    try:
        select_device_profile(profiles, "GPU-missing")
    except ValueError as exc:
        assert "did not match" in str(exc)
    else:
        raise AssertionError("unknown GPU selector must fail")


def test_compute_capability_flags_distinguish_ampere_and_ada() -> None:
    ampere = device_profile._capability_flags("8.6")
    ada = device_profile._capability_flags("8.9")

    assert ampere["supports_bf16"] is True
    assert ampere["supports_fp8"] is False
    assert ada["supports_fp8"] is True


def test_memory_snapshot_uses_selected_gpu(monkeypatch) -> None:
    profiles = (_gpu(0, "RTX A", 8, "GPU-a"), _gpu(1, "RTX B", 20, "GPU-b"))
    monkeypatch.setenv("H3_CUDA_DEVICE", "1")
    monkeypatch.setattr(device_profile, "probe_device_profiles", lambda **_: profiles)
    monkeypatch.setattr("h3_workbench.memory_planner.probe_device_profiles", lambda **_: profiles)
    monkeypatch.setattr("h3_workbench.memory_planner._last_probe_at", 0.0)
    monkeypatch.setattr("h3_workbench.memory_planner._last_probe_snapshot", None)

    snapshot = probe_gpu_memory()

    assert snapshot.index == 1
    assert snapshot.uuid == "GPU-b"
    assert snapshot.device_key == "GPU-b"
    assert snapshot.free_bytes == 20 * 1024**3
