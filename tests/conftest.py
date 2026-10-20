import torch


def pytest_configure(config):
    torch.set_num_threads(1)


def assert_verified(results):
    """Every result of a verification step is ``exact`` or ``close``."""
    failed = [str(r) for r in results if r.status == "FAIL"]
    assert not failed, "\n".join(failed)
    assert results, "the check returned no result"
