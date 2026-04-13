from sts2_rl.runtime.ports import allocate_ports


def test_allocate_ports_returns_contiguous_ports() -> None:
    assert allocate_ports(8080, 4) == [8080, 8081, 8082, 8083]
