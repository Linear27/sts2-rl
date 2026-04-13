from __future__ import annotations


def allocate_ports(first_port: int, instance_count: int) -> list[int]:
    if first_port <= 0:
        raise ValueError("first_port must be positive")
    if instance_count <= 0:
        raise ValueError("instance_count must be positive")
    return [first_port + offset for offset in range(instance_count)]
