from pyfolds.telemetry.buffer import RingBufferThreadSafe


def test_buffer_insert_overflow_wraparound():
    b = RingBufferThreadSafe[int](capacity=3)
    assert b.push(1)
    assert b.push(2)
    assert b.push(3)
    assert b.snapshot() == [1, 2, 3]
    assert b.push(4)
    assert b.snapshot() == [2, 3, 4]
    assert b.dropped_events_count() == 1


def test_buffer_snapshot_and_drain():
    b = RingBufferThreadSafe[int](capacity=4)
    for i in range(4):
        b.push(i)
    assert b.snapshot(max_events=2) == [2, 3]
    assert b.drain(max_events=2) == [0, 1]
    assert b.size() == 2
