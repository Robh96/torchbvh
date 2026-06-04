import torchbvh


def _split2(x: int) -> int:
    x &= 0x0000FFFF
    out = 0
    for bit in range(16):
        out |= ((x >> bit) & 1) << (2 * bit)
    return out


def _split3(x: int) -> int:
    x &= 0x000003FF
    out = 0
    for bit in range(10):
        out |= ((x >> bit) & 1) << (3 * bit)
    return out


def _encode2_int(x: int, y: int) -> int:
    return _split2(x) | (_split2(y) << 1)


def _encode3_int(x: int, y: int, z: int) -> int:
    return _split3(x) | (_split3(y) << 1) | (_split3(z) << 2)


def test_morton_split2_known_small_values():
    expected = {
        0: 0b000000,
        1: 0b000001,
        2: 0b000100,
        3: 0b000101,
        4: 0b010000,
        5: 0b010001,
    }

    for x, code in expected.items():
        assert torchbvh.morton_split2(x) == code

    assert torchbvh.morton_split2(0x1FFFF) == _split2(0xFFFF)


def test_morton_split3_known_small_values():
    expected = {
        0: 0b000000000,
        1: 0b000000001,
        2: 0b000001000,
        3: 0b000001001,
        4: 0b001000000,
        5: 0b001000001,
    }

    for x, code in expected.items():
        assert torchbvh.morton_split3(x) == code

    assert torchbvh.morton_split3(0x7FF) == _split3(0x3FF)


def test_morton_encode_2d_known_small_integer_values():
    scene_min = (0.0, 0.0)
    scene_max = (65536.0, 65536.0)

    assert torchbvh.morton_encode_2d(0.0, 0.0, scene_min, scene_max) == 0
    assert torchbvh.morton_encode_2d(1.0, 0.0, scene_min, scene_max) == 1
    assert torchbvh.morton_encode_2d(0.0, 1.0, scene_min, scene_max) == 2
    assert torchbvh.morton_encode_2d(1.0, 1.0, scene_min, scene_max) == 3
    assert torchbvh.morton_encode_2d(2.0, 1.0, scene_min, scene_max) == _encode2_int(2, 1)


def test_morton_encode_3d_known_small_integer_values():
    scene_min = (0.0, 0.0, 0.0)
    scene_max = (1024.0, 1024.0, 1024.0)

    assert torchbvh.morton_encode_3d(0.0, 0.0, 0.0, scene_min, scene_max) == 0
    assert torchbvh.morton_encode_3d(1.0, 0.0, 0.0, scene_min, scene_max) == 1
    assert torchbvh.morton_encode_3d(0.0, 1.0, 0.0, scene_min, scene_max) == 2
    assert torchbvh.morton_encode_3d(0.0, 0.0, 1.0, scene_min, scene_max) == 4
    assert torchbvh.morton_encode_3d(2.0, 1.0, 3.0, scene_min, scene_max) == _encode3_int(2, 1, 3)


def test_morton_encode_2d_normalizes_to_unit_interval():
    code = torchbvh.morton_encode_2d(0.5, 0.25, (0.0, 0.0), (1.0, 1.0))
    assert code == _encode2_int(32768, 16384)

    shifted = torchbvh.morton_encode_2d(5.0, 12.5, (4.0, 10.0), (6.0, 20.0))
    assert shifted == _encode2_int(32768, 16384)


def test_morton_encode_3d_normalizes_to_unit_interval():
    code = torchbvh.morton_encode_3d(
        0.5,
        0.25,
        0.75,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )
    assert code == _encode3_int(512, 256, 768)

    shifted = torchbvh.morton_encode_3d(
        5.0,
        12.5,
        37.5,
        (4.0, 10.0, 30.0),
        (6.0, 20.0, 40.0),
    )
    assert shifted == _encode3_int(512, 256, 768)


def test_morton_encode_clamps_outside_scene_bounds():
    assert torchbvh.morton_encode_2d(
        -1.0,
        2.0,
        (0.0, 0.0),
        (1.0, 1.0),
    ) == torchbvh.morton_encode_2d(0.0, 1.0, (0.0, 0.0), (1.0, 1.0))

    assert torchbvh.morton_encode_3d(
        -1.0,
        0.5,
        2.0,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ) == torchbvh.morton_encode_3d(0.0, 0.5, 1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_duplicate_coordinates_produce_duplicate_morton_codes():
    scene_min = (-1.0, -2.0, -3.0)
    scene_max = (1.0, 2.0, 3.0)

    first = torchbvh.morton_encode_3d(0.25, -0.5, 1.25, scene_min, scene_max)
    second = torchbvh.morton_encode_3d(0.25, -0.5, 1.25, scene_min, scene_max)

    assert first == second


def test_degenerate_scene_extents_normalize_axis_to_zero():
    assert torchbvh.morton_encode_2d(
        5.0,
        0.5,
        (5.0, 0.0),
        (5.0, 1.0),
    ) == _encode2_int(0, 32768)

    assert torchbvh.morton_encode_3d(
        5.0,
        0.5,
        9.0,
        (5.0, 0.0, 9.0),
        (5.0, 1.0, 9.0),
    ) == _encode3_int(0, 512, 0)
