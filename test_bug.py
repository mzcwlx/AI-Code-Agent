from bug import calculate_discount, calculate_total


def test_calculate_discount():
    assert calculate_discount(100, 0.2) == 80


def test_calculate_total():
    assert calculate_total(100, 0.2) == 80