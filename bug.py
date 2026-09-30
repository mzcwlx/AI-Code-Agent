def calculate_discount(price, discount):
    return price * (1 - discount)

def calculate_total(price, discount):
    return calculate_discount(price, discount)