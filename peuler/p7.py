def is_prime(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

count = 0
n = 2
while True:
    if is_prime(n):
        count += 1
        if count == 10001:
            print(n)
            break
    n += 1
