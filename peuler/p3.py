number = 600851475143
i = 2
while i * i <= number:
    if number % i == 0:
        number //= i
    else:
        i += 1
print(number)
