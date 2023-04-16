def is_palindrome(n):
    return str(n) == str(n)[::-1]

largest_palindrome = 0
for i in range(100, 1000):
    for j in range(i, 1000):
        product = i * j
        if is_palindrome(product) and product > largest_palindrome:
            largest_palindrome = product
print(largest_palindrome)
