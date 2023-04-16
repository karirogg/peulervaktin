def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

lcm_val = 1
for i in range(1, 21):
    lcm_val = lcm(lcm_val, i)
print(lcm_val)
