def comp(a, b):
    if a == b:
        return 0
    if a[-4] == '.':
        a = a[:-4]
        b = b[:-4]
    elif a[-3] == '.':
        a = a[:-3]
        b = b[:-3]
    num_a = ''
    num_b = ''
    for s in reversed(a):
        if not s.isdecimal():
            break
        num_a = s + num_a
    for s in reversed(b):
        if not s.isdecimal():
            break
        num_b = s + num_b
    return (int(num_a) > int(num_b)) * 2 - 1

