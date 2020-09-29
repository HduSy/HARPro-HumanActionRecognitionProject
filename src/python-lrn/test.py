def func(*args):
    print(args)


func(1, 2, 3)


def func(**args):
    print(args)


func(a=1, b=2, c=3)

print(*(1, 2, 3))

a, *b = 'long'
print(a)
print(b)
((a, b), c) = ('lo', 'ng')
print(a, b, c)
