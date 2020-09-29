hello = 'hello'
world = 'world'
hw = hello + ' ' + world
print(hw)

print(len(hw))

hw2 = '%s %s %d' % (hello, world, 2)
print(hw2)

print(hw.capitalize())
print(hw.upper())
print(hw.center(15))
print(hw.rjust(15))
print(hw.replace('l', '(ell)'))
print(' hello world  '.strip())
# list=>python array
animals = ['monkey', 'dog', 'cat']


def printAnima(animals):
    for index, animal in enumerate(animals):
        print('%d %s' % (index, animal))


printAnima(animals)

# 列表推导

a = range(10)
b = [x ** 2 for x in a]
c = [x ** 2 for x in a if x % 2 == 0]

print(a, b, c)

animals_dict = {'cat': 'cute', 'dog': 'furry'}
print('cat' in animals_dict)
print(animals_dict.get('fish', 'N/A'))
animals_dict['fish'] = 'wet'
del animals_dict['fish']


def printAnimaDict(dict):
    for key in dict:
        feature = dict[key]
        print('%s is %s' % (key, animals_dict[key]))


printAnimaDict(animals_dict)

# 字典推导

dict = {x: x ** 2 for x in range(5) if x % 2 == 1}
print(dict)

set = {'cat', 'dog'}
set.add('monkey')
print(len(set))
set.add('monkey')
print(len(set))
set.remove('cat')
print(len(set))

set = {x for x in range(5) if x > 0}
print(set)

# 元组与列表区别在于元组可作为字典中的键和集合中的元素

dict = {(x, x + 2): x for x in range(10) if x > 0}
print(dict)
print(dict[(5, 7)])  # 5


# 类与函数

class Greeter:
    # constructor
    def __init__(self, name):
        self.name = name

    def greeting(self, loud=False):
        if loud:
            print('Hello %s' % self.name.upper())
        else:
            print('Hello %s' % self.name)


g = Greeter('fred')
g.greeting(True)
g.greeting(False)
