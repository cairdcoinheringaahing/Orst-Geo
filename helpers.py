import itertools
import math
import random

builtin_exec = exec
builtin_max = max
builtin_min = min
builtin_range = range
builtin_zip = zip

def all_equal(array):
    eq = array.pop()
    for elem in array:
        if eq != elem:
            return False
    return True

def append(array, elem):
    array.append(elem)
    return array

def append_write(text, filename):
    with open(filename, mode = 'a', encoding = 'utf-8') as file:
        file.write(text)
    return text

def choice(array):
    return random.choice(list(array))

def decrement(value):
    if isinstance(value, (list, str)):
        return value[0]
    return value - 1

def deduplicate(array):
    unique = []
    for element in array:
        if element not in unique:
            unique.append(element)
    return unique

def deduplicate_predicate(func, array):
    results = list(map(func, array))
    unique = []
    for ded, elem in builtin_zip(results, array):
        if results.count(ded) == 1:
            unique.append(elem)
    return unique

def duplicate(value):
    return (value, value)

def exec(function, *args):
    return function(*args)

def extend(array, ext):
    array.extend(ext)
    return array

def factors(integer, prime = False, proper = False):
    flist = []
    for i in builtin_range(1, integer + 1):
        if integer % i == 0:
            flist.append(i)

    if prime:
        flist = list(filter(isprime, flist))
    if proper:
        flist = flist[1:-1]
        
    return flist

def flatten(array):
    flat = []
    for elem in array:
        if isinstance(elem, list):
            flat += flatten(elem)
        else:
            flat.append(elem)
    return flat

def from_base(digits, base = 10):
    total = 0
    for index, elem in enumerate(digits[::-1]):
        total += elem * base ** index
    return total

def gcd(x, y):
    if x and not y:
        return x
    if y and not x:
        return y
    
    x = prime_product(x)
    y = prime_product(y)
    
    union = []
    
    for element in x:
        if element in y:
            union.append(element)
            y.remove(element)
            
    return product(union)

def groupby(func, array):
    groups = {}
    results = list(map(func, array))
    for i, value in enumerate(array):
        if results[i] not in list(groups.keys()):
            groups[results[i]] = [value]
        else:
            groups[results[i]] += [value]
    return list(map(lambda a: a[-1], sorted(groups.items(), key = lambda a: a[0])))

def ifnot_statement(opt, *branches):
    return if_statement(not opt, *branches)

def if_statement(opt, *branches):
    if len(branches) == 2:
        if opt: ret = branches[0].call()
        else:   ret = branches[1].call()
    else:
        if opt: ret = branches[0].call()
        else:   ret = None
        
    return ret

def increment(value):
    if isinstance(value, (list, str)):
        return value[-1]
    return value + 1

def increments(array):
    forwards = []
    for left, right in builtin_zip(array, array[1:]):
        forwards.append(right - left)
    return forwards

def insert(array, index, elem):
    array.insert(index, elem)
    return array

def isprime(integer):
    for i in builtin_range(2, integer):
        if integer % i == 0:
            return False
    return integer > 1 and isinstance(integer, int)

def listify(function, multi = False):
    def inner(*args):
        if multi:
            return list(map(list, function(*args)))
        return list(function(*args))
    return inner

def max(func, iterable):
    return builtin_max(iterable, key = func)

def min(func, iterable):
    return builtin_min(iterable, key = func)

def assign(variables, var, val):
    variables[var] = val

def nfind(func, amount, tail = False, head = False):
    found = []
    index = 1
    
    while len(found) < amount:
        if func(index):
            found.append(index)
        index += 1
        
    if tail:
        return found[-1]
    if head:
        return found[0]
    return found

def prime_product(value):
    prime = 2
    parts = []
    while value > 1:
        if value % prime == 0:
            value //= prime
            parts.append(prime)
        else:
            prime += 1
    return parts

def product(array):
    total = 1
    for elem in array:
        total *= elem
    return total

def range(*args):
    if len(args) == 1:
        arg = args[0]
        if arg < 0:
            sign = -1
        elif arg == 0:
            return []
        else:
            sign = 1
            
        i = 1
        while i <= abs(arg):
            yield i * sign
            i += 1

    elif len(args) == 2:
        start, stop = sorted(args)
        while start <= stop:
            yield start
            start += 1

    else:
        start, stop, step = args
        start, stop = sorted([start, stop])
        while start <= stop:
            yield start
            start += step

def read(filename):
    with open(filename, encoding = 'utf-8') as file:
        contents = file.read()
    return contents

def reduce(func, array, accumulate = False):
    ret = [array.pop(0)]
    while array:
        ret.append(func(ret[-1], array.pop(0)))
        
    if accumulate:
        return ret
    return ret.pop()

def repeat(func, iters):
    for _ in builtin_range(iters):
        func.call()

def rle(array):
    counts = []
    last = None
    for elem in array:
        if elem == last:
            counts[-1][1] += 1
        else:
            counts.append([elem, 1])
            last = elem
    return counts

def setitem(array, index, value):
    array[index] = value
    return array.copy()

def shuffle(array):
    if type(array) == str:
        array = list(array)
        random.shuffle(array)
        return ''.join(array)
    
    random.shuffle(array)
    return array

def sort(array, key = None, reverse = False):
    if key is None:
        return sorted(array, reverse = reverse)
    return sorted(array, key = key, reverse = reverse)

def tail(array, take = 1):
    if take == 1:
        return array[-1]
    return array[-take:]

def to_base(integer, base = 10):
    digits = []
    sign = (integer > 0) - (integer < 0)
    integer = abs(integer)
    
    while integer:
        integer, rem = divmod(integer, base)
        digits.append(rem)
        
    return list(map(lambda a: sign * a, digits[::-1]))

def totient(integer):
    count = 0
    for i in builtin_range(integer):
        if gcd(integer, i) == 1:
            count += 1
    return count

def wrap(elem):
    return [elem]

def write(text, filename):
    with open(filename, mode = 'w', encoding = 'utf-8') as file:
        file.write(text)
    return text

def zip(array, left = None):
    if left is None:
        zipped = itertools.zip_longest(*array)
    else:
        zipped = itertools.zip_longest(array, left)
        
    nones = itertools.cycle([None])
    return map(listify(filter), nones, zipped)















