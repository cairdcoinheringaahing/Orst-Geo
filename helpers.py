import copy
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

def apply_even(func, array):
    indexes = [2 * i + 1 for i in builtin_range(len(array) // 2)]
    return sparse(func, indexes, array)

def apply_odd(func, array):
    indexes = [2 * i for i in builtin_range(len(array) // 2)]
    return sparse(func, indexes, array)

def assign(variables, var, val):
    variables[var] = val

def behead(array):
    return array[1:]

def bounce(array):
    return array[:-1] + array[::-1]

def choice(array):
    return random.choice(list(array))

def chunks_of_n(array, n):
    for i in builtin_range(1, len(array) + n, n):
        i -= 1
        yield array[i: i+n]

def contiguous_sublists(array):
    array = list(array)
    ret = []

    for width in builtin_range(1, len(array) + 1):
        ret.extend(windowed_list(array, width))

    return ret

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

def depth(array):
    if hasattr(array, '__iter__'):
        return 1 + builtin_max(map(depth, array))
    return 0

def derangements(array):
    for perm in itertools.permutations(array):
        if is_derangement(perm, array):
            yield list(perm)

def difference(right, left):
    ret = []
    for elem in left:
        if elem not in right:
            ret.append(elem)
    return ret

def duplicate(value):
    if isinstance(value, list):
        return (value.copy(), value.copy())
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

def find_predicate(pred, array, left = None, retall = False, find = 'elem'):
    results = []

    if left is None:
        for index, elem in enumerate(array):
            if pred(elem):
                results.append((index, elem))
    else:
        for index, elem in enumerate(array):
            if pred(elem, left):
                results.append((index, elem))

    if find == 'elem':
        results = list(map(tail, results))
    if find == 'index':
        results = list(map(head, results))

    if retall:
        return results
    return results[0]

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

def from_below(y, x):
    return (x, y, x)

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

def grade_up(array):
    return map(head, sorted(enumerate(array, 1), key = tail))

def groupby(func, array):
    groups = {}
    results = list(map(func, array))
    for i, value in enumerate(array):
        if results[i] not in list(groups.keys()):
            groups[results[i]] = [value]
        else:
            groups[results[i]] += [value]
    return list(map(lambda a: a[-1], sorted(groups.items(), key = lambda a: a[0])))

def group_equal(array):
    chunks = []
    temp = [array.pop(0)]
    for elem in array:
        if elem in temp:
            temp.append(elem)
        else:
            chunks.append(temp)
            temp = [elem]
            
    if temp:
        chunks.append(temp)
    return chunks

def head(array, take = 1):
    if take == 1:
        return array[0]
    return array[:take]

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

def intersection(array, left):
    return list(set(array) & set(left))

def invariant(func, value, left = None):
    if left is not None:
        return func(value, left) == value
    return func(value) == value

def is_derangement(permutation, array):
    for x, y in builtin_zip(permutation, array):
        if x == y:
            return False
    return sorted(permutation) == sorted(array)

def isprime(integer):
    for i in builtin_range(2, integer):
        if integer % i == 0:
            return False
    return integer > 1 and isinstance(integer, int)

def is_sorted(array):
    array = list(array)
    sort = sorted(array)
    rev = sorted(array, reverse = True)
    return sort == array or rev == array

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

def nchunks(array, n):
    step, left = divmod(len(array), n)
    for _ in builtin_range(n):
        ret = []
        while len(ret) < step:
            ret.append(array.pop(0))
            
        if left:
            ret.append(array.pop(0))
            left -= 1
        yield ret

def neighbours(func, array, dyad = False):
    array = windowed_list(array, 2)
    for pair in array:
        if dyad:
            yield func(*pair)
        else:
            yield func(pair)

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

def nrepeat(array, repeats, wrap = False, inplace = False):
    if wrap:
        array = [array]
        
    if inplace:
        ret = []
        for elem in array:
            ret.extend([elem] * repeats)
        return ret
    
    return array * repeats

def nth_elements(array, n):
    return array[::n] if n else array + array[::-1]

def pair(x, y):
    return [y, x]

def partitions(array):
    string = isinstance(array, str)
    array = list(array)
    
    if len(array) == 0:
        return [[]]
    if len(array) == 1:
        return [[array]]
    
    ret = []
    for i in builtin_range(1, len(array)+1):
        for part in partitions(array[i:]):
            ret.append([array[:i]] + part)

    if string:
        return [list(map(''.join, i))for i in ret]
    return ret
		    
def pop(array, index = -1):
    array = array[:index] + array[index+1:]
    return array.copy()

def powerset(s):
    x = len(s)
    result = []
    for i in builtin_range(1 << x):
        result.append([s[j] for j in builtin_range(x) if (i & (1 << j))])
    return result

def prefix(array):
    for i in builtin_range(len(array)):
        yield array[:i + 1]

def prefix_predicate(func, array):
    return list(map(func, prefix(array)))

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

def repeat(func, iters, accumulate = False):
    rets = [func() for _ in builtin_range(iters)]
    if accumulate:
        return rets
    return rets[-1]

def rld(array):
    reconstructed = []
    for count, elem in array:
        for _ in builtin_range(count):
            reconstructed.append(elem)
    return reconstructed

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

def shorten(array):
    return array[:-1]

def shuffle(array):
    if type(array) == str:
        array = list(array)
        random.shuffle(array)
        return ''.join(array)
    
    random.shuffle(array)
    return array

def sign(n):
    return n and int(math.copysign(1, n))

def sort(array, key = None, reverse = False):
    if key is None:
        return sorted(array, reverse = reverse)
    return sorted(array, key = key, reverse = reverse)

def sparse(func, indexes, array, left = None, useindex = False):
    for index in indexes:
            
        if left is None:
            if useindex:
                array[index] = func(array[index], index)
            else:
                array[index] = func(array[index])
        else:
            array[index] = func(arg, left)
            
    return array

def subfactorial(n):
    sum_n = 0
    for i in builtin_range(n + 1):
        sum_n += (-1) ** i / math.factorial(i)
    return int(math.factorial(n) * sum_n)

def suffix(array):
    for i in builtin_range(len(array)):
        yield array[~i:]

def suffix_predicate(func, array):
    return list(map(func, suffix(array)))

def table(func, left, right):
    final = []
    right = list(right)
    for element in left:
        final.append([func(element, i) for i in right])
    return final

def tail(array, take = 1):
    if take == 1:
        return array[-1]
    return array[-take:]

def tie(*args):
    *funcs, array = args
    funcs = itertools.cycle(funcs)
    return list(map(lambda a, b: [a, b, a(b)], funcs, array))

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

def union(array, left):
    return list(set(array) | set(left))

# Execute `body` until `condition` is True
def until_loop(body, condition, accumulate = False):
    found = []
    while not condition():
        ret = body()
        found.append(ret)
        newcopy = copy.deepcopy(body.proc.stacks)
        condition.proc.stacks = newcopy
    body.proc.stacks[body.proc.index].pop()
    
    if accumulate:
        return found
    if found:
        return found[-1]

# Execute `func` while the result is different from the previous result
def until_repeated(func, value, accumulate = False):
    found = [value]
    while True:
        last = found[-1]
        value = func(value)
        if last == value:
            break
        found.append(value)

    if accumulate:
        return found
    return value

# Execute `body` while `condition` is True
def while_loop(body, condition, accumulate = False, do = False):
    found = []
    if do:
        ret = body()
        found.append(ret)
        
    while condition():
        ret = body()
        found.append(ret)
        newcopy = copy.deepcopy(body.proc.stacks)
        condition.proc.stacks = newcopy
    body.proc.stacks[body.proc.index].pop()
    
    if accumulate:
        return found
    if found:
        return found[-1]

# Execute `func` while all the results are identical
def while_same(func, value, accumulate = False):
    found = [value]
    while True:
        ret = func(value)
        if ret not in found:
            break
        found.append(ret)

    if accumulate:
        return found
    return ret

# Execute `func` while all the results are unique
def while_unique(func, value, accumulate = False):
    found = [value]
    while True:
        ret = func(value)
        if ret in found:
            break
        found.append(ret)

    if accumulate:
        return found
    return ret

def windowed_list(array, window):
    subs = []
    for i in builtin_range(len(array) - window + 1):
        subs.append(array[i : i + window])
    return subs

def wrap(elem):
    return [elem]

def write(text, filename):
    with open(filename, mode = 'w', encoding = 'utf-8') as file:
        file.write(text)
    return text

def zip(array, left = None, filler = None):
    if left is None:
        zipped = itertools.zip_longest(*array, fillvalue = filler)
    else:
        zipped = itertools.zip_longest(array, left, fillvalue = filler)
        
    nones = itertools.cycle([None])
    return map(listify(filter), nones, zipped)

def zipwith(func, left, right):
    for pair in zip(left, right):
        if len(pair) == 1:
            yield pair[0]
        else:
            yield func(*pair)
