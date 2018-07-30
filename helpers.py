import enum
import itertools
import math
import operator
import random

import dictionary

builtin_exec = exec
builtin_max = max
builtin_min = min
builtin_range = range
builtin_zip = zip

changes = {

    0: lambda w: w + ' ',
    1: lambda w: ' ' + w,
    2: lambda w: w.title(),
    3: lambda w: w.upper(),
    4: lambda w: w,

}

def all_equal(array):
    eq = array.pop()
    for elem in array:
        if eq != elem:
            return False
    return True

def append(array, *elems):
    for elem in elems:
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
    return val

def behead(array):
    return array[1:]

def bounce(array):
    return array[:-1] + array[::-1]

def cases(string, switch):
    switch %= 5
    if switch == 0:
        return string.lower()
    if switch == 1:
        return string.upper()
    if switch == 2:
        return string.swapcase()
    if switch == 3:
        return string.title()
    if switch == 4:
        return string.title().swapcase()

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

def cycle(array):
    ret = InfiniteList('', InfGen.cycle(array), repeated = len(array))
    return ret

def cycle_ref(name, array):
    ret = cycle(array)
    InfSeq[name] = ret
    return ret

def decrement(value):
    if isinstance(value, (list, str)):
        return value[0]
    return value - 1

def decompress(ints, codepage):
    # Uses the dictionary of words at:
    # https://github.com/DennisMitchell/jellylanguage/blob/master/jelly/dictionary.py
    if isinstance(ints, list):
        num = from_base(ints, 503)
    else:
        num = ints
        
    ret = []
    while num:
        num, mod = divmod(num, 4)
        
        if mod == 0:
            num, switch = divmod(num, 5)
            num, index = divmod(num, len(dictionary.short))
            word = changes[switch](dictionary.short[index])
            
        if mod == 1:
            num, switch = divmod(num, 5)
            num, index = divmod(num, len(dictionary.long))
            word = changes[switch](dictionary.long[index])
            
        if mod == 2:
            num, index = divmod(num, 95)
            word = chr(index + 32)
            
        if mod == 3:
            num, index = divmod(num, 512)
            word = codepage[index]
            
        ret.append(word)
    return ''.join(ret)

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

def fib(x):
    a = b = 1
    for _ in builtin_range(x - 2):
        a, b = a + b, a
    return a

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
    return results[0] if results else (0 if find == 'elem' else -1)

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

def fibonacci(count, a = 1, b = 1, func = operator.add):
    for _ in builtin_range(count - 2):
        a, b = func(a, b), a
    return a

def generator(block):
    return InfiniteList('', block)

def generator_ref(name, block):
    ret = generator(block)
    InfSeq[name] = ret
    return ret

def grade_up(array):
    return map(head, sorted(enumerate(array, 1), key = tail))

def grid(array):
    if depth(array) == 1:
        return join(' ', array)
    ret = []
    for elem in array:
        elem = list(map(str, elem))
        length = builtin_max(map(len, elem))
        elem = list(map(str.ljust, elem, itertools.repeat(length)))
        ret.append(' '.join(elem))
    return '\n'.join(ret)

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
    rev = sort[::-1]
    return sort == array or rev == array

def join(char, array):
    return char.join(map(str, array))

def keep(*args):
    return args[0]

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

def repeat(func, iters, *args):
    return [func(*args) for _ in builtin_range(iters)]

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
    if not isinstance(indexes, list):
        indexes = [indexes]
        
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
        newcopy = body.proc.stacks.copy()
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
        newcopy = body.proc.stacks.copy()
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

class InfiniteList:

    def __init__(self, name, infinite_function, ordered = True, uniques = None, repeated = False):
        self.name = name
        self.inf = infinite_function
        self.order = ordered
        self.gen = self.inf()
        self.uniques = uniques
        self.repeat = repeated

    def __contains__(self, obj):
        if self.uniques:
            return obj in self.uniques
        if self.repeat:
            return obj in self.take(self.repeat)
        
        for elem in self.inf():

            if elem == obj:
                return True
            
            if elem > obj and self.order == 1:
                return False
            elif elem < obj and self.order == -1:
                return False

            return False

    def __iter__(self):
        return self.inf()

    def __getitem__(self, index):
        return self.take(index)[-1]

    def __next__(self):
        return next(self.gen)

    def __repr__(self):
        if self.repeat:
            return '[{}...]'.format(''.join(map(str, self.take(self.repeat))))
        
        return '[⋮{}...]'.format(self.name)

    __str__ = __repr__

    @property
    def elements(self):
        for elem in self.inf():
            print(elem, end = '')
        return ''

    def index(self, elem):
        index = 0
        if elem not in self:
            return -1

        for gen_elem in self.inf():
            if gen_elem == elem:
                return index
            index += 1

    def take(self, num):
        taken = []
        for elem in self.inf():
            if len(taken) == num:
                return taken
            taken.append(elem)

    def drop(self, num = 1):
        for _ in builtin_range(num): next(self.gen)
        return self

    def reset(self):
        self.gen = self.inf()
        return self
		
class InfGen(enum.Enum):

    def π():
        a = 10000
        c = 2800
        b = d = e = g = 0
        f = [a / 5] * (c + 1)
        while True:
            g = c * 2
            b = c
            d += f[b] * a
            g -= 1
            f[b] = d % g
            d //= g
            g -= 1
            b -= 1

            while b:
                d *= b
                d += f[b] * a
                g -= 1
                f[b] = d % g
                d //= g
                g -= 1
                b -= 1

            digits = '%.4d' % (e+d/a)
            for digit in digits:
                yield int(digit)
            e = d%a

    def τ():
        a = 10000
        c = 2800
        b = d = e = g = 0
        f = [a / 5] * (c + 1)
        while True:
            g = c * 2
            b = c
            d += f[b] * a
            g -= 1
            f[b] = d % g
            d //= g
            g -= 1
            b -= 1

            while b:
                d *= b
                d += f[b] * a
                g -= 1
                f[b] = d % g
                d //= g
                g -= 1
                b -= 1

            digits = '%.4d' % (2*(e+d/a))
            if len(digits) == 5:
                digits = digits[1:]
                
            for digit in digits:
                yield int(digit)
            e = d%a
        

    def φ():
        r = 11
        x = 400
        yield 1
        yield 6
        while True:
            d = 0
            while 20*r*d + d*d < x:
                d += 1
            d -= 1
            yield d
            x = 100* (x - (20*r+d) * d)
            r = 10 * r + d

    def ε():
        n = 2
        yield 2
        yield 7
        yield 1
        while True:
            ret = -10 * math.floor(math.e * 10 ** (n - 2)) + math.floor(math.e * 10 ** (n + 1))
            yield to_base(ret)[-1]
            n += 1

    def powers(base):
        def inner():
            x = 1
            while True:
                yield base ** x
                x += 1
        return inner

    def palindromes(base):
        def inner():
            x = 1
            while True:
                conv = to_base(x, base)
                if conv == conv[::-1]:
                    yield x
                x += 1
        return inner

    def constant(value):
        def inner():
            while True:
                yield value
        return inner

    def cycle(values):
        def inner():
            index = 0
            while True:
                yield values[index]
                index += 1
                index %= len(values)
        return inner

    def triangle():
        x = 1
        index = 1
        while True:
            yield x
            x += index
            index += 1

    def square():
        x = 1
        while True:
            yield x ** 2
            x += 1

    def cube():
        x = 1
        while True:
            yield x ** 3
            x += 1

    def xth_power():
        x = 1
        while True:
            yield x ** x
            x += 1

    def reciprocals():
        index = 1
        while True:
            yield 1 / index
            index += 1

    def fibonacci():
        a = b = 1
        while True:
            yield a
            a, b = b, a+b

    def naturals():
        index = 1
        while True:
            yield index
            index += 1

    def signed_integers():
        index = 1
        while True:
            yield index
            yield -index
            index += 1

    def negative_naturals():
        index = -1
        while True:
            yield index
            index -= 1

    def zahlen():
        index = 0
        while True:
            yield -index
            index += 1
            yield index

    def primes():
        num = 2
        while True:
            if all(num%i for i in builtin_range(2, num)):
                yield num
            num += 1

    def palindrome_primes():
        num = 2
        while True:
            conv = to_base(num)
            if all(num%i for i in builtin_range(2, num)) and conv == conv[::-1]:
                yield num
            num += 1
                
InfSeq = {
    
    'π': InfiniteList('π', InfGen.π, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'τ': InfiniteList('τ', InfGen.τ, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'φ': InfiniteList('φ', InfGen.φ, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    'ε': InfiniteList('ε', InfGen.ε, ordered = 0, uniques = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    
    'Δ': InfiniteList('Δ', InfGen.triangle),
    '²': InfiniteList('²', InfGen.square),
    'Ξ': InfiniteList('Ξ', InfGen.cube),
    'Â': InfiniteList('Â', InfGen.xth_power),
    
    '₂': InfiniteList('₂', InfGen.powers(2)),
    '₃': InfiniteList('₃', InfGen.powers(3)),
    '₅': InfiniteList('₅', InfGen.powers(5)),
    '₇': InfiniteList('₇', InfGen.powers(7)),
    '.': InfiniteList('.', InfGen.powers(0.5), ordered = -1),
    '⁻': InfiniteList('⁻', InfGen.powers(-1), ordered = 0),
    '₀': InfiniteList('₀', InfGen.powers(10)),
    
    'Β': InfiniteList('Β', InfGen.palindromes(2)),
    'β': InfiniteList('β', InfGen.palindromes(10)),
    
    '÷': InfiniteList('÷', InfGen.reciprocals, ordered = -1),
    'F': InfiniteList('F', InfGen.fibonacci),
    'N': InfiniteList('N', InfGen.naturals),
    'S': InfiniteList('S', InfGen.signed_integers, ordered = 0),
    '-': InfiniteList('-', InfGen.negative_naturals, ordered = -1),
    'Z': InfiniteList('Z', InfGen.zahlen, ordered = 0),
    'P': InfiniteList('P', InfGen.primes),
    '₽': InfiniteList('₽', InfGen.palindrome_primes),

    '⁰': InfiniteList('⁰', InfGen.constant(0)),
    '¹': InfiniteList('¹', InfGen.constant(1)),
    '²': InfiniteList('²', InfGen.constant(2)),
    '³': InfiniteList('³', InfGen.constant(3)),
    '⁴': InfiniteList('⁴', InfGen.constant(4)),
    '⁵': InfiniteList('⁵', InfGen.constant(5)),
    '⁶': InfiniteList('⁶', InfGen.constant(6)),
    '⁷': InfiniteList('⁷', InfGen.constant(7)),
    '⁸': InfiniteList('⁸', InfGen.constant(8)),
    '⁹': InfiniteList('⁹', InfGen.constant(9)),
    '⁺': InfiniteList('⁺', InfGen.constant(10)),
    '•': InfiniteList('•', InfGen.constant(0.5)),

    '!': InfiniteList('!', InfGen.cycle([0, 1]), repeated = 2),
    '¡': InfiniteList('¡', InfGen.cycle([1, 0]), repeated = 2),
    'A': InfiniteList('A', InfGen.cycle('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), repeated = 26),
    'a': InfiniteList('a', InfGen.cycle('abcdefghijklmnopqrstuvwxyz'), repeated = 26),
    'ß': InfiniteList('ß', InfGen.cycle('ABCDEFGHJIKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'), repeated = 52),
    'Α': InfiniteList('Α', InfGen.cycle('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'), repeated = 24),
    'Ά': InfiniteList('Ά', InfGen.cycle('ΑΆΒΓΔΕΈΖΗΉΘΙΊΚΛΜΝΞΟΌΠΡΣΤΥΎΦΧΨΩΏ'), repeated = 31),
    'α': InfiniteList('α', InfGen.cycle('αβγδεζηθικλμνξοπσςτυφχψω'), repeated = 24),
    'ά': InfiniteList('ά', InfGen.cycle('αάβγδεέζηήθιίΐκλμνξοόπσςτυύΰφχψωώ'), repeated = 33),
    'ť': InfiniteList('ť', InfGen.cycle('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπσςτυφχψω'), repeated = 48),
    'ŧ': InfiniteList('ŧ', InfGen.cycle('ΑΆΒΓΔΕΈΖΗΉΘΙΊΚΛΜΝΞΟΌΠΡΣΤΥΎΦΧΨΩΏαάβγδεέζηήθιίΐκλμνξοόπσςτυύΰφχψωώ'), repeated = 64),
    'Æ': InfiniteList('Æ', InfGen.cycle('ZYXWVUTSRQPONMLKJIHGFEDCBA'), repeated = 26),
    'æ': InfiniteList('æ', InfGen.cycle('zyxwvutsrqponmlkjihgfedcba'), repeated = 26),
    'Å': InfiniteList('Å', InfGen.cycle('zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA'), repeated = 52),

}
