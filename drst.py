import custom_types, datetime, itertools, math, random, re, regex, time, sys

CMD_REGEX = r"Kk|(E[^\d ])|([?SmFvW][^\d]|([gro].)|([tU]..)|('[^']+')|[^\da-f ]"
NESTED = [1, 3, 9]

ALPHABETU = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHABETL = 'abcdefghijklmnopqrstuvwxyz'
ALPHABETB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
CONSONANTSU = 'BCDFGHJKLMNPQRSTVWXYZ'
CONSONANTSL = 'bcdfghjklmnpqrstvwxyz'
CONSONANTSB = 'BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz'
VOWELU = 'AEIOU'
VOWELL = 'aeiou'
VOWELB = 'AEIOUaeiou'
VOWELYU = 'AEIOUY'
VOWELYL = 'aeiouy'
VOWELYB = 'AEIOUYaeiouy'
QWERTYU = ['QWERTYUIOP','ASDFGHJKL','ZXCVBNM']
QWERTYL = ['qwertyuiop','asdfghjkl','zxcvbnm']
KEYBOARDML = ['§1234567890-=','qwertyuiop[]',"asdfghjkl;'\\",'`zxcvbnm,./']
KEYBOARDMU = ['±!@#$%^&*()_+','QWERTYUIOP{}','ASDFGHJKL:"|','~ZXCVBNM<>?']
KEYBOARDWU = ['¬!"£$%^&*()_+','QWERTYUIOP{}','ASDFGHJKL:@~','|ZXCVBNM<>?']
KEYBOARDWL = ['`1234567890-=','qwertyuiop[]',"asdfghjkl;'#",'\\zxcvbnm,./']
DIGIT = '0123456789'
HEX = '0123456789abcdef'
QUINE = 'EqO'
WORD  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
PRINTABLE = '''!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'''
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2

def binary(value):
    if isinstance(value, (int, float, bool)):
        return bin(int(value)[2:])
    return ''.join(map(lambda c: bin(ord(c))[2:], value))

def bin_(value):
    if isinstance(value, list):
        return list(map(bin_, value))
    return bin(value)[2:]

def choice(value):
    if isinstance(value, (int, float)):
        return random.randrange(0, value)
    return random.choice(value)

def combinations(n, r):
    return list(itertools.combinations(n, r))

def deltas(value):
    if isinstance(value, str):
        value = list(map(ord, value))
    if isinstance(value, (int, float)):
        value = list(range(abs(int(value))))
    if isinstance(value, list):
        value = list(filter(lambda a: isinstance(a, int), value))
    final = []
    for i in range(1,len(value)):
        final.append(value[i] - value[i-1])
    return final

def divisors(x):
    if isinstance(x, list):
        return list(map(divisors, x))
    return list(filter(lambda a: int(x) % a == 0, range(1,int(x)+1)))

def divmod_(x, y):
    if isinstance(x, list):
        return list(map(lambda a: divmod_(a, y), x))
    if isinstance(y, list):
        return list(map(lambda a: divmod_(x, a), y))
    if isinstance(x, list) and isinstance(y, list):
        return list(map(divmod_, x, y))
    return [x//y, x%y]

def elementwise(left, right, mode):
    left, right = str(left), str(right)
    length = min(len(left), len(right))
    left = left[:length]
    right = right[:length]
    
    SWITCHES = {
        1:lambda lc, rc: chr(ord(lc) + ord(rc)),
        2:lambda lc, rc: chr(ord(lc) - ord(rc)),
        3:lambda lc, rc: chr(ord(lc) * ord(rc)),
        4:lambda lc, rc: chr(ord(lc) // ord(rc)),
        5:lambda lc, rc: chr(min(ord(lc), ord(rc))),
        6:lambda lc, rc: chr(max(ord(lc), ord(rc))),
        
        7:lambda lc, rc: chr(((ord(lc)-32) + (ord(rc)-32)) % 95 + 32),
        8:lambda lc, rc: chr(((ord(lc)-32) * (ord(rc)-32)) % 95 + 32),
        9:lambda lc, rc: chr(((ord(lc)-32) ** (ord(rc)-32)) % 95 + 32),
        
        10:lambda lc, rc: random.choice([lc, rc]),
    }

    try:
        switch = SWITCHES[int(mode)]
    except:
        switch = lambda lc, rc: lc
        
    return ''.join(map(switch, left, right))

def eval_(value):
    if isinstance(value, list):
        return list(map(eval_, value))
    try:
        return float(value)
    except:
        try:
            return int(value)
        except:
            try:
                return eval(value)
            except:
                return str(value)
            
def fib_helper(a=[1,1]):
    a[:]=a[1],sum(a)
    return a

def fib(n):
    if instance(n, list):
        return list(map(fib, n))
    for _ in range(n):a=f()[1]
    return a
           
def flatten(array):
    array = str(array)
    array = '['+array.replace('[','').replace(']','')+']'
    return eval(array)

def hexadecimal(value):
    if isinstance(value, (int, float, bool)):
        return hex(int(value)[2:])
    return ''.join(map(lambda c: hex(ord(c))[2:], value))

def hex_(value):
    if isinstance(value, list):
        return list(map(hex_, value))
    return hex(value)[2:]

def indexes(var, val):
    zipped = zip(str(var), str(val))
    return list(map(lambda a: int(a[0] == a[1]), zipped))

def isfib(n):
    f=lambda n, a=1, b=1: n>a and f(n, b, a+b) or n==a
    return f(n)

def isprime(val):
    i = 0
    if isinstance(val, list):
        return list(map(isprime, val))
    for i in range(2,val):
        if val % i == 0:
            return False
    return i > 1

def length(value):
    try:
        return len(value)
    except:
        return len(str(value))

def print_(*values, sep=' ', end='\n'):
    print(*values, sep=sep, end=str(end), file=open('stdout.txt','a'))
    return sep.join(map(str, values))+end

def permutations(value):
    try:
        return list(itertools.permutations(value))
    except:
        return list(itertools.permutations(str(value)))

def repeat(iterable, repetitions):
    final = []
    for i in range(len(repetitions)):
        final.append(iterable)
    if isinstance(iterable, str):
        return ''.join(final)
    return final

def reverse(value):
    try:
        return value[::-1]
    except:
        return eval(str(value)[::-1])

def set_(string):
    final = []
    for char in string:
        if char not in final:
            final.append(char)
    return final

def shuffle(string):
    random.shuffle(string)
    return string

def swap(stack):
    stack[-1], stack[-2] = stack[-2], stack[-1]

class Stack(list):
    def __getitem__(self, index_slice):
        try:
            return super().__getitem__(index_slice)
        except IndexError:
            return 1

    def init(self):
        self.sort = True
        self.time = time.time()
        
    def flatten(self):
        copy = self.copy()
        copy = flatten(copy)
        self.clear()
        self.push(*copy)
        
    def deduplicate(self):
        copy = self.copy()
        copy = set_(copy)
        self.clear()
        self.push(*copy)
        
    def head(self, index):
        self = self[:index]
        
    def tail(self, index):
        self = self[index:]
        
    def push(self, *values, unsort=False):
        for v in values:
            self.append(v)

    def peek(self,index=-1):
        return self[index]
            
    def pop(self, index=-1):
        try:
            return super().pop(index)
        except:
            return 1

COMMANDS = {

    '!':lambda i,s: 'For loop',
    '"':lambda i,s: 'If statement',
    '#':lambda i,s: 'While loop',
    '$':lambda i,s: None,
    '%':lambda i,s: s.push(s.pop(i) % s.pop()),
    '&':lambda i,s: s.push(s.pop(i) & s.pop()),
    "'":lambda i,s: 'String terminator',
    '(':lambda i,s: s.push(s.pop(i) + 1),
    ')':lambda i,s: s.push(s.pop(i) - 1),
    '*':lambda i,s: s.push(s.pop(i) * s.pop()),
    '+':lambda i,s: s.push(s.pop(i) + s.pop()),
    ',':lambda i,s: s.push(s.pop(i) // s.pop()),
    '-':lambda i,s: s.push(s.pop(i) - s.pop()),
    '.':lambda i,s: s.push(s.pop(i) ** s.pop()),
    '/':lambda i,s: s.push(s.pop(i) / s.pop()),
    
    '0':lambda i,s: s.push(0),
    '1':lambda i,s: s.push(1),
    '2':lambda i,s: s.push(2),
    '3':lambda i,s: s.push(3),
    '4':lambda i,s: s.push(4),
    '5':lambda i,s: s.push(5),
    '6':lambda i,s: s.push(6),
    '7':lambda i,s: s.push(7),
    '8':lambda i,s: s.push(8),
    '9':lambda i,s: s.push(9),
    
    ':':lambda i,s: s.push(s.peek(i)),
    ';':lambda i,s: s.push(s.pop(i)),
    '<':lambda i,s: s.push(s.pop(i) < s.pop()),
    '=':lambda i,s: s.push(s.pop(i) == s.pop()),
    '>':lambda i,s: s.push(s.pop(i) > s.pop()),
    '?':lambda i,s: 'Boolean prefix',
    '@':lambda i,s: swap(s),
    
    'A':lambda i,s: s.push(abs(s.pop(i))),
    'B':lambda i,s: s.push(s.pop(i)%2),
    'C':lambda i,s: s.push(combinations(s.pop(i), s.peek())),
    'D':lambda i,s: s.push(divmod_(s.pop(i), s.pop())),
    'E':lambda i,s: 'Extension prefix',
    'F':lambda i,s: 'Filter prefix',
    'G':lambda i,s: s.push(math.gcd(s.pop(i), s.pop())),
    'H':lambda i,s: s.push(s.pop(i)/2),
    'I':lambda i,s: s.push(indexes(s.pop(i), s.pop())),
    'J':lambda i,s: s.push(list(range(s.pop(i)))),
    'K':lambda i,s: 'Sort determinate',
    'L':lambda i,s: s.push(len(s.pop(i))),
    'M':lambda i,s: s.push(elementwise(s.pop(i), s.pop(), s.pop())),
    'N':lambda i,s: s.push(not s.pop(i)),
    'O':lambda i,s: print_(s.peek(i)),
    'P':lambda i,s: s.push(permutations(s.pop(i))),
    'Q':lambda i,s: s.push(''.join(map(str, set_(s.pop(i))))),
    'R':lambda i,s: s.push(s.pop(i).replace(s.pop(), s.pop())),
    'S':lambda i,s: 'Sort control prefix',
    'T':lambda i,s: s.push(deltas(s.pop(i))),
    'U':lambda i,s: 'Unpacked 2 character literal',
    'V':lambda i,s: s.push(reverse(s.pop(i))),
    'W':lambda i,s: 'Datetime prefix',
    'X':lambda i,s: s.push(choice(s.pop(i))),
    'Y':lambda i,s: s.push('\n'.join(map(str, s.pop(i)))),
    'Z':lambda i,s: s.push(''.join(map(''.join, zip(map(str, s.pop(i)), map(str, s.pop()))))),

    '[':lambda i,s: s.push([s.pop(i)]),
    '\\':lambda i,s: s.push(s.pop(i)//1),
    ']':lambda i,s: s.flatten(),
    '^':lambda i,s: s.push(s.pop(i) ^ s.pop()),
    '_':lambda i,s: print_(*s),
    '`':lambda i,s: s.push(s.pop(i).split(s.pop())),
    
    'a':lambda i,s: s.push(10),
    'b':lambda i,s: s.push(11),
    'c':lambda i,s: s.push(12),
    'd':lambda i,s: s.push(13),
    'e':lambda i,s: s.push(14),
    'f':lambda i,s: s.push(15),
    'g':lambda i,s: 'Case sensitive regex prefix',
    'h':lambda i,s: s.head(i),
    'i':lambda i,s: s.push(int(s.pop(i))),
    'j':lambda i,s: s.push(chr(s.pop(i))),
    'k':lambda i,s: 'Inverse sort determinate',
    'l':lambda i,s: s.push(-i),
    'm':lambda i,s: 'Apply-to-each prefix',
    'n':lambda i,s: s.push('\n'.join(map(str, s))),
    'o':lambda i,s: 'One character literal',
    'p':lambda i,s: s.pop(i),
    'q':lambda i,s: s.push(ord(s.pop(i))),
    'r':lambda i,s: 'Case insensitive regex prefix',
    's':lambda i,s: s.push(shuffle(s.pop(i))),
    't':lambda i,s: 'Two character literal',
    'u':lambda i,s: s.tail(i),
    'v':lambda i,s: 'Vectorise prefix',
    'w':lambda i,s: s[-1].pop(i),
    'x':lambda i,s: s.push(random.randint(s.pop(i), s.pop())),
    'y':lambda i,s: print_('\n'.join(map(str, s))),
    'z':lambda i,s: 'Retrieve input',
    
    '{':lambda i,s: s.push(s.pop(i) <= s.pop()),
    '|':lambda i,s: s.push(s.pop(i) | s.pop()),
    '}':lambda i,s: s.push(s.pop(i) >= s.pop()),
    '~':lambda i,s: s.push(~s.pop(i)),

}

BOOLEANS = {

    'A':lambda s, i: all(s),
    'B':lambda s, i: bool(s.pop(i)),
    'D':lambda s, i: bool(open('stdout.txt').read()),
    'E':lambda s, i: all(s[i] == s[i+1] for i in range(len(s)-1)),
    'F':lambda s, i: isfib(s.pop(i)),
    'I':lambda s, i: isinstance(s.pop(i), int),
    'K':lambda s, i: s.sort,
    'L':lambda s, i: all(c.islower() for c in s.pop(i)),
    'O':lambda s, i: list(s) in [sorted(s), sorted(s,reverse=True)],
    'P':lambda s, i: isprime(s.pop(i)),
    'Q':lambda s, i: sorted(set(s)) == sorted(s),
    'R':lambda s, i: s == s[::-1],
    'S':lambda s, i: isinstance(s.pop(i), str),
    'U':lambda s, i: all(c.isupper() for c in s.pop(i)),
    'X':lambda s, i: isinstance(s.pop(i), list),
    'Z':lambda s, i: s.pop(i) > 0,

}

SORTS = {

    'L':length,
    'M':max,
    'I':min,
    'B':binary,
    'H':hexadecimal,
    'E':eval_,

}

MAPPINGS = {

    'a':abs,
    'b':bin_,
    'c':chr,
    'e':eval_,
    'h':hex_,
    'i':int,
    'j':length,
    'l':list,
    'o':ord,
    'p':print_,
    'q':set_,
    'r':reverse,
    's':str,
    'x':lambda i: int(i, 2),
    'z':sorted,
    
}

FILTERS = {

    'A':lambda s: list(filter(lambda a: a.isalpha(), s)),
    'a':lambda s: list(filter(lambda a: not a.isalpha(), s)),
    'B':lambda s: list(filter(lambda a: bool(a), s)),
    'b':lambda s: list(filter(lambda a: not a, s)),
    'C':lambda s: list(filter(lambda a: a in s[-1], s)),
    'c':lambda s: list(filter(lambda a: a not in s[-1], s)),
    'D':lambda s: list(filter(lambda a: a.isdigit(), s)),
    'd':lambda s: list(filter(lambda a: not a.isdigit(), s)),
    'E':lambda s: list(filter(lambda a: a == s[-1], s)),
    'e':lambda s: list(filter(lambda a: a != s[-1], s)),
    'I':lambda s: list(filter(lambda a: isinstance(a, int), s)),
    'i':lambda s: list(filter(lambda a: not isinstance(a, int), s)),
    'J':lambda s: list(filter(lambda a: length(a) == s[-1], s)),
    'j':lambda s: list(filter(lambda a: length(a) != s[-1], s)),
    'L':lambda s: list(filter(lambda a: isinstance(a, list), s)),
    'l':lambda s: list(filter(lambda a: not isinstance(a, list), s)),
    'O':lambda s: list(filter(lambda a: list(a) in [sorted(a), sorted(a,reverse=True)], s)),
    'o':lambda s: list(filter(lambda a: list(a) not in [sorted(a), sorted(a,reverse=True)], s)),
    'P':lambda s: list(filter(lambda a: isprime(a), s)),
    'p':lambda s: list(filter(lambda a: not isprime(a), s)),
    'S':lambda s: list(filter(lambda a: isinstance(a, str), s)),
    's':lambda s: list(filter(lambda a: not isinstance(a, str), s)),
    'V':lambda s: list(filter(lambda a: a == a[::-1], s)),
    'v':lambda s: list(filter(lambda a: a != a[::-1], s)),
}

REGEX = {

    'c':(3, lambda p, s, c: regex.contains(p, s, c)),
    'f':(3, lambda p, s, c: regex.findall(p, s, c)),
    'g':(3, lambda p, s, c: regex.group(p, s, c)),
    'm':(3, lambda p, s, c: regex.match(p, s, c)),
    'b':(3, lambda p, s, c: regex.start(p, s, c)),
    'p':(3, lambda p, s, c: regex.split(p, s, c)),
    'o':(3, lambda p, s, c: regex.findor(p, s, c)),
    's':(4, lambda p, s, r, c: regex.sub(p, s, r, c)),
    'u':(4, lambda p, s, r, c: regex.unsub(p, s, r, c)),

    'C':(3, lambda p, s, c: flatten(regex.contains(p, s, c))),
    'F':(3, lambda p, s, c: flatten(regex.findall(p, s, c))),
    'G':(3, lambda p, s, c: flatten(regex.group(p, s, c))),
    'M':(3, lambda p, s, c: flatten(regex.match(p, s, c))),
    'B':(3, lambda p, s, c: flatten(regex.start(p, s, c))),
    'P':(3, lambda p, s, c: flatten(regex.split(p, s, c))),
    'O':(3, lambda p, s, c: flatten(regex.findor(p, s, c))),
    'S':(4, lambda p, s, r, c: flatten(regex.sub(p, s, r, c))),
    'U':(4, lambda p, s, r, c: flatten(regex.unsub(p, s, r, c))),
    
}

EXTENSIONS = {
    
    'A':lambda i, s: s.push(s[i]),
    'B':lambda i, s: s.push(int(s.pop(i), s[-1])),
    'C':lambda i, s: s.clear(),
    'D':lambda i, s: s.push(divisors(s.pop(i))),
    'E':lambda i, s: s.push(enumerate(s.pop(i))),
    'F':lambda i, s: s.push(fib(s.pop(i))),
    'I':lambda i, s: s.push(int(s.pop(i))),
    'J':lambda i, s: s.push(''.join(map(str, s.pop(i)))),
    'L':lambda i, s: s.push(str(s.pop(i)).lower()),
    'N':lambda i, s: s.push(str(s.pop(i)).count(str(s.pop()))),
    'O':lambda i, s: print_(end=str(s.peek(i))),
    'Q':lambda i, s: s.deduplicate(),
    'R':lambda i, s: s.reverse(),
    'S':lambda i, s: s.push(sorted(s.pop(i))),
    'T':lambda i, s: s.push(str(s.pop(i)).title()),
    'U':lambda i, s: s.push(str(s.pop(i)).upper()),
    'V':lambda i, s: s.push(eval_(s.pop(i))),
    'W':lambda i, s: s.push(str(s.pop(i)).swapcase()),
    'Y':lambda i, s: exec(str(s.pop(i))),
    '"':lambda i, s: s.push(str(s.pop(i))),
    '[':lambda i, s: s.push(list(s.pop(i))),
    ']':lambda i, s: s.push(flatten(s.pop(i))),
    '.':lambda i, s: s.push(list(s.pop(i)).index(s.pop())),
    '{':lambda i, s: s.push(s, True),
    '}':lambda i, s: repeat(s, s.pop(i)),
    '>':lambda i, s: repeat(s.pop(i), s.pop()),
    '_':lambda i, s: print_(s),
    ';':lambda i, s: s.push(''.join(map(str, s))),
    '|':lambda i, s: s.push(*s.pop(i)),
    
    'a':lambda i, s: s.push(PRINTABLE),
    'b':lambda i, s: s.push(ALPHABETU),
    'c':lambda i, s: s.push(CONSONANTSB),
    'd':lambda i, s: s.push(DIGIT),
    'e':lambda i, s: s.push(E),
    'f':lambda i, s: s.push(ALPHABETL),
    'g':lambda i, s: s.push(ALPHABETB),
    'h':lambda i, s: s.push(HEX),
    'i':lambda i, s: s.push(PHI),
    'j':lambda i, s: s.push(CONSONANTSU),
    'k':lambda i, s: s.push(KEYBOARDWU),
    'l':lambda i, s: s.push(KEYBOARDMU),
    'm':lambda i, s: s.push(CONSONANTSL),
    'n':lambda i, s: s.push(VOWELU),
    'o':lambda i, s: s.push(VOWELL),
    'p':lambda i, s: s.push(PI),
    'q':lambda i, s: s.push(QUINE),
    'r':lambda i, s: s.push(QWERTYU),
    's':lambda i, s: s.push(VOWELYU),
    't':lambda i, s: s.push(VOWELYL),
    'u':lambda i, s: s.push(QWERTYL),
    'v':lambda i, s: s.push(VOWELB),
    'w':lambda i, s: s.push(WORD),
    'x':lambda i, s: s.push(KEYBOARDWL),
    'y':lambda i, s: s.push(VOWELYB),
    'z':lambda i, s: s.push(KEYBOARDML),
    
    ' ':lambda i, s: s.push(100),
    '!':lambda i, s: s.push(128),
    '#':lambda i, s: s.push(256),
    '$':lambda i, s: s.push(512),
    '%':lambda i, s: s.push(1000),
    '&':lambda i, s: s.push(1024),
    "'":lambda i, s: s.push(2048),
    '(':lambda i, s: s.push(4096),
    ')':lambda i, s: s.push(8192),
    '*':lambda i, s: s.push(16384),
    '+':lambda i, s: s.push(32768),
    ',':lambda i, s: s.push(65536),
    '-':lambda i, s: s.push(1234567890),
   
}

DATETIME = {

    'H':lambda s: s.push(datetime.datetime.today().hour),
    'M':lambda s: s.push(datetime.datetime.today().minute),
    'S':lambda s: s.push(datetime.datetime.today().second),
    'd':lambda s: s.push(datetime.datetime.today().day),
    'm':lambda s: s.push(datetime.datetime.today().month),
    'y':lambda s: s.push(datetime.datetime.today().year),
    'R':lambda s: s.push(time.time() - s.time),
    'D':lambda s: s.push(datetime.datetime(s.pop(), s.pop(), s.pop())),
    'N':lambda s: s.push(datetime.datetime.now()),
    'W':lambda s: time.sleep(s.pop()),

}

def stack_sort(stack, command):
    if command.isupper():
        key, reverse = SORTS[command], False
    else:
        command = command.upper()
        key, reverse = SORTS[command], True
    return sorted(stack, key=key, reverse=reverse)

def cond_sort(value):
    return value[0] not in """tUo'!"#$"""

def sort(line):
    y = iter(sorted((w for w in line if cond_sort(w)), key=lambda a: a[0]))
    return [w if not cond_sort(w) else next(y) for w in line]

def parser(code, sort_lines):
    code = code.split('\n')
    for i in range(len(code)):
        group = regex.findor(CMD_REGEX, code[i], 0)
        for j in range(len(group)):
            enum = enumerate(group[j])
            group[j] = list(filter(lambda a: a[1] and a[0] not in NESTED, enum))
            group[j] = list(map(lambda a: a[1], group[j]))
        if sort_lines:
            group = sort(group)
        code[i] = list(map(lambda a: a[0], group))
    return code

def arg(arg_value):
    try:return -int(arg_value, 16)
    except:return -1

def execute_line(code, stack, inputs):
    for char in code:

        if char == 'K':
            stack.sort = True
        elif char == 'k':
            stack.sort = False
        
        elif char.startswith("'"):
            stack.push(char[1:-1])

        elif char.startswith('o'):
            stack.push(char[1])

        elif char.startswith('t'):
            stack.push(char[1:3])

        elif char.startswith('U'):
            stack.push(*char[1:3])
            
        elif char[0] == '?':
            cmd = char[1]
            command = BOOLEANS[cmd.upper()]
            if len(char) < 3:
                result = command(stack, -1)
            else:
                result = command(stack, arg(char[2:]))
            if cmd.islower():
                result = not result
            stack.push(result)
            
        elif char[0] == 'S':
            cmd = char[1]
            stack = Stack(stack_sort(stack, cmd))
            
        elif char[0] == 'm':
            cmd = char[1]
            command = MAPPINGS[cmd.lower()]
            for i in range(len(stack)):
                if cmd.islower():
                    stack[i] = command(stack[i])
                else:
                    stack[i] = [list(map(command, i)) for i in stack[i]]
                
        elif char[0] == 'F':
            cmd = char[1]
            command = FILTERS[cmd]
            stack = Stack(command(stack))

        elif char[0] == 'v':
            cmd = MAPPINGS[char[1]]
            value = stack.pop()
            try:list(value)
            except:value = list(range(abs(int(value))))
            value = list(map(cmd, value))
            stack.push(value)

        elif char[0] == 'E':
            ext = EXTENSIONS[char[1]]
            argument = arg(char[2:])
            ext(argument, stack)

        elif char[0] == 'W':
            dtm = DATETIME[char[1]]
            dtm(stack)

        elif char[0] in 'gr':
            if char[0] == 'r':
                case = re.IGNORECASE
            else:
                case = 0
            args, cmd = REGEX[char[1]]
            pattern = str(stack.pop())
            string = str(stack.pop())
            if args == 4:
                other = stack.pop()
                result = cmd(pattern, string, other, case)
            else:
                result = cmd(pattern, string, case)

            stack.push(result)
            
        else:
            if char[0] == 'z':
                stack.push(inputs)
            else:
                cmd = char[0]
                command = COMMANDS[cmd]
                if len(char) == 1:
                  command(-1, stack)
                else:
                    command(arg(char[1:]), stack)
                
    return stack

def interpreter(code, input_file, argv, stack, flags):
    STDIN = list(map(eval_, map(str.strip,open(input_file, 'r').readlines())))
    argv = list(map(eval_, argv))
    
    stack.init()
    stack.push(*STDIN)
    stack.push(*argv)
    inputs = STDIN.copy()
    inputs.extend(argv)
    
    for line in parser(code, '-c' in flags):
        
        if not line:
            continue
        
        if line[0] == '!':
            x = stack.peek()
            if isinstance(x, (int, bool, float)):
                x = abs(int(x))
            else:
                x = len(x)
            for i in range(x):
                stack = execute_line(line[1:], stack, inputs)
                
        elif line[0] == '"':
            if stack.peek():
                stack = execute_line(line[1:], stack, inputs)
                
        elif line[0] == '#':
            while stack.peek():
                stack = execute_line(line[1:], stack, inputs)
                
        else:
            stack = execute_line(line, stack, inputs)

    if not open('stdout.txt').read():
        if '-s' in flags:
            print_(stack)
        else:
            print_(stack.pop())

def run(program, inputs):
    open('stdout.txt','w').close()
    flags = list(filter(lambda a: str(a)[0] == '-', inputs))
    inputs = list(filter(lambda a: a not in flags, inputs))
    interpreter(program.strip(), 'stdin.txt', inputs, Stack(), flags)

if __name__ == '__main__':

    prog = sys.argv[1]
    argv = sys.argv[2:]
    try:
        prog = open(prog).read()
    except:
        pass
    
    run(prog, argv)
