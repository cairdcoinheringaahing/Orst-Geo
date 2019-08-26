import argparse
import cmath
import datetime
import functools
import itertools
import math
import operator
import os
import random
import re
import sys

import helpers

code_page = '''
AÀÁÂÄÆÃÅĀĄBCĆČÇD
ĎÐEÈÉÊËĒĖĚĘFGHIÌ
ÍÎÏĪĮJKLĹĽŁMNŃŇÑ
ŊOÒÓÔÖŒÕØŌPQRŔŘS
ŚŠTŤŦUÙÚÛÜŮŪVWŴX
YÝŶŸZŹŽŻaàáâäæãå
āąbcćčçdďðeèéêëē
ėěęfghiìíîïīįjkl
ĺľłmnńňñŋoòóôöœø
ōõpqrŕřsßśštťŧuù
úûüůūvwŵxyýŷÿzźž
żΑΆΒΓΔΕΈΖΗΉΘΙΊΚΛ
ΜΝΞΟΌΠΡΣΤΥΎΦΧΨΩΏ
αάβγδεέζηήθιίΐκλ
μνξοόπσςτυύΰφχψω
ώ0123456789. ",
ᏣᏤᏥᏦᏧᏨᏔᏖᏘᎧᎿᏀᏍᏜᏮᏭ
ᏬᏫᏩᏪᏴᏳᏲᏱᏰᏯᎦᎨᎩᎪᎫᎬ
ᎭᎮᎯᎰᎱᎲᎳᎴᎵᎶᎷᎸᎹᎺᎻᎼ
ᎽᏂᎾᏁᏃᏄᏅᏋᏊᏈᏆᏇᏉᏒᏑᏏ
ᏌᏎᏐᏛᏚᏙᏗᏓᏕᏢᏡᏠᏟᏞᏝᎡ
ᎢᎣᎤᎥ₽¥£$¢€₩[&…§]
ЉЊЕРТЗУИОПШАСДФГ
ХЈКЛЧЋЅЏЦВБНМЂЖљ
њертзуиопшасдфгх
јклчћѕџцвбнмђж¤þ
{@%‰#}(?¿!¡)ªº‹›
°◊•—/\:;”“„'’‘`^
*+=_|~-<≤«·»≥>ᴇ¶
ˋˆ¨´ˇ∞≠†∂ƒ⋮¬≈µﬁﬂ
×⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾
÷₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'''.replace('\n', '')

codepage = code_page
for char in '''”“„'’‘`°"''':
    codepage = codepage.replace(char, '')

DIGITTRANS = str.maketrans('₁₂₃₄₅₆₇₈₉₀ᴇ', '1234567890e')
identity = lambda a: a
variables = {'x': 0}

class Output:
    def __init__(self, string = ''):
        self.out = string

    def write(self, new):
        self.out += str(new)

    def flush(self):
        pass

    def __repr__(self):
        return self.out.strip()

    def output(self):
        print(self, file = STDOUT)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        if args:
            print(*args, sep = '\n', file = sys.stderr)
            sys.exit(1)
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, attr):
        try:
            ret = super().__getattribute__(attr)
            has = True
        except AttributeError:
            has = False
            
        if not has:
            return False
        return ret

class Clean:
    def __init__(self, code, argv, stackcount):
        self.proc = Processor(code[1:], argv, stackcount)

    def call(self, *ignored):
        ret = []
        for stk in self.proc.execute().copy():
            if stk:
                ret.append(stk.pop())

        if len(ret) == 1:
            ret = ret.pop()
        self.reset()
        return ret.pop()

    def reset(self):
        length = len(self.proc.stacks)
        self.proc.stacks = []
        for _ in range(length):
            self.proc.stacks.append(Stack())

    def __str__(self):
        return '(%s)' % self.proc.preserve

    def __repr__(self):
        return '(%s)' % self.proc.preserve

    __call__ = call
    
class Block(Clean):
    def call(self, *args):
        if args:
            self.proc.stacks[self.proc.index].push(*args)
        
        ret = []
        for stk in self.proc.execute().copy():
            if stk:
                ret.append(stk.pop())

        if len(ret) == 1:
            ret = ret.pop()
        self.reset()
        return ret.pop()
        
    def __str__(self):
        return '{%s}' % self.proc.preserve

    def __repr__(self):
        return '{%s}' % self.proc.preserve

    __call__ = call
        
class Processor:
    def __init__(self, code, args, stacks = 1, start = False):
        self.prefixes = '"' + "°“'’”`ªº⋮"
        self.doubles = '„‘'
        self.ends = '"' + "“'’‘”„`"
        
        self.preserve = code
        self.code = tokeniser(code)
        self.stacks = [Stack(starters = args) for _ in range(stacks)]
        self.args = args
        self.index = 0
        self.flag = False
        
        if start and args:
            var_order = ['x', 'y', 'z']
            for index, var in enumerate(var_order):
                variables[var] = args[index % len(args)]

    def __str__(self):
        return '''[
    {}
]'''.format('\n    '.join(map(str, self.stacks)))

    __repr__ = __str__

    def execute(self, out = False):

        if out:
            print(self.code, file = sys.stderr)
            
        acollect = ''
        array = 0
        
        bcollect = ''
        block = 0
        
        ccollect = ''
        clean = 0

        gen = (i for i in range(1))

        '''
        
        ... indicates any value

        ° - Single index        (°a        -> 88)
        “ - Wrapped index       (“a        -> [88])
        ' - Single ordinal      ('a        -> 97)
        ’ - Wrapped char        (’a        -> ['a'])
        ‘ - Two char base 512   (‘ab       -> ...)
        „ - Two char literal    („ab       -> "ab")
        ” - Char literal        (”a        -> 'a')
        ` - Wrapped ordinal     (`a        -> [97])
        

        ° - String separator    ("abc°def" -> ["abc", "def"]
        “ - Compressed string   ("abc“     -> "...")
        ' - Code page indexes   ("abc'     -> [88, 98, 99])
        ’ - Base 510 literal    ("abc’     -> ...)
        ‘ - Ordinal indexes     ("abc‘     -> [97, 98, 99])
        ` - Docstring           ("a`       -> "Arccosine (real → real)"
        „ - Wrapped string      ("abc„     -> ["abc"])
        ” - Char array          ("abc”     -> ['a', 'b', 'c']


        ª - Extended commands
        º - Extended commands
        ⋮ - Sequences
        '''

        for index, char in enumerate(self.code):
                
            if len(char) > 1 or char.isdigit():
                if not (array or block or clean):
                    if char[0] not in (self.prefixes + self.doubles):
                        if '≈' in char:
                            char = '+'.join(char.split('≈')) + 'j'
                        self.stacks[self.index].push(eval(char.translate(DIGITTRANS)))
                        continue

            if char == '{':
                block += 1
            if char == '}':
                block -= 1
                if block == 0:
                    self.stacks[self.index].push(Block(bcollect, self.args, len(self.stacks)))
                    bcollect = ''
                    continue

            if char == '(':
                clean += 1
            if char == ')':
                clean -= 1
                if clean == 0:
                    try: count = self.stacks[self.index].peek()
                    except: count = 1
                    if not isinstance(count, int):
                        count = 1
                    self.stacks[self.index].push(Clean(ccollect, self.args, count))
                    ccollect = ''
                    continue

            if char == '[':
                array += 1
            if char == ']':
                array -= 1
                if array == 0:
                    try:
                        self.stacks[self.index].push(eval(acollect + ']'))
                        acollect = ''
                        continue
                    except:
                        print(self, acollect)
                
            if array:
                acollect += char
                continue
            if block:
                bcollect += char
                continue
            if clean:
                ccollect += char
                continue

            if char in ' \t\n':
                continue

            if char[0] == '"' or (char[0] in 'ªº' and char[1] == '"'):
                if char[0] in 'ªº' and char[1] == '"':
                    extra = char[0]
                    char = char[1:]
                else:
                    extra = False
                    
                char = char[1:]
                if char[-1] in self.ends:
                    *char, end = char
                else:
                    char = list(char)
                    end = '"'

                if extra:
                    end = extra + end
                    
                char = ''.join(char).split('°')
                transform = string_formats[end]
                char = list(map(transform, char))
                if len(char) == 1:
                    char = char.pop()
                self.stacks[self.index].push(char)
                continue

            if char[0] in (self.prefixes + self.doubles) and char[0] not in 'ªº⋮':
                effect = prefix_formats[char[0]]
                char = char[1:]
                self.stacks[self.index].push(effect(char))
                continue

            if char[0] == '⋮' and len(char) == 2:
                self.stacks[self.index].push(helpers.InfSeq[char[1]])
                continue

            if char == '=':
                self.flag ^= 1
                continue

            try:
                cmd = commands[char](self.index, self.stacks)
                new = cmd.new
            except KeyError:
                print('Unknown command: "{}". Stack trace:\n{}'.format(char, repr(self)), file = sys.stderr)
                continue
            
            if cmd.arity >= 0:
                args = self.stacks[self.index].pop(count = cmd.arity, unpack = False)
                try:
                    ret = cmd.call(*args)
                except Exception as e:
                    print('Error when running "{}": {}'.format(char, e), file = sys.stderr)
                    continue
                
            else:
                print('Error when running "{}"'.format(char), file = sys.stderr)
                continue

            self.index = new

            if ret is None or cmd.is_none:
                continue
            
            if type(ret) == type(gen):
                ret = list(ret)

            if cmd.empty:
                self.stacks[self.index].clear()
            
            if cmd.unpack:
                self.stacks[self.index].push(*ret)
            else:
                ret = simplify(ret)
                self.stacks[self.index].push(ret)

        # Garbage collection for uncollected syntax types
        for collect in [acollect, bcollect, ccollect]:
            if collect:
                self.stacks[self.index].push(collect)

        return self.stacks.copy()

    def output(self, sep = '\n', flag = None):
        strings = []
        out = list(filter(None, self.stacks))

        if flag is not None:
            self.flag = flag
        
        for stk in out:
            stk.reverse()
            if len(stk) > 1:
                if self.flag:
                    stk = '\n'.join(map(convert, stk))
                else:
                    stk = convert(stk[0])
                
            elif len(stk) == 1:
                stk = convert(stk[0])
                
            else:
                stk = ''
                
            strings.append(stk)
        
        return sep.join(strings).rstrip().strip('\n')

class Stack:
    def __init__(self, array = None, starters = None, mod = True):
        if starters is None:
            starters = [0]
        self.input = starters
        self.modular = mod
        self.elements = array if array is not None else []

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        final = []
        for elem in self:
            final.append(convert(elem, num = True))
        return str(final)

    def __getitem__(self, index):
        length = len(self)
        
        if not length:
            
            ret = self.input.pop()
            self.input = (self.input[::-1] + [ret])[::-1]
            
            if self.modular:
                return Stack()
            else:
                return ret
            
        index = index % self.__len__()
        return self.elements[index]

    def last(self, value):
        
        final = []
        
        for _ in range(value):
            final.append(self.peek(~_))
            
        return final
        
    def push(self, *values):
        for value in values:
            self.elements.append(value)
        return value

    def peek(self, index=-1):
        return self.elements[index]

    def pop(self, count = 1, indexes = (-1,), unpack = False):
        
        popped = []
        next_in = 0
        
        for _ in range(count):
            
            try:
                popped.append(self.elements.pop(indexes[_] if _ < len(indexes) else -1))
            except IndexError:
                popped.append(self.input[next_in % len(self.input)])
                next_in += 1
                
        if unpack and popped:
            return popped if len(popped) > 1 else popped[0]
        return popped

    def reverse(self):
        self.elements = self.elements[::-1]

    def remove(self, index = -1):
        self.pop(index)
        return None

def convert(value, num = False):
    if not isinstance(value, (int, float, complex, list, str, helpers.InfiniteList)):
        try: value = list(value)
        except: pass
        
    if num:
        return value
    
    return str(value)

def simplify(value, unpack = False):
    if isinstance(value, tuple):
        value = list(value)
        
    if isinstance(value, list):
        #if all(isinstance(i, str) for i in value):
        #    return ''.join(value)
        
        if len(value) == 1 and unpack:
            return simplify(value[0])
        return list(map(simplify, value))

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value

    if isinstance(value, complex):
        if value.imag == 0:
            return simplify(value.real)
        return complex(simplify(value.real), simplify(value.imag))

    if value in (True, False):
        return int(value)

    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)

    return value

def digit(string):
    regex = r'''^(-?\z+)(((\k)|(\e-?)|(\j-?))(?(5)$|(\z+(\k\z+)?)))?$''' \
            .replace(r'\z', r'[\d₀₁₂₃₄₅₆₇₈₉]')                          \
            .replace(r'\e', r'[ᴇ]')                                     \
            .replace(r'\j', r'[≈]')                                     \
            .replace(r'\k', r'[\.•]')
            
    return (not re.search(regex, string)) ^ 1

def group(array, concat = True):
    tkns = ['']
    string = 0
    take = 0
    
    for elem in array:
        if take:
            take -= 1
            tkns[-1] += elem
            
        elif elem == '"':
            string ^= 1
            if string:
                tkns.append('"')
            else:
                tkns[-1] += '"'

        elif elem in "“'’‘”„`°":
            if string:
                string = (elem == "°")
                tkns[-1] += elem
            else:
                take = string_takes[elem]
                tkns.append(elem)
                
        else:
            if string:
                tkns[-1] += elem
            else:
                tkns.append(elem)

    tkns = list(filter(None, tkns))
    final = ['']
    index = 0
    skips = 1
    
    while index < len(tkns):
        
        if concat and digit(tkns[index]):
            while index < len(tkns) and digit(tkns[index]):
                if not digit(final[-1]):
                    final.append('')
                final[-1] += tkns[index]
                index += 1
        elif not concat and digit(tkns[index]):
            final.append(tkns[index])

        if index < len(tkns) and not digit(tkns[index]):
            
            if tkns[index] in 'ªº⋮' and index < len(tkns) - 1:
                tkns[index] += tkns[index+1]
                skips = 2
                
            final.append(tkns[index])
        index += skips
        skips = 1

    return list(filter(lambda a: a != '\n', filter(None, final)))

def tokeniser(string):        
    index = 0
    final = []
    temp = ''
    readstring = False
    
    while index < len(string):
        char = string[index]
        
        if char == '"' or char in "“'’‘”„`":
            readstring ^= 1

        if char in '-1234567890₁₂₃₄₅₆₇₈₉₀' and not readstring:
            if char in '0₀' and string[index + 1] not in '.≈':
                final.append(char)
                index += 1
                continue
            
            while char in '-1234567890₁₂₃₄₅₆₇₈₉₀.ᴇ≈':
                temp += char
                index += 1
                try:
                    char = string[index]
                except IndexError:
                    return group(final + [temp])
            
        if temp:
            final.append(temp)
            temp = ''
            
        final.append(char)
        index += 1
        
    if temp:
        final.append(temp)

    return group(final)

def decode(bytestring):
    decoded = ''
    continue_byte = False
    
    for index, byte in enumerate(bytestring):
        if continue_byte:
            continue_byte = False
            continue
        
        if byte == 0xff:
            continue_byte = True
            byte += bytestring[index + 1] + 1

        try:
            decoded += code_page[byte]
        except:
            raise UnicodeDecodeError('Unknown byte value: {}'.format(byte))
    
    return decoded

def nest(*functions):
    functions = list(functions)
    def inner(*args):
        ret = functions.pop()(*args)
        while functions:
            ret = functions.pop()(ret)
        return ret
    return inner

def reverse(polyad):
    def inner(*args):
        args = list(args)[::-1]
        return polyad(*args)
    return inner

def identify(function):
    def inner(arg):
        function(arg)
        return arg
    return inner

def peaceful(function):
    def inner(*args):
        return args + (function(*args),)
    return inner

def partargs(function, index_args, filler = 0):
    def inner(*args):
        args = list(args)
        for index, arg in index_args.items():
            args.insert(index - 1, arg)
        return function(*args)
    return inner

def runattr(obj, attr):
    def inner(*args):
        getattr(obj, attr)(*args)
    return inner

def dynamic_arity(stacks, current, minimum = 2, default = 2):
    stk = stacks[current]
    if not isinstance(stk.peek(), int) or stk.peek() < 0:
        return default + 1
    
    arity = stk.pop(unpack = True)
    if arity == 0:
        arity = minimum - 1
    return arity + 1

def application(types):
    def outer(function, indexes = None):
        def inner(*args):
            args = list(args)

            for index, arg in enumerate(args):
                perform = indexes is None or index in indexes
                
                for ty in types.keys():
                    if isinstance(ty, type):
                        if isinstance(arg, ty) and perform:
                            arg = types[ty](arg)
                            
                    elif callable(ty):
                        if ty(arg) and perform:
                            arg = types[ty](arg)

                    else:
                        raise ValueError()

                args[index] = arg
                
            return function(*args)
        return inner
    return outer

def iterate(function):
    def inner(*args):
        args = list(args)
        if len(args) == 1:
            arg = args[0]
            if not hasattr(arg, '__iter__'):
                arg = [arg]
            return map(function, arg)

        else:
            iterable = any(map(lambda a: hasattr(a, '__iter__'), args))
            length = max(map(lambda a: hasattr(a, '__iter__') and len(a) or 1, args))
            
            for index in range(len(args)):
                arg = args[index]
                if not hasattr(arg, '__iter__'):
                    arg = [arg] * length
                args[index] = arg
                
            ret = list(map(function, *args))
            if not iterable:
                return ret[0]
            return ret
            
    return inner

def tostack(func, preserve):
    def inner(*stack):
        stack = list(stack)
        if preserve:
            return stack[::-1] + [func(stack)]
        return func(stack)
    return inner
        
def nilad(value):
    def inner():
        return value
    return inner

def exec_orst(code, argv):
    proc = Processor(code, argv)
    ret = proc.execute()
    return ret

def tuple_application(*functions):
    def inner(*args):
        return [func(*args) for func in functions]
    return inner

def inverse(function):
    def inner(value):
        for i in range(value):
            if function(i) >= value:
                return i
    return inner

listify = helpers.listify
partial = functools.partial

digits = application(
    {int: helpers.to_base}
)

ranges = application(
    {int: helpers.range}
)

listargs = application(
    {partargs(hasattr, {2: '__iter__'}): list}
)

def get_help(chars):
    chars = group(chars, False)
    ret = []
    for elem in chars:
        if elem in commands:
            ret.append(commands[elem](0, [Stack([0])]).doc)
            
        elif elem[0] == '⋮' and elem[1] in helpers.InfSeq:
            ret.append('InfiniteList({})'.format(repr(helpers.InfSeq[elem[1]])))
            
        elif elem in full_help:
            ret.append(full_help[elem])
            
        else:
            ret.append('NotImplemented: {}'.format(elem))
            
    return ret if len(ret) > 1 else ret.pop()

full_help = {

    '.': 'Decimal point',
    ' ': 'Space',
    '[': 'Open array',
    ']': 'Close array',
    '{': 'Open block',
    '}': 'Close block',
    '(': 'Open clean',
    ')': 'Close clean',
    'ª': 'Extended command prefix',
    'º': 'Extended command prefix',
    '•': 'Base-511 numeric compression',
    'ᴇ': 'Scientific notation',
    '≈': 'Complex number notation',

}

for c in '0123456789₀₁₂₃₄₅₆₇₈₉':
    full_help[c] = 'Integer digit: ' + c.translate(DIGITTRANS)

string_formats = {

    '“': lambda s: helpers.decompress(list(map(codepage.find, s)), code_page),
    "'": lambda s: list(map(code_page.find, s)),
    '’': lambda s: helpers.from_base(list(map(code_page.find, s)), 503),
    '‘': lambda s: list(map(ord, s)),
    '„': lambda s: [s],
    '”': lambda s: list(s),
    '"': lambda s: s,
    '`': lambda s: get_help(s),
}

string_takes = {

    '°': 1,
    '“': 1,
    "'": 1,
    '’': 1,
    '”': 1,
    '`': 1,

    '„': 2,
    '‘': 2,
}

prefix_formats = {

    '°': lambda s: code_page.find(s),
    '“': lambda s: [code_page.find(s)],
    "'": lambda s: ord(s),
    '’': lambda s: [s],
    '‘': lambda s: helpers.from_base(list(map(code_page.find, s)), 512),
    '„': lambda s: s,
    '”': lambda s: s,
    '`': lambda s: [ord(s)],
}

subs = {

    'À' : '+',
    'F' : '_',
    'E' : '×',
    'G' : '÷',
    'Ė' : '*',
    'Ð' : '%',
    
    'Á' : '&',
    'Ë' : '|',
    'I' : '^',
    'C' : '~',
    'Ê' : '¬',
    'D' : '«',
    'Ě' : '»',
    
    'Æ' : '=',
    'È' : '≠',
    'Ď' : '<',
    'Ç' : '≤',
    'Å' : '≥',
    'Ą' : '>',
    
    'r' : '/',
    'î' : '\\',
    'Ñ' : '€',

}

direct_subs = {

    'ᴇ' : 'e',
    '≈' : 'j',
    '-' : ' ',

}

commands = {

    # 1 byte tokens

    'A':lambda index, stacks: AttrDict(
        doc = 'Absolute value (real → real)',
        call = iterate(abs),
        arity = 1,
        new = index,
    ),

    'À':lambda index, stacks: AttrDict(
        doc = 'Addition (num → num → num)',
        call = iterate(operator.add),
        arity = 2,
        new = index,
    ),

    'Á':lambda index, stacks: AttrDict(
        doc = 'Bitwise AND (int → int → int)',
        call = iterate(operator.and_),
        arity = 2,
        new = index,
    ),

    'Â':lambda index, stacks: AttrDict(
        doc = 'Element in array? (iter → any → bool)',
        call = reverse(operator.contains),
        arity = 2,
        new = index,
    ),

    'Ä':lambda index, stacks: AttrDict(
        doc = 'Count occurences (iter → any → int)',
        call = reverse(operator.countOf),
        arity = 2,
        new = index,
    ),

    'Æ':lambda index, stacks: AttrDict(
        doc = 'Equality (iterative) (any → any → bool)',
        call = iterate(operator.eq),
        arity = 2,
        new = index,
    ),

    'Ã':lambda index, stacks: AttrDict(
        doc = 'Floor division (num → num → int)',
        call = iterate(reverse(operator.floordiv)),
        arity = 2,
        new = index,
    ),

    'Å':lambda index, stacks: AttrDict(
        doc = 'Greater than or equal to (real → real → bool)',
        call = iterate(reverse(operator.ge)),
        arity = 2,
        new = index,
    ),

    'Ā':lambda index, stacks: AttrDict(
        doc = 'Indexing (iter → int → any)',
        call = reverse(operator.getitem),
        arity = 2,
        new = index,
    ),

    'Ą':lambda index, stacks: AttrDict(
        doc = 'Greater than (real → real → bool)',
        call = iterate(reverse(operator.gt)),
        arity = 2,
        new = index,
    ),

    'B':lambda index, stacks: AttrDict(
        doc = 'Index of element (iter → any → int)',
        call = reverse(operator.indexOf),
        arity = 2,
        new = index,
    ),

    'C':lambda index, stacks: AttrDict(
        doc = 'Bitwise NOT (int → int)',
        call = iterate(operator.inv),
        arity = 1,
        new = index,
    ),

    'Ć':lambda index, stacks: AttrDict(
        doc = 'Memory identity equality (any → any → bool)',
        call = iterate(operator.is_),
        arity = 2,
        new = index,
    ),

    'Č':lambda index, stacks: AttrDict(
        doc = 'Memory identity inequality (any → any → bool)',
        call = iterate(operator.is_not),
        arity = 2,
        new = index,
    ),

    'Ç':lambda index, stacks: AttrDict(
        doc = 'Less than or equal to (real → real → bool)',
        call = iterate(reverse(operator.le)),
        arity = 2,
        new = index,
    ),

    'D':lambda index, stacks: AttrDict(
        doc = 'Left bitshift (int → int → int)',
        call = iterate(reverse(operator.lshift)),
        arity = 2,
        new = index,
    ),

    'Ď':lambda index, stacks: AttrDict(
        doc = 'Less than (real → real → bool)',
        call = iterate(reverse(operator.lt)),
        arity = 2,
        new = index,
    ),

    'Ð':lambda index, stacks: AttrDict(
        doc = 'Modulo (num → num → num)',
        call = iterate(reverse(operator.mod)),
        arity = 2,
        new = index,
    ),

    'E':lambda index, stacks: AttrDict(
        doc = 'Multiplication (num → num → num)',
        call = iterate(operator.mul),
        arity = 2,
        new = index,
    ),

    'È':lambda index, stacks: AttrDict(
        doc = 'Inequality (iterative) (any → any → bool)',
        call = iterate(operator.ne),
        arity = 2,
        new = index,
    ),

    'É':lambda index, stacks: AttrDict(
        doc = 'Negate (num → num)',
        call = iterate(operator.neg),
        arity = 1,
        new = index,
    ),

    'Ê':lambda index, stacks: AttrDict(
        doc = 'Logical NOT (any → bool)',
        call = iterate(operator.not_),
        arity = 1,
        new = index,
    ),

    'Ë':lambda index, stacks: AttrDict(
        doc = 'Bitwise OR (int → int → int)',
        call = iterate(operator.or_),
        arity = 2,
        new = index,
    ),

    'Ē':lambda index, stacks: AttrDict(
        doc = 'Positive (num → num)',
        call = iterate(operator.pos),
        arity = 1,
        new = index,
    ),

    'Ė':lambda index, stacks: AttrDict(
        doc = 'Power (num → num → num)',
        call = iterate(reverse(operator.pow)),
        arity = 2,
        new = index,
    ),

    'Ě':lambda index, stacks: AttrDict(
        doc = 'Bitwise right shift (int → int → int)',
        call = iterate(reverse(operator.rshift)),
        arity = 2,
        new = index,
    ),

    'Ę':lambda index, stacks: AttrDict(
        doc = 'Set element of array (array → int → any → array)',
        call = reverse(helpers.setitem),
        arity = 3,
        new = index,
    ),

    'F':lambda index, stacks: AttrDict(
        doc = 'Subtraction (num → num → num)',
        call = iterate(reverse(operator.sub)),
        arity = 2,
        new = index,
    ),

    'G':lambda index, stacks: AttrDict(
        doc = 'True division (num → num → num)',
        call = iterate(reverse(operator.truediv)),
        arity = 2,
        new = index,
    ),

    'H':lambda index, stacks: AttrDict(
        doc = 'Boolean (any → bool)',
        call = iterate(bool),
        arity = 1,
        new = index,
    ),

    'I':lambda index, stacks: AttrDict(
        doc = 'Bitwise XOR (int → int → int)',
        call = iterate(operator.xor),
        arity = 2,
        new = index,
    ),

    'Ì':lambda index, stacks: AttrDict(
        doc = 'All elements are truthy? (iter → bool)',
        call = digits(all),
        arity = 1,
        new = index,
    ),

    'Í':lambda index, stacks: AttrDict(
        doc = 'Any elements are truthy? (iter → bool)',
        call = digits(any),
        arity = 1,
        new = index,
    ),

    'Î':lambda index, stacks: AttrDict(
        doc = 'Convert to binary (int → array)',
        call = iterate(partial(helpers.to_base, base = 2)),
        arity = 1,
        new = index,
    ),

    'Ï':lambda index, stacks: AttrDict(
        doc = 'Convert from code points (array → str)',
        call = nest(''.join, iterate(chr)),
        arity = 1,
        new = index,
    ),

    'Ī':lambda index, stacks: AttrDict(
        doc = 'Convert to complex (real → real → complex)',
        call = iterate(reverse(complex)),
        arity = 2,
        new = index,
    ),

    'Į':lambda index, stacks: AttrDict(
        doc = 'Divmod (num → num → [num, num])',
        call = iterate(reverse(divmod)),
        arity = 2,
        new = index,
    ),

    'J':lambda index, stacks: AttrDict(
        doc = 'Enumerate (array → array)',
        call = digits(enumerate),
        arity = 1,
        new = index,
    ),

    'K':lambda index, stacks: AttrDict(
        doc = 'Evaluate Python code (str → any)',
        call = eval,
        arity = 1,
        new = index,
    ),

    'L':lambda index, stacks: AttrDict(
        doc = 'Execute Python code (str → None)',
        call = exec,
        arity = 1,
        is_none = True,
        new = index,
    ),

    'Ĺ':lambda index, stacks: AttrDict(
        doc = 'Filter on predicate (pred → iter → iter)',
        call = digits(filter),
        arity = 2,
        new = index,
    ),

    'Ľ':lambda index, stacks: AttrDict(
        doc = 'Convert to floating point (num → float)',
        call = iterate(float),
        arity = 1,
        new = index,
    ),

    'Ł':lambda index, stacks: AttrDict(
        doc = 'Convert to base 16 (int → array)',
        call = iterate(partial(helpers.to_base, base = 16)),
        arity = 1,
        new = index,
    ),

    'M':lambda index, stacks: AttrDict(
        doc = 'Identity (any → any)',
        call = identity,
        arity = 1,
        new = index,
    ),

    'N':lambda index, stacks: AttrDict(
        doc = 'Take input',
        call = nest(eval, input),
        arity = 0,
        new = index,
    ),

    'Ń':lambda index, stacks: AttrDict(
        doc = 'Convert to integer (num → int)',
        call = iterate(int),
        arity = 1,
        new = index,
    ),

    'Ň':lambda index, stacks: AttrDict(
        doc = 'Length (iter → int)',
        call = digits(len),
        arity = 1,
        new = index,
    ),

    'Ñ':lambda index, stacks: AttrDict(
        doc = 'Map predicate over iterable (pred → iter → iter)',
        call = digits(map),
        arity = 2,
        new = index,
    ),

    'Ŋ':lambda index, stacks: AttrDict(
        doc = 'Convert to base 8 (int → array)',
        call = iterate(partial(helpers.to_base, base = 8)),
        arity = 1,
        new = index,
    ),

    'O':lambda index, stacks: AttrDict(
        doc = 'Take the maximum of a list (array → real)',
        call = digits(max),
        arity = 1,
        new = index,
    ),

    'Ò':lambda index, stacks: AttrDict(
        doc = 'Take the maximum of two arguments (real → real → real)',
        call = iterate(max),
        arity = 2,
        new = index,
    ),

    'Ó':lambda index, stacks: AttrDict(
        doc = 'Take the maximum, based on a predicate (pred → array → any)',
        call = digits(helpers.max),
        arity = 2,
        new = index,
    ),

    'Ô':lambda index, stacks: AttrDict(
        doc = 'Take the minimum of a list (array → real)',
        call = digits(min),
        arity = 1,
        new = index,
    ),

    'Ö':lambda index, stacks: AttrDict(
        doc = 'Take the minimum of two arguments (real → real → real)',
        call = iterate(min),
        arity = 2,
        new = index,
    ),

    'Œ':lambda index, stacks: AttrDict(
        doc = 'Take the minimum, based on a predicate (pred → array → any)',
        call = digits(helpers.min),
        arity = 2,
        new = index,
    ),

    'Õ':lambda index, stacks: AttrDict(
        doc = 'Convert to a list of ordinals (str → [int])',
        call = nest(partial(simplify, unpack = True), listify(iterate(ord))),
        arity = 1,
        new = index,
    ),

    'Ø':lambda index, stacks: AttrDict(
        doc = 'Power modulo (num → num → num → num)',
        call = iterate(pow),
        arity = 3,
        new = index,
    ),

    'Ō':lambda index, stacks: AttrDict(
        doc = 'Print (any → any)',
        call = identify(print),
        arity = 1,
        new = index,
    ),

    'P':lambda index, stacks: AttrDict(
        doc = 'Print the stack',
        call = partial(print, stacks[index]),
        arity = 0,
        new = index,
    ),

    'Q':lambda index, stacks: AttrDict(
        doc = 'End execution',
        call = quit,
        arity = 0,
        is_none = True,
        new = index,
    ),

    'R':lambda index, stacks: AttrDict(
        doc = 'Monadic range (int → array)',
        call = iterate(helpers.range),
        arity = 1,
        new = index,
    ),

    'Ŕ':lambda index, stacks: AttrDict(
        doc = 'Dyadic range (int → int → array)',
        call = iterate(helpers.range),
        arity = 2,
        new = index,
    ),

    'Ř':lambda index, stacks: AttrDict(
        doc = 'Triadic range (int → int → int → array)',
        call = iterate(helpers.range),
        arity = 3,
        new = index,
    ),

    'S':lambda index, stacks: AttrDict(
        doc = 'Reverse (iter → iter)',
        call = digits(reversed),
        arity = 1,
        new = index,
    ),

    'Ś':lambda index, stacks: AttrDict(
        doc = 'Round to integer (real → int)',
        call = iterate(round),
        arity = 1,
        new = index,
    ),

    'Š':lambda index, stacks: AttrDict(
        doc = 'Round to decimal places (real → int → real)',
        call = iterate(reverse(round)),
        arity = 2,
        new = index,
    ),

    'T':lambda index, stacks: AttrDict(
        doc = 'Read from a file (str → str)',
        call = helpers.read,
        arity = 1,
        new = index,
    ),

    'Ť':lambda index, stacks: AttrDict(
        doc = 'Write to a file (str → str → str)',
        call = helpers.write,
        arity = 2,
        new = index,
    ),

    'Ŧ':lambda index, stacks: AttrDict(
        doc = 'Append to a file (str → str → str)',
        call = helpers.append_write,
        arity = 2,
        new = index,
    ),

    'U':lambda index, stacks: AttrDict(
        doc = 'Deduplicate (iter → array)',
        call = digits(helpers.deduplicate),
        arity = 1,
        new = index,
    ),

    'Ù':lambda index, stacks: AttrDict(
        doc = 'Deduplicate based on a predicate (pred → iter → array)',
        call = digits(helpers.deduplicate_predicate),
        arity = 2,
        new = index,
    ),

    'Ú':lambda index, stacks: AttrDict(
        doc = 'Sort (iter → array)',
        call = digits(helpers.sort),
        arity = 1,
        new = index,
    ),

    'Û':lambda index, stacks: AttrDict(
        doc = 'Sort by key (func → iter → array)',
        call = reverse(digits(helpers.sort)),
        arity = 2,
        new = index,
    ),

    'Ü':lambda index, stacks: AttrDict(
        doc = 'Sort, reversed or not, by key (func → iter → bool → array)',
        call = reverse(digits(helpers.sort)),
        arity = 3,
        new = index,
    ),

    'Ů':lambda index, stacks: AttrDict(
        doc = 'Take the sum of an array (array → num)',
        call = ranges(sum),
        arity = 1,
        new = index,
    ),

    'Ū':lambda index, stacks: AttrDict(
        doc = 'Take the sum of an array (array → num)',
        call = digits(sum),
        arity = 1,
        new = index,
    ),

    'V':lambda index, stacks: AttrDict(
        doc = 'Convert to a string (any → str)',
        call = str,
        arity = 1,
        new = index,
    ),

    'W':lambda index, stacks: AttrDict(
        doc = 'Transpose ([array] → [array])',
        call = digits(helpers.zip),
        arity = 1,
        new = index,
    ),

    'Ŵ':lambda index, stacks: AttrDict(
        doc = 'Interweave two arrays (array → array → array)',
        call = digits(helpers.zip),
        arity = 2,
        new = index,
    ),

    'X':lambda index, stacks: AttrDict(
        doc = 'Yield a random float (int → int → float)',
        call = random.uniform,
        arity = 2,
        new = index,
    ),

    'Y':lambda index, stacks: AttrDict(
        doc = 'Make a random choice (array → any)',
        call = digits(helpers.choice),
        arity = 1,
        new = index,
    ),

    'Ý':lambda index, stacks: AttrDict(
        doc = 'Shuffle an array (array → array)',
        call = digits(helpers.shuffle),
        arity = 1,
        new = index,
    ),

    'Ŷ':lambda index, stacks: AttrDict(
        doc = 'Yield a random integer (int → int → int)',
        call = random.randint,
        arity = 2,
        new = index,
    ),

    'Ÿ':lambda index, stacks: AttrDict(
        doc = 'Yield a random float between 0 and 1',
        call = random.random,
        arity = 0,
        new = index,
    ),

    'Z':lambda index, stacks: AttrDict(
        doc = 'Randomly choose from monadic range (int → int)',
        call = iterate(random.randrange),
        arity = 1,
        new = index,
    ),

    'Ź':lambda index, stacks: AttrDict(
        doc = 'Randomly choose from dyadic range (int → int → int)',
        call = iterate(random.randrange),
        arity = 2,
        new = index,
    ),

    'Ž':lambda index, stacks: AttrDict(
        doc = 'Randomly choose from triadic range (int → int → int → int)',
        call = iterate(random.randrange),
        arity = 3,
        new = index,
    ),

    'Ż':lambda index, stacks: AttrDict(
        doc = 'Take a random sample (array → int → array)',
        call = digits(random.sample, (0,)),
        arity = 2,
        new = index,
    ),

    'a':lambda index, stacks: AttrDict(
        doc = 'Arccosine (real → real)',
        call = iterate(math.acos),
        arity = 1,
        new = index,
    ),

    'à':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arccosine (real → real)',
        call = iterate(math.acosh),
        arity = 1,
        new = index,
    ),

    'á':lambda index, stacks: AttrDict(
        doc = 'Arcsine (real → real)',
        call = iterate(math.asin),
        arity = 1,
        new = index,
    ),

    'â':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arcsine (real → real)',
        call = iterate(math.asinh),
        arity = 1,
        new = index,
    ),

    'ä':lambda index, stacks: AttrDict(
        doc = 'Arctangent (real → real)',
        call = iterate(math.atan),
        arity = 1,
        new = index,
    ),

    'æ':lambda index, stacks: AttrDict(
        doc = 'Dyadic arctangent (real → real → real)',
        call = iterate(math.atan2),
        arity = 2,
        new = index,
    ),

    'ã':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arctangent (real → real)',
        call = iterate(math.atanh),
        arity = 1,
        new = index,
    ),

    'å':lambda index, stacks: AttrDict(
        doc = 'Round up to integer (real → int)',
        call = iterate(math.ceil),
        arity = 1,
        new = index,
    ),

    'ā':lambda index, stacks: AttrDict(
        doc = 'Cosine (real → real)',
        call = iterate(math.cos),
        arity = 1,
        new = index,
    ),

    'ą':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic cosine (real → real)',
        call = iterate(math.cosh),
        arity = 1,
        new = index,
    ),

    'b':lambda index, stacks: AttrDict(
        doc = 'Convert to degrees (real → real)',
        call = iterate(math.degrees),
        arity = 1,
        new = index,
    ),

    'c':lambda index, stacks: AttrDict(
        doc = 'Yield Euler\'s constant',
        call = nilad(math.e),
        arity = 0,
        new = index,
    ),

    'ć':lambda index, stacks: AttrDict(
        doc = 'Raise e to the power (real → real)',
        call = iterate(math.exp),
        arity = 1,
        new = index,
    ),

    'č':lambda index, stacks: AttrDict(
        doc = 'Factorial (int → int)',
        call = iterate(math.factorial),
        arity = 1,
        new = index,
    ),

    'ç':lambda index, stacks: AttrDict(
        doc = 'Round down (real → int)',
        call = iterate(math.floor),
        arity = 1,
        new = index,
    ),

    'd':lambda index, stacks: AttrDict(
        doc = 'Gamma function (real → real)',
        call = iterate(math.gamma),
        arity = 1,
        new = index,
    ),

    'ď':lambda index, stacks: AttrDict(
        doc = 'Calculate the hypotenuse (real → real → real)',
        call = iterate(math.hypot),
        arity = 2,
        new = index,
    ),

    'ð':lambda index, stacks: AttrDict(
        doc = 'Natural logarithm (real → real)',
        call = iterate(math.log),
        arity = 1,
        new = index,
    ),

    'e':lambda index, stacks: AttrDict(
        doc = 'Logarithm with specified base (real → int → real)',
        call = iterate(math.log),
        arity = 2,
        new = index,
    ),

    'è':lambda index, stacks: AttrDict(
        doc = 'Logarithm in base 10 (real → real)',
        call = iterate(partargs(math.log, {2: 10})),
        arity = 1,
        new = index,
    ),

    'é':lambda index, stacks: AttrDict(
        doc = 'Logarithm in base 2 (real → real)',
        call = iterate(partargs(math.log, {2: 2})),
        arity = 1,
        new = index,
    ),

    'ê':lambda index, stacks: AttrDict(
        doc = 'Yield π',
        call = nilad(math.pi),
        arity = 0,
        new = index,
    ),

    'ë':lambda index, stacks: AttrDict(
        doc = 'Convert to radians (real → real)',
        call = iterate(math.radians),
        arity = 1,
        new = index,
    ),

    'ē':lambda index, stacks: AttrDict(
        doc = 'Sine (real → real)',
        call = iterate(math.sin),
        arity = 1,
        new = index,
    ),

    'ė':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic sine (real → real)',
        call = iterate(math.sinh),
        arity = 1,
        new = index,
    ),

    'ě':lambda index, stacks: AttrDict(
        doc = 'Square root (real → real)',
        call = iterate(math.sqrt),
        arity = 1,
        new = index,
    ),

    'ę':lambda index, stacks: AttrDict(
        doc = 'Tangent (real → real)',
        call = iterate(math.tan),
        arity = 1,
        new = index,
    ),

    'f':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic tangent (real → real)',
        call = iterate(math.tanh),
        arity = 1,
        new = index,
    ),

    'g':lambda index, stacks: AttrDict(
        doc = 'Yield τ',
        call = nilad(2 * math.pi),
        arity = 0,
        new = index,
    ),

    'h':lambda index, stacks: AttrDict(
        doc = 'Truncate (real → int)',
        call = iterate(math.trunc),
        arity = 1,
        new = index,
    ),

    'i':lambda index, stacks: AttrDict(
        doc = 'Complex phase (complex → real)',
        call = iterate(cmath.phase),
        arity = 1,
        new = index,
    ),

    'ì':lambda index, stacks: AttrDict(
        doc = 'Convert to polar form (complex → [real, real])',
        call = iterate(cmath.polar),
        arity = 1,
        new = index,
    ),

    'í':lambda index, stacks: AttrDict(
        doc = 'Convert to rectangular form (real → real → complex)',
        call = iterate(cmath.rect),
        arity = 2,
        new = index,
    ),

    'î':lambda index, stacks: AttrDict(
        doc = 'Scan a dyad over an array (func → array → array)',
        call = partial(digits(helpers.reduce), accumulate = True),
        arity = 2,
        new = index,
    ),

    'ï':lambda index, stacks: AttrDict(
        doc = 'Combinations (array → int → array)',
        call = reverse(digits(itertools.combinations, (0,))),
        arity = 2,
        new = index,
    ),

    'ī':lambda index, stacks: AttrDict(
        doc = 'Combinations with replacement (array → int → array)',
        call = reverse(digits(itertools.combinations_with_replacement, (0,))),
        arity = 2,
        new = index,
    ),

    'į':lambda index, stacks: AttrDict(
        doc = 'Drop elements while they return True (func → array → array)',
        call = digits(itertools.dropwhile),
        arity = 2,
        new = index,
    ),

    'j':lambda index, stacks: AttrDict(
        doc = 'Filter on false results (func → array → array)',
        call = digits(itertools.filterfalse),
        arity = 2,
        new = index,
    ),

    'k':lambda index, stacks: AttrDict(
        doc = 'Group elements by a function (func → array → array)',
        call = digits(helpers.groupby),
        arity = 2,
        new = index,
    ),

    'l':lambda index, stacks: AttrDict(
        doc = 'Yield all permutations (array → array)',
        call = digits(itertools.permutations),
        arity = 1,
        new = index,
    ),

    'ĺ':lambda index, stacks: AttrDict(
        doc = 'Yield all permutations of a specified length (array → int → array)',
        call = reverse(digits(itertools.permutations, (0,))),
        arity = 2,
        new = index,
    ),

    'ľ':lambda index, stacks: AttrDict(
        doc = 'Perform the Cartesian product of a single array (array → array)',
        call = digits(itertools.product),
        arity = 1,
        new = index,
    ),

    'ł':lambda index, stacks: AttrDict(
        doc = 'Cartesian product (array → array → [array])',
        call = nest(partial(map, list), digits(itertools.product)),
        arity = 2,
        new = index,
    ),

    'm':lambda index, stacks: AttrDict(
        doc = 'Repeat an array (array → int → array)',
        call = reverse(digits(itertools.repeat, (0,))),
        arity = 2,
        new = index,
    ),

    'n':lambda index, stacks: AttrDict(
        doc = 'Map a function over sublists (func → [array] → array)',
        call = digits(itertools.starmap),
        arity = 2,
        new = index,
    ),

    'ń':lambda index, stacks: AttrDict(
        doc = 'Take elements while they return True (func → array → array)',
        call = digits(itertools.takewhile),
        arity = 2,
        new = index,
    ),

    'ň':lambda index, stacks: AttrDict(
        doc = 'Create a partial function (func → int → ...any → func)',
        call = functools.partial,
        arity = dynamic_arity(stacks, index),
        new = index,
    ),

    'ñ':lambda index, stacks: AttrDict(
        doc = 'Remove the top element',
        call = identity,
        arity = 1,
        is_none = True,
        new = index,
    ),

    'ŋ':lambda index, stacks: AttrDict(
        doc = 'Remove a specified element (int → None)',
        call = stacks[index].remove,
        arity = 1,
        is_none = True,
        new = index,
    ),

    'o':lambda index, stacks: AttrDict(
        doc = 'Run a function (func → ...any → ...any)',
        call = helpers.exec,
        arity = len(stacks[index]),
        new = index,
    ),

    'ò':lambda index, stacks: AttrDict(
        doc = 'Run a niladic function (func → any)',
        call = helpers.exec,
        arity = 1,
        new = index,
    ),

    'ó':lambda index, stacks: AttrDict(
        doc = 'Run a monadic function (func → any → any)',
        call = helpers.exec,
        arity = 2,
        new = index,
    ),

    'ô':lambda index, stacks: AttrDict(
        doc = 'Run a dyadic function (func → any → any → any)',
        call = helpers.exec,
        arity = 3,
        new = index,
    ),

    'ö':lambda index, stacks: AttrDict(
        doc = 'Run a function with a specified number of arguments (func → int → ...any → any)',
        call = helpers.exec,
        arity = dynamic_arity(stacks, index, minimum = 1),
        new = index,
    ),

    'œ':lambda index, stacks: AttrDict(
        doc = 'Take the complex conjugate (complex → complex)',
        call = iterate(complex.conjugate),
        arity = 1,
        new = index,
    ),

    'ø':lambda index, stacks: AttrDict(
        doc = 'Real part (complex → real)',
        call = iterate(partargs(getattr, {2: 'real'})),
        arity = 1,
        new = index,
    ),

    'ō':lambda index, stacks: AttrDict(
        doc = 'Imaginary part (complex → real)',
        call = iterate(partargs(getattr, {2: 'imag'})),
        arity = 1,
        new = index,
    ),

    'õ':lambda index, stacks: AttrDict(
        doc = 'Convert a float to a fraction (float → [int, int])',
        call = iterate(float.as_integer_ratio),
        arity = 1,
        unpack = True,
        new = index,
    ),

    'p':lambda index, stacks: AttrDict(
        doc = 'Check if a float is an integer (float → bool)',
        call = iterate(float.is_integer),
        arity = 1,
        new = index,
    ),

    'q':lambda index, stacks: AttrDict(
        doc = 'Count the number of bits (int → int)',
        call = iterate(int.bit_length),
        arity = 1,
        new = index,
    ),

    'r':lambda index, stacks: AttrDict(
        doc = 'Reduce an array by a dyadic function (func → array → any)',
        call = digits(helpers.reduce),
        arity = 2,
        new = index,
    ),

    'ŕ':lambda index, stacks: AttrDict(
        doc = 'Append an element to an array (array → any → array)',
        call = digits(helpers.append),
        arity = 2,
        new = index,
    ),

    'ř':lambda index, stacks: AttrDict(
        doc = 'Concatenate two arrays (array → array → array)',
        call = digits(helpers.extend),
        arity = 2,
        new = index,
    ),

    's':lambda index, stacks: AttrDict(
        doc = 'Find the first index of an element in the array (array → any → int)',
        call = digits(list.index, (0,)),
        arity = 2,
        new = index,
    ),

    'ß':lambda index, stacks: AttrDict(
        doc = 'Find the first index of an element in a sublist (array → any → int → int → int)',
        call = digits(list.index, (0,)),
        arity = 4,
        new = index,
    ),

    'ś':lambda index, stacks: AttrDict(
        doc = 'Remove the last element of an array (array → array)',
        call = digits(helpers.pop),
        arity = 1,
        new = index,
    ),

    'š':lambda index, stacks: AttrDict(
        doc = 'Remove the element at the specified index (array → int → array)',
        call = reverse(digits(helpers.pop, (0,))),
        arity = 2,
        new = index,
    ),

    't':lambda index, stacks: AttrDict(
        doc = 'Insert an element at the index (array → int → any → array)',
        call = digits(list.insert, (0,)),
        arity = 3,
        new = index,
    ),

    'ť':lambda index, stacks: AttrDict(
        doc = 'Capitalise the first letter of a string (str → str)',
        call = str.capitalize,
        arity = 1,
        new = index,
    ),

    'ŧ':lambda index, stacks: AttrDict(
        doc = 'Surround a string with spaces (str → int → str)',
        call = str.center,
        arity = 2,
        new = index,
    ),

    'u':lambda index, stacks: AttrDict(
        doc = 'Surround a string with a given character (str → int → str → str)',
        call = str.center,
        arity = 3,
        new = index,
    ),

    'ù':lambda index, stacks: AttrDict(
        doc = 'Count occurences of a substring (str → str → int)',
        call = str.count,
        arity = 2,
        new = index,
    ),

    'ú':lambda index, stacks: AttrDict(
        doc = 'Count occurences of a substring in a suffix (str → str → int → int)',
        call = str.count,
        arity = 3,
        new = index,
    ),

    'û':lambda index, stacks: AttrDict(
        doc = 'Count occurences of a substring in a substring (str → str → int → int → int)',
        call = str.count,
        arity = 4,
        new = index,
    ),

    'ü':lambda index, stacks: AttrDict(
        doc = 'Does the string end with the substring? (str → str → bool)',
        call = str.endswith,
        arity = 2,
        new = index,
    ),

    'ů':lambda index, stacks: AttrDict(
        doc = 'Does a suffix end with the substring? (str → str → int → bool)',
        call = str.endswith,
        arity = 3,
        new = index,
    ),

    'ū':lambda index, stacks: AttrDict(
        doc = 'Does a substring end with the substring? (str → str → int → int → bool)',
        call = str.endswith,
        arity = 4,
        new = index,
    ),

    'v':lambda index, stacks: AttrDict(
        doc = 'Return the index of the substring (str → str → int)',
        call = str.find,
        arity = 2,
        new = index,
    ),

    'w':lambda index, stacks: AttrDict(
        doc = 'Return the index of the substring in a suffix (str → str → int → int)',
        call = str.find,
        arity = 3,
        new = index,
    ),

    'ŵ':lambda index, stacks: AttrDict(
        doc = 'Return the index of the substring in a substring (str → str → int → int → int)',
        call = str.find,
        arity = 4,
        new = index,
    ),

    'x':lambda index, stacks: AttrDict(
        doc = 'Is the string alphanumeric? (str → bool)',
        call = str.isalnum,
        arity = 1,
        new = index,
    ),

    'y':lambda index, stacks: AttrDict(
        doc = 'Is the string alphabetical? (str → bool)',
        call = str.isalpha,
        arity = 1,
        new = index,
    ),

    'ý':lambda index, stacks: AttrDict(
        doc = 'Is the string numerical? (str → bool)',
        call = str.isdigit,
        arity = 1,
        new = index,
    ),

    'ŷ':lambda index, stacks: AttrDict(
        doc = 'Is the string lowercase? (str → bool)',
        call = str.islower,
        arity = 1,
        new = index,
    ),

    'ÿ':lambda index, stacks: AttrDict(
        doc = 'Is the string uppercase? (str → bool)',
        call = str.isupper,
        arity = 1,
        new = index,
    ),

    'z':lambda index, stacks: AttrDict(
        doc = 'Join an array on a string (str → array → str)',
        call = helpers.join,
        arity = 2,
        new = index,
    ),

    'ź':lambda index, stacks: AttrDict(
        doc = 'Justify-left with spaces (str → int → str)',
        call = str.ljust,
        arity = 2,
        new = index,
    ),

    'ž':lambda index, stacks: AttrDict(
        doc = 'Justify-left (str → int → str → str)',
        call = str.ljust,
        arity = 3,
        new = index,
    ),

    'ż':lambda index, stacks: AttrDict(
        doc = 'Lowercase (str → str)',
        call = str.lower,
        arity = 1,
        new = index,
    ),

    'Α':lambda index, stacks: AttrDict(
        doc = 'Remove leading whitespace (str → str)',
        call = str.lstrip,
        arity = 1,
        new = index,
    ),

    'Ά':lambda index, stacks: AttrDict(
        doc = 'Remove leading characters (str → str → str)',
        call = str.lstrip,
        arity = 2,
        new = index,
    ),

    'Β':lambda index, stacks: AttrDict(
        doc = 'String replacement (str → str → str → str)',
        call = str.replace,
        arity = 3,
        new = index,
    ),

    'Γ':lambda index, stacks: AttrDict(
        doc = 'Counted string replacement (str → str → str → int → str)',
        call = str.replace,
        arity = 4,
        new = index,
    ),

    'Δ':lambda index, stacks: AttrDict(
        doc = 'Justify-right with spaces (str → int → str)',
        call = str.rjust,
        arity = 2,
        new = index,
    ),

    'Ε':lambda index, stacks: AttrDict(
        doc = 'Justify-right (str → int → str → str)',
        call = str.rjust,
        arity = 3,
        new = index,
    ),

    'Έ':lambda index, stacks: AttrDict(
        doc = 'Remove trailing whitespace (str → str)',
        call = str.rstrip,
        arity = 1,
        new = index,
    ),

    'Ζ':lambda index, stacks: AttrDict(
        doc = 'Remove trailing characters (str → str → str)',
        call = str.rstrip,
        arity = 2,
        new = index,
    ),

    'Η':lambda index, stacks: AttrDict(
        doc = 'Split on whitespace (str → array)',
        call = str.split,
        arity = 1,
        new = index,
    ),

    'Ή':lambda index, stacks: AttrDict(
        doc = 'Split on a substring (str → str → array)',
        call = str.split,
        arity = 2,
        new = index,
    ),

    'Θ':lambda index, stacks: AttrDict(
        doc = 'Does the string start with a substring? (str → str → bool)',
        call = str.startswith,
        arity = 2,
        new = index,
    ),

    'Ι':lambda index, stacks: AttrDict(
        doc = 'Does a suffix start with a substring? (str → str → int → bool)',
        call = str.startswith,
        arity = 3,
        new = index,
    ),

    'Ί':lambda index, stacks: AttrDict(
        doc = 'Does a substring start with the substring? (str → str → int → int → bool)',
        call = str.startswith,
        arity = 4,
        new = index,
    ),

    'Κ':lambda index, stacks: AttrDict(
        doc = 'Remove leading and trailing whitespace (str → str)',
        call = str.strip,
        arity = 1,
        new = index,
    ),

    'Λ':lambda index, stacks: AttrDict(
        doc = 'Remove leading and trailing characters (str → str → str)',
        call = str.strip,
        arity = 2,
        new = index,
    ),

    'Μ':lambda index, stacks: AttrDict(
        doc = 'Uppercase (str → str)',
        call = str.upper,
        arity = 1,
        new = index,
    ),

    'Ν':lambda index, stacks: AttrDict(
        doc = 'Convert to digits (int → array)',
        call = iterate(helpers.to_base),
        arity = 1,
        new = index,
    ),

    'Ξ':lambda index, stacks: AttrDict(
        doc = 'Duplicate (any → any)',
        call = helpers.duplicate,
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ο':lambda index, stacks: AttrDict(
        doc = 'Increment (num → num)',
        call = iterate(helpers.increment),
        arity = 1,
        new = index,
    ),

    'Ό':lambda index, stacks: AttrDict(
        doc = 'Decrement (num → num)',
        call = iterate(helpers.decrement),
        arity = 1,
        new = index,
    ),

    'Π':lambda index, stacks: AttrDict(
        doc = 'Euler\'s totient function (int → int)',
        call = iterate(helpers.totient),
        arity = 1,
        new = index,
    ),

    'Ρ':lambda index, stacks: AttrDict(
        doc = 'Convert from binary (array → int)',
        call = partial(digits(helpers.from_base), base = 2),
        arity = 1,
        new = index,
    ),

    'Σ':lambda index, stacks: AttrDict(
        doc = 'Complement (num → num)',
        call = iterate(partial(operator.sub, 1)),
        arity = 1,
        new = index,
    ),

    'Τ':lambda index, stacks: AttrDict(
        doc = 'Convert from digits (array → int)',
        call = digits(helpers.from_base),
        arity = 1,
        new = index,
    ),

    'Υ':lambda index, stacks: AttrDict(
        doc = 'Convert from base (array → int → int)',
        call = digits(helpers.from_base, (0,)),
        arity = 2,
        new = index,
    ),

    'Ύ':lambda index, stacks: AttrDict(
        doc = 'Are all elements equal? (array → bool)',
        call = digits(helpers.all_equal),
        arity = 1,
        new = index,
    ),

    'Φ':lambda index, stacks: AttrDict(
        doc = 'Flatten ([array] → array)',
        call = helpers.flatten,
        arity = 1,
        new = index,
    ),

    'Χ':lambda index, stacks: AttrDict(
        doc = 'Parity (real → int)',
        call = iterate(partargs(operator.mod, {2: 2})),
        arity = 1,
        new = index,
    ),

    'Ψ':lambda index, stacks: AttrDict(
        doc = 'Forward differences (array → array)',
        call = digits(helpers.increments),
        arity = 1,
        new = index,
    ),

    'Ω':lambda index, stacks: AttrDict(
        doc = 'Prime test (int → bool)',
        call = iterate(helpers.isprime),
        arity = 1,
        new = index,
    ),

    'Ώ':lambda index, stacks: AttrDict(
        doc = 'Prime factor decomposition (int → array)',
        call = iterate(helpers.prime_product),
        arity = 1,
        new = index,
    ),

    'α':lambda index, stacks: AttrDict(
        doc = 'Factors (int → array)',
        call = iterate(helpers.factors),
        arity = 1,
        new = index,
    ),

    'ά':lambda index, stacks: AttrDict(
        doc = 'Proper factors (int → array)',
        call = iterate(partial(helpers.factors, proper = True)),
        arity = 1,
        new = index,
    ),

    'β':lambda index, stacks: AttrDict(
        doc = 'Unique prime factors (int → array)',
        call = iterate(partial(helpers.factors, prime = True)),
        arity = 1,
        new = index,
    ),

    'γ':lambda index, stacks: AttrDict(
        doc = 'Unique proper prime factors (int → array)',
        call = iterate(partial(helpers.factors, prime = True, proper = True)),
        arity = 1,
        new = index,
    ),

    'δ':lambda index, stacks: AttrDict(
        doc = 'Product (array → num)',
        call = digits(helpers.product),
        arity = 1,
        new = index,
    ),

    'ε':lambda index, stacks: AttrDict(
        doc = 'Ternary if statement (bool → func → func → any)',
        call = helpers.if_statement,
        arity = 3,
        new = index,
    ),

    'έ':lambda index, stacks: AttrDict(
        doc = 'Conditional execution (bool → func → any)',
        call = helpers.if_statement,
        arity = 2,
        new = index,
    ),

    'ζ':lambda index, stacks: AttrDict(
        doc = 'Negated conditional execution (bool → func → any)',
        call = helpers.ifnot_statement,
        arity = 2,
        new = index,
    ),

    'η':lambda index, stacks: AttrDict(
        doc = 'Push the input array',
        call = nilad(stacks[index].input),
        arity = 0,
        new = index,
    ),

    'ή':lambda index, stacks: AttrDict(
        doc = 'Push the individual inputs',
        call = nilad(stacks[index].input),
        arity = 0,
        unpack = True,
        new = index,
    ),

    'θ':lambda index, stacks: AttrDict(
        doc = 'Find the first integers which satisfy the given predicate (pred → int → array)',
        call = helpers.nfind,
        arity = 2,
        new = index,
    ),

    'ι':lambda index, stacks: AttrDict(
        doc = 'First the nth integer which satisfies the given predicate (pred → int → int)',
        call = partial(helpers.nfind, tail = True),
        arity = 2,
        new = index,
    ),

    'ί':lambda index, stacks: AttrDict(
        doc = 'Find the first integer which satisfies the given predicate (pred → int → int)',
        call = partargs(helpers.nfind, {2: 1, 4: True}),
        arity = 1,
        new = index,
    ),

    'ΐ':lambda index, stacks: AttrDict(
        doc = 'Push the value of $x',
        call = nilad(variables['x']),
        arity = 0,
        new = index,
    ),

    'κ':lambda index, stacks: AttrDict(
        doc = 'Set the value of $x',
        call = partargs(helpers.assign, {1: variables, 2: 'x'}),
        arity = 1,
        new = index,
    ),

    'λ':lambda index, stacks: AttrDict(
        doc = 'Factor sum (int → int)',
        call = iterate(nest(sum, helpers.factors)),
        arity = 1,
        new = index,
    ),

    'μ':lambda index, stacks: AttrDict(
        doc = 'Retrieve the value of a variable (str → any)',
        call = variables.get,
        arity = 1,
        new = index,
    ),

    'ν':lambda index, stacks: AttrDict(
        doc = 'Assign a variable to a value (str → any → any)',
        call = partargs(helpers.assign, {1: variables}),
        arity = 2,
        new = index,
    ),

    'ξ':lambda index, stacks: AttrDict(
        doc = 'Flipped parity (real → int)',
        call = iterate(nest(partial(operator.sub, 1), partargs(operator.mod, {2: 2}))),
        arity = 1,
        new = index,
    ),

    'ο':lambda index, stacks: AttrDict(
        doc = 'Wrap (any → [any])',
        call = helpers.wrap,
        arity = 1,
        new = index,
    ),

    'ό':lambda index, stacks: AttrDict(
        doc = 'Product (array → int)',
        call = ranges(helpers.product),
        arity = 1,
        new = index,
    ),

    'π':lambda index, stacks: AttrDict(
        doc = '10 to the power, modulo (real → real → real)',
        call = iterate(partial(pow, 10)),
        arity = 2,
        new = index,
    ),

    'σ':lambda index, stacks: AttrDict(
        doc = '2 to the power, modulo (real → real → real)',
        call = iterate(partial(pow, 2)),
        arity = 2,
        new = index,
    ),

    'ς':lambda index, stacks: AttrDict(
        doc = 'Prime factors and exponents (int → [array])',
        call = iterate(nest(helpers.rle, helpers.prime_product)),
        arity = 1,
        new = index,
    ),

    'τ':lambda index, stacks: AttrDict(
        doc = 'Run-length encode (array → [array])',
        call = digits(helpers.rle),
        arity = 1,
        new = index,
    ),

    'υ':lambda index, stacks: AttrDict(
        doc = 'Prime exponents (int → array)',
        call = iterate(nest(partial(map, helpers.tail), helpers.rle, helpers.prime_product)),
        arity = 1,
        new = index,
    ),

    'ύ':lambda index, stacks: AttrDict(
        doc = 'Double (num → num)',
        call = iterate(partial(operator.mul, 2)),
        arity = 1,
        new = index,
    ),

    'ΰ':lambda index, stacks: AttrDict(
        doc = 'Specified number of prime numbers (int → array)',
        call = iterate(partial(helpers.nfind, helpers.isprime)),
        arity = 1,
        new = index,
    ),

    'φ':lambda index, stacks: AttrDict(
        doc = 'Yield φ',
        call = nilad((1 + math.sqrt(5)) / 2),
        arity = 0,
        new = index,
    ),

    'χ':lambda index, stacks: AttrDict(
        doc = 'Zip with function (array → array → func → array)',
        call = helpers.zipwith,
        arity = 3,
        new = index,
    ),

    'ψ':lambda index, stacks: AttrDict(
        doc = 'Outer product (array → array → func → array)',
        call = helpers.table,
        arity = 3,
        new = index,
    ),

    'ω':lambda index, stacks: AttrDict(
        doc = 'Move to next stack (any → any)',
        call = stacks[(index + 1) % len(stacks)].push,
        arity = 1,
        new = (index + 1) % len(stacks),
    ),

    'ώ':lambda index, stacks: AttrDict(
        doc = 'Move to previous stack (any → any)',
        call = stacks[(index - 1) if (index - 1) < 0 else (len(stacks) + ~index)].push,
        arity = 1,
        new = (index - 1) if (index - 1) < 0 else (len(stacks) + ~index),
    ),

    ',':lambda index, stacks: AttrDict(
        doc = 'Pair (any → any → [any, any])',
        call = helpers.pair,
        arity = 2,
        new = index,
    ),

    # Two byte commands

    'Ꮳ':lambda index, stacks: AttrDict(
        doc = 'While loop (pred → func → any)',
        call = helpers.while_loop,
        arity = 2,
        new = index,
    ),

    'Ꮴ':lambda index, stacks: AttrDict(
        doc = 'Collected while loop (pred → func → array)',
        call = partial(helpers.while_loop, accumulate = True),
        arity = 2,
        new = index,
    ),

    'Ꮵ':lambda index, stacks: AttrDict(
        doc = 'Until loop (pred → func → any)',
        call = helpers.until_loop,
        arity = 2,
        new = index,
    ),

    'Ꮶ':lambda index, stacks: AttrDict(
        doc = 'Collected until loop (pred → func → array)',
        call = partial(helpers.until_loop, accumulate = True),
        arity = 2,
        new = index,
    ),

    'Ꮷ':lambda index, stacks: AttrDict(
        doc = 'Loop until two adjacent iterations are equal (pred → any → any)',
        call = helpers.until_repeated,
        arity = 2,
        new = index,
    ),

    'Ꮸ':lambda index, stacks: AttrDict(
        doc = 'Loop until two adjacent iterations are equal, collecting the results (pred → any → array)',
        call = partial(helpers.until_repeated, accumulate = True),
        arity = 2,
        new = index,
    ),

    'Ꮤ':lambda index, stacks: AttrDict(
        doc = 'Loop while the results are unique (pred → any → any)',
        call = helpers.while_unique,
        arity = 2,
        new = index,
    ),

    'Ꮦ':lambda index, stacks: AttrDict(
        doc = 'Loop while the results are unique, collecting the results (pred → any → array)',
        call = partial(helpers.while_unique, accumulate = True),
        arity = 2,
        new = index,
    ),

    'Ꮨ':lambda index, stacks: AttrDict(
        doc = 'Loop with the results are identical (pred → any → any)',
        call = helpers.while_same,
        arity = 2,
        new = index,
    ),

    'Ꭷ':lambda index, stacks: AttrDict(
        doc = 'Loop while the results are identical, collecting the results (pred → any → array)',
        call = partial(helpers.while_same, accumulate = True),
        arity = 2,
        new = index,
    ),

    'Ꮏ':lambda index, stacks: AttrDict(
        doc = 'Find the first element which is truthy under a predicate (pred → array → any)',
        call = helpers.find_predicate,
        arity = 2,
        new = index,
    ),

    'Ꮐ':lambda index, stacks: AttrDict(
        doc = 'Find all elements which are truthy under a predicate (pred → array → array)',
        call = partial(helpers.find_predicate, retall = True),
        arity = 2,
        new = index,
    ),

    'Ꮝ':lambda index, stacks: AttrDict(
        doc = 'Find the first index where the corrosponding element is truthy under a predicate (pred → array → int)',
        call = partial(helpers.find_predicate, find = 'index'),
        arity = 2,
        new = index,
    ),

    'Ꮬ':lambda index, stacks: AttrDict(
        doc = 'Find all indexes where the corrosponding element is truhy under a predicate (pred → array → array)',
        call = partial(helpers.find_predicate, retall = True, find = 'index'),
        arity = 2,
        new = index,
    ),

    'Ꮾ':lambda index, stacks: AttrDict(
        doc = 'Find the first element which is truthy under a dyadic predicate (pred → array → any → any)',
        call = helpers.find_predicate,
        arity = 3,
        new = index,
    ),

    'Ꮽ':lambda index, stacks: AttrDict(
        doc = 'Find all elements which are truthy under a dyadic predicate (pred → array → any → array)',
        call = partial(helpers.find_predicate, retall = True),
        arity = 3,
        new = index,
    ),

    'Ꮼ':lambda index, stacks: AttrDict(
        doc = 'Find the first index where the corrosponding element is truthy under a dyadic predicate (pred → array → any → int)',
        call = partial(helpers.find_predicate, find = 'index'),
        arity = 3,
        new = index,
    ),

    'Ꮻ':lambda index, stacks: AttrDict(
        doc = 'Find all indexes where the corrosponding element is truthy under a dyadic predicate (pred → array → any → array)',
        call = partial(helpers.find_predicate, retall = True, find = 'index'),
        arity = 3,
        new = index,
    ),

    'Ꮹ':lambda index, stacks: AttrDict(
        doc = 'Sparse application (func → [int] → array → array)',
        call = helpers.sparse,
        arity = 3,
        new = index,
    ),

    'Ꮺ':lambda index, stacks: AttrDict(
        doc = 'Dyadic sparse application, using the indexes (func → [int] → array → array)',
        call = partial(helpers.sparse, useindex = True),
        arity = 3,
        new = index,
    ),

    'Ᏼ':lambda index, stacks: AttrDict(
        doc = 'Dyadic sparse application (func → [int] → array → any → array)',
        call = helpers.sparse,
        arity = 4,
        new = index,
    ),

    'Ᏻ':lambda index, stacks: AttrDict(
        doc = 'Execute a function a specfied number of times (func → int → ...any → any)',
        call = helpers.repeat,
        arity = len(stacks[index]),
        new = index,
    ),

    'Ᏺ':lambda index, stacks: AttrDict(
        doc = 'Is a value identical under a function? (func → any → bool)',
        call = helpers.invariant,
        arity = 2,
        new = index,
    ),

    'Ᏹ':lambda index, stacks: AttrDict(
        doc = 'Is a value identical under a dyadic function (func → any → any → bool)',
        call = helpers.invariant,
        arity = 3,
        new = index,
    ),

    'Ᏸ':lambda index, stacks: AttrDict(
        doc = 'Iterate a function over overlapping pairs of an array (func → array → array)',
        call = helpers.neighbours,
        arity = 2,
        new = index,
    ),

    'Ꮿ':lambda index, stacks: AttrDict(
        doc = 'Iterate a dyadic function over overlapping pairs of an array (func → array → array)',
        call = partial(helpers.neighbours, dyad = True),
        arity = 2,
        new = index,
    ),

    'Ꭶ':lambda index, stacks: AttrDict(
        doc = 'Push the entire stack',
        call = nilad(stacks[index]),
        arity = 0,
        empty = True,
        new = index,
    ),

    'Ꭸ':lambda index, stacks: AttrDict(
        doc = 'Yield all prefixes of an array (array → [array])',
        call = digits(helpers.prefix),
        arity = 1,
        new = index,
    ),

    'Ꭹ':lambda index, stacks: AttrDict(
        doc = 'Yield all suffixes of an array (array → [array])',
        call = digits(helpers.suffix),
        arity = 1,
        new = index,
    ),

    'Ꭺ':lambda index, stacks: AttrDict(
        doc = 'Iterate a function over each prefix of an array (func → array → array)',
        call = digits(helpers.prefix_predicate),
        arity = 2,
        new = index,
    ),

    'Ꭻ':lambda index, stacks: AttrDict(
        doc = 'Iterate a function over each suffix of an array (func → array → array)',
        call = digits(helpers.suffix_predicate),
        arity = 2,
        new = index,
    ),

    'Ꭼ':lambda index, stacks: AttrDict(
        doc = 'Cycle through an array and a series of functions, creating a new array from the results (array → ..func → array)',
        call = helpers.tie,
        arity = dynamic_arity(stacks, index),
        new = index,
    ),

    'Ꭽ':lambda index, stacks: AttrDict(
        doc = 'Apply a function to elements at even indexes (func → array → array)',
        call = ranges(helpers.apply_even),
        arity = 2,
        new = index,
    ),

    'Ꭾ':lambda index, stacks: AttrDict(
        doc = 'Apply a function to elements at odd indexes (func → array → array)',
        call = ranges(helpers.apply_odd),
        arity = 2,
        new = index,
    ),

    'Ꭿ':lambda index, stacks: AttrDict(
        doc = 'Do-while loop (pred → func → any)',
        call = partial(helpers.while_loop, do = True),
        arity = 2,
        new = index,
    ),

    'Ꮀ':lambda index, stacks: AttrDict(
        doc = 'Do-while loop, collecting the results (pred → func → array)',
        call = partial(helpers.while_loop, accumulate = True, do = True),
        arity = 2,
        new = index,
    ),

    'Ꮁ':lambda index, stacks: AttrDict(
        doc = 'Absolute difference (num → num → num)',
        call = iterable(nest(abs, operator.sub)),
        arity = 2,
        new = index,
    ),

    'Ꮂ':lambda index, stacks: AttrDict(
        doc = 'Group equal elements in an array (array → [array])',
        call = digits(helpers.group_equal),
        arity = 1,
        new = index,
    ),

    'Ꮃ':lambda index, stacks: AttrDict(
        doc = 'Zip with a filler for the shorter array (array → array → any → [array])',
        call = helpers.zip,
        arity = 3,
        new = index,
    ),

    'Ꮄ':lambda index, stacks: AttrDict(
        doc = 'Repeat an array (array → int → array)',
        call = reverse(helpers.nrepeat),
        arity = 2,
        new = index,
    ),

    'Ꮅ':lambda index, stacks: AttrDict(
        doc = 'Repeat a nested array (array → int → [array])',
        call = reverse(partial(helpers.nrepeat, wrap = True)),
        arity = 2,
        new = index,
    ),

    'Ꮆ':lambda index, stacks: AttrDict(
        doc = 'Repeat each element in place (array → int → array)',
        call = reverse(partial(helpers.nrepeat, inplace = True)),
        arity = 2,
        new = index,
    ),

    'Ꮇ':lambda index, stacks: AttrDict(
        doc = 'Set difference (array → array → array)',
        call = helpers.difference,
        arity = 2,
        new = index,
    ),

    'Ꮈ':lambda index, stacks: AttrDict(
        doc = 'Yield []',
        call = nilad([]),
        arity = 0,
        new = index,
    ),

    'Ꮉ':lambda index, stacks: AttrDict(
        doc = 'Yield the empty string',
        call = nilad(''),
        arity = 0,
        new = index,
    ),

    'Ꮊ':lambda index, stacks: AttrDict(
        doc = 'Yield a newline',
        call = nilad('\n'),
        arity = 0,
        new = index,
    ),

    'Ꮋ':lambda index, stacks: AttrDict(
        doc = 'Yield a space',
        call = nilad(' '),
        arity = 0,
        new = index,
    ),

    'Ꮌ':lambda index, stacks: AttrDict(
        doc = 'Yield 10',
        call = nilad(10),
        arity = 0,
        new = index,
    ),

    'Ꮍ':lambda index, stacks: AttrDict(
        doc = 'Yield 16',
        call = nilad(16),
        arity = 0,
        new = index,
    ),

    'Ꮒ':lambda index, stacks: AttrDict(
        doc = 'Yield 100',
        call = nilad(100),
        arity = 0,
        new = index,
    ),

    'Ꮎ':lambda index, stacks: AttrDict(
        doc = 'Raise 2 to the power (num → num)',
        call = iterable(partial(operator.pow, 2)),
        arity = 1,
        new = index,
    ),

    'Ꮑ':lambda index, stacks: AttrDict(
        doc = 'Raise 10 to the power (num → num)',
        call = iterable(partial(operator.pow, 10)),
        arity = 1,
        new = index,
    ),

    'Ꮓ':lambda index, stacks: AttrDict(
        doc = 'Subfactorial (int → int)',
        call = iterable(helpers.subfactorial),
        arity = 1,
        new = index,
    ),

    'Ꮔ':lambda index, stacks: AttrDict(
        doc = 'Numerical sign (real → int)',
        call = iterable(helpers.sign),
        arity = 1,
        new = index,
    ),

    'Ꮕ':lambda index, stacks: AttrDict(
        doc = 'Reciprocal (num → num)',
        call = iterable(partial(operator.truediv, 1)),
        arity = 1,
        new = index,
    ),

    'Ꮛ':lambda index, stacks: AttrDict(
        doc = 'Is positive? (real → bool)',
        call = iterable(partial(operator.lt, 0)),
        arity = 1,
        new = index,
    ),

    'Ꮚ':lambda index, stacks: AttrDict(
        doc = 'Yield the prime at the specified index (int → int)',
        call = iterable(partial(helpers.nfind, helpers.isprime, tail = True)),
        arity = 1,
        new = index,
    ),

    'Ꮘ':lambda index, stacks: AttrDict(
        doc = 'Is a perfect square? (int → bool)',
        call = iterable(partial(helpers.invariant, nest(partargs(operator.pow, {2: 2}), int, math.sqrt))),
        arity = 1,
        new = index,
    ),

    'Ꮖ':lambda index, stacks: AttrDict(
        doc = 'Is negative? (real → bool)',
        call = iterable(partial(operator.gt, 0)),
        arity = 1,
        new = index,
    ),

    'Ꮗ':lambda index, stacks: AttrDict(
        doc = 'Run as Orst code (str → array)',
        call = partargs(exec_orst, {2: stacks[index].input}),
        arity = 1,
        new = index,
    ),

    'Ꮙ':lambda index, stacks: AttrDict(
        doc = 'Is sorted ascendingly? (array → bool)',
        call = digits(nest(helpers.invariant, sorted)),
        arity = 1,
        new = index,
    ),

    'Ꮢ':lambda index, stacks: AttrDict(
        doc = 'Is sorted ascendingly or decendingly? (array → bool)',
        call = digits(helpers.is_sorted),
        arity = 1,
        new = index,
    ),

    'Ꮡ':lambda index, stacks: AttrDict(
        doc = 'Yield all contiguous sublists (array → array)',
        call = digits(helpers.contiguous_sublists),
        arity = 1,
        new = index,
    ),

    'Ꮟ':lambda index, stacks: AttrDict(
        doc = 'Run-length decode ([array] → array)',
        call = helpers.rld,
        arity = 1,
        new = index,
    ),

    'Ꮜ':lambda index, stacks: AttrDict(
        doc = 'Yield all derangements (array → array)',
        call = digits(helpers.derangements),
        arity = 1,
        new = index,
    ),

    'Ꮞ':lambda index, stacks: AttrDict(
        doc = 'Is the array a derangement of another array? (array → array → bool)',
        call = helpers.is_derangement,
        arity = 2,
        new = index,
    ),

    'Ꮠ':lambda index, stacks: AttrDict(
        doc = 'Unpack the head of the array (array → [any → array])',
        call = tuple_application(helpers.head, helpers.behead),
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ꮫ':lambda index, stacks: AttrDict(
        doc = 'Unpack the head of the array (array → [array → any])',
        call = tuple_application(helpers.behead, helpers.head),
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ꮪ':lambda index, stacks: AttrDict(
        doc = 'Unpack the tail of the array (array → [any → array])',
        call = tuple_application(helpers.tail, helpers.shorten),
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ꮩ':lambda index, stacks: AttrDict(
        doc = 'Unpack the tail of the array (array → [array → any])',
        call = tuple_application(helpers.shorten, helpers.tail),
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ꮧ':lambda index, stacks: AttrDict(
        doc = 'Split on spaces (str → array)',
        call = partial(str.split, sep = ' '),
        arity = 1,
        new = index,
    ),

    'Ꮣ':lambda index, stacks: AttrDict(
        doc = 'Unpack to the stack (array → ...any)',
        call = identity,
        arity = 1,
        unpack = True,
        new = index,
    ),

    'Ꮥ':lambda index, stacks: AttrDict(
        doc = 'Partitions of the array (array → [array])',
        call = digits(helpers.partitions),
        arity = 1,
        new = index,
    ),

    'Ꮲ':lambda index, stacks: AttrDict(
        doc = 'Palindromise the array (array → array)',
        call = digits(helpers.bounce),
        arity = 1,
        new = index,
    ),

    'Ꮱ':lambda index, stacks: AttrDict(
        doc = 'Split on newlines (str → array)',
        call = partial(str.split, sep = '\n'),
        arity = 1,
        new = index,
    ),

    'Ꮰ':lambda index, stacks: AttrDict(
        doc = 'Join an array (array → str)',
        call = digits(partial(helpers.join, '')),
        arity = 1,
        new = index,
    ),

    'Ꮯ':lambda index, stacks: AttrDict(
        doc = 'Join an array on spaces (array → str)',
        call = digits(partial(helpers.join, ' ')),
        arity = 1,
        new = index,
    ),

    'Ꮮ':lambda index, stacks: AttrDict(
        doc = 'Join an array on newlines (array → str)',
        call = digits(partial(helpers.join, '\n')),
        arity = 1,
        new = index,
    ),

    'Ꮭ':lambda index, stacks: AttrDict(
        doc = 'Split on newlines, then split each line at spaces (str → [array])',
        call = nest(partial(map, partial(str.split, sep = ' ')), partial(str.split, sep = '\n')),
        arity = 1,
        new = index,
    ),

    'Ꭱ':lambda index, stacks: AttrDict(
        doc = 'Concatenate an array internally (array → str)',
        call = nest(''.join, partial(map, str)),
        arity = 1,
        new = index,
    ),

    'Ꭲ':lambda index, stacks: AttrDict(
        doc = 'Grade up (array → array)',
        call = helpers.grade_up,
        arity = 1,
        new = index,
    ),

    'Ꭳ':lambda index, stacks: AttrDict(
        doc = 'Grade up twice (array → array)',
        call = nest(helpers.grade_up, helpers.grade_up),
        arity = 1,
        new = index,
    ),

    'Ꭴ':lambda index, stacks: AttrDict(
        doc = 'Depth (array → int)',
        call = helpers.depth,
        arity = 1,
        new = index,
    ),

    'Ꭵ':lambda index, stacks: AttrDict(
        doc = 'Enumerate, starting at 1 (array → array)',
        call = digits(partargs(enumerate, {2: 1})),
        arity = 1,
        new = index,
    ),

    '₽':lambda index, stacks: AttrDict(
        doc = 'Powerset (array → array)',
        call = digits(helpers.powerset),
        arity = 1,
        new = index,
    ),

    '¥':lambda index, stacks: AttrDict(
        doc = 'Length range (array → array)',
        call = digits(nest(helpers.range, len)),
        arity = 1,
        new = index,
    ),

    '£':lambda index, stacks: AttrDict(
        doc = 'Form a matrix display ([array] → str)',
        call = nest('\n'.join, partial(map, partial(helpers.join, ' '))),
        arity = 1,
        new = index,
    ),

    '$':lambda index, stacks: AttrDict(
        doc = 'Return every nth element (array → int → array)',
        call = reverse(helpers.nth_elements),
        arity = 2,
        new = index,
    ),

    '¢':lambda index, stacks: AttrDict(
        doc = 'Return every other element (array → array)',
        call = digits(partargs(helpers.nth_elements, {2: 2})),
        arity = 1,
        new = index,
    ),

    '€':lambda index, stacks: AttrDict(
        doc = 'Set union (array → array → array)',
        call = helpers.union,
        arity = 2,
        new = index,
    ),

    '₩':lambda index, stacks: AttrDict(
        doc = 'Slice into chunks of the specified length (array → int → array)',
        call = reverse(helpers.chunks_of_n),
        arity = 2,
        new = index,
    ),

    '&':lambda index, stacks: AttrDict(
        doc = 'Set intersection (array → array → array)',
        call = helpers.intersection,
        arity = 2,
        new = index,
    ),

    '…':lambda index, stacks: AttrDict(
        doc = 'Slice into the specified number of chunks (array → int → array)',
        call = reverse(helpers.nchunks),
        arity = 2,
        new = index,
    ),

    '§':lambda index, stacks: AttrDict(
        doc = 'Push the second element on the stack (any → any → [any → any])',
        call = helpers.from_below,
        arity = 2,
        unpack = True,
        new = index,
    ),

    'Љ':lambda index, stacks: AttrDict(
        doc = 'Take the first element of an array (array → any)',
        call = helpers.head,
        arity = 1,
        new = index,
    ),

    'Њ':lambda index, stacks: AttrDict(
        doc = 'Arcsecant (real → real)',
        call = iterate(nest(math.acos, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Е':lambda index, stacks: AttrDict(
        doc = 'Arccosecant (real → real)',
        call = iterate(nest(math.asin, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Р':lambda index, stacks: AttrDict(
        doc = 'Arccotangent (real → real)',
        call = iterate(nest(math.atan, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Т':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arcsecant (real → real)',
        call = iterate(nest(math.acosh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'З':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arccosecant (real → real)',
        call = iterate(nest(math.asinh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'У':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic arccotangent (real → real)',
        call = iterate(nest(math.atanh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'И':lambda index, stacks: AttrDict(
        doc = 'Secant (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.cos)),
        arity = 1,
        new = index,
    ),

    'О':lambda index, stacks: AttrDict(
        doc = 'Cosecant (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.sin)),
        arity = 1,
        new = index,
    ),

    'П':lambda index, stacks: AttrDict(
        doc = 'Cotangent (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.tan)),
        arity = 1,
        new = index,
    ),

    'Ш':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic secant (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.cosh)),
        arity = 1,
        new = index,
    ),

    'А':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic cosecant (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.sinh)),
        arity = 1,
        new = index,
    ),

    'С':lambda index, stacks: AttrDict(
        doc = 'Hyperbolic cotangent (real → real)',
        call = iterate(nest(partial(operator.truediv, 1), math.tanh)),
        arity = 1,
        new = index,
    ),

    'Д':lambda index, stacks: AttrDict(
        doc = 'Complex arccosine (complex → complex)',
        call = iterate(cmath.acos),
        arity = 1,
        new = index,
    ),

    'Ф':lambda index, stacks: AttrDict(
        doc = 'Complex arcsine (complex → complex)',
        call = iterate(cmath.asin),
        arity = 1,
        new = index,
    ),

    'Г':lambda index, stacks: AttrDict(
        doc = 'Complex arctangent (complex → complex)',
        call = iterate(cmath.atan),
        arity = 1,
        new = index,
    ),

    'Х':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arccosine (complex → complex)',
        call = iterate(cmath.acosh),
        arity = 1,
        new = index,
    ),

    'Ј':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arcsine (complex → complex)',
        call = iterate(cmath.asinh),
        arity = 1,
        new = index,
    ),

    'К':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arctangent (complex → complex)',
        call = iterate(cmath.atanh),
        arity = 1,
        new = index,
    ),

    'Л':lambda index, stacks: AttrDict(
        doc = 'Complex cosine (complex → complex)',
        call = iterate(cmath.cos),
        arity = 1,
        new = index,
    ),

    'Ч':lambda index, stacks: AttrDict(
        doc = 'Complex sine (complex → complex)',
        call = iterate(cmath.sin),
        arity = 1,
        new = index,
    ),

    'Ћ':lambda index, stacks: AttrDict(
        doc = 'Complex tangent (complex → complex)',
        call = iterate(cmath.tan),
        arity = 1,
        new = index,
    ),

    'Ѕ':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic cosine (complex → complex)',
        call = iterate(cmath.cosh),
        arity = 1,
        new = index,
    ),

    'Џ':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic sine (complex → complex)',
        call = iterate(cmath.sinh),
        arity = 1,
        new = index,
    ),

    'Ц':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic tangent (complex → complex)',
        call = iterate(cmath.tanh),
        arity = 1,
        new = index,
    ),

    'В':lambda index, stacks: AttrDict(
        doc = 'Complex arcsecant (complex → complex)',
        call = iterate(nest(cmath.acos, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Б':lambda index, stacks: AttrDict(
        doc = 'Complex arccosecant (complex → complex)',
        call = iterate(nest(cmath.asin, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Н':lambda index, stacks: AttrDict(
        doc = 'Complex arccotangent (complex → complex)',
        call = iterate(nest(cmath.atan, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'М':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arcsecant (complex → complex)',
        call = iterate(nest(cmath.acosh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Ђ':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arccosecant (complex → complex)',
        call = iterate(nest(cmath.asinh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'Ж':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic arccotangent (complex → complex)',
        call = iterate(nest(cmath.atanh, partial(operator.truediv, 1))),
        arity = 1,
        new = index,
    ),

    'љ':lambda index, stacks: AttrDict(
        doc = 'Complex secant (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.cos)),
        arity = 1,
        new = index,
    ),

    'њ':lambda index, stacks: AttrDict(
        doc = 'Complex cosecant (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.sin)),
        arity = 1,
        new = index,
    ),

    'е':lambda index, stacks: AttrDict(
        doc = 'Complex cotangent (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.tan)),
        arity = 1,
        new = index,
    ),

    'р':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic secant (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.cosh)),
        arity = 1,
        new = index,
    ),

    'т':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic cosecant (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.sinh)),
        arity = 1,
        new = index,
    ),

    'з':lambda index, stacks: AttrDict(
        doc = 'Complex hyperbolic cotangent (complex → complex)',
        call = iterate(nest(partial(operator.truediv, 1), cmath.tanh)),
        arity = 1,
        new = index,
    ),

    'у':lambda index, stacks: AttrDict(
        doc = 'Complex exponential function (complex → complex)',
        call = iterate(cmath.exp),
        arity = 1,
        new = index,
    ),

    'и':lambda index, stacks: AttrDict(
        doc = 'Complex natural logarithm (complex → complex)',
        call = iterate(cmath.log),
        arity = 1,
        new = index,
    ),

    'о':lambda index, stacks: AttrDict(
        doc = 'Complex logarithm (complex → num → complex)',
        call = iterate(cmath.log),
        arity = 2,
        new = index,
    ),

    'п':lambda index, stacks: AttrDict(
        doc = 'Complex logarithm in base 2 (complex → complex)',
        call = iterate(partargs(cmath.log, {2: 2})),
        arity = 1,
        new = index,
    ),

    'ш':lambda index, stacks: AttrDict(
        doc = 'Complex logarithm in base 10 (complex → complex)',
        call = iterate(partargs(cmath.log, {2: 10})),
        arity = 1,
        new = index,
    ),

    'а':lambda index, stacks: AttrDict(
        doc = 'Square root, handling negative numbers (num → num)',
        call = iterate(cmath.sqrt),
        arity = 1,
        new = index,
    ),

    'с':lambda index, stacks: AttrDict(
        doc = 'Generalised fibonacci function (int → int → int → int)',
        call = iterate(helpers.fibonacci),
        arity = 3,
        new = index,
    ),

    'д':lambda index, stacks: AttrDict(
        doc = 'Function-modified fibonacci (int → func → int)',
        call = reverse(partargs(helpers.fibonacci, {2: 1, 3: 1})),
        arity = 2,
        new = index,
    ),

    'ф':lambda index, stacks: AttrDict(
        doc = 'Completely generalised fibonacci (int → int → int → func → int)',
        call = reverse(helpers.fibonacci),
        arity = 4,
        new = index,
    ),

    'г':lambda index, stacks: AttrDict(
        doc = 'Final digit (int → int)',
        call = iterate(partargs(operator.mod, {2: 10})),
        arity = 1,
        new = index,
    ),

    'х':lambda index, stacks: AttrDict(
        doc = 'Yield the uppercase alphabet',
        call = nilad('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        arity = 0,
        new = index,
    ),

    'ј':lambda index, stacks: AttrDict(
        doc = 'Yield the lowercase alphabet',
        call = nilad('abcdefghijklmnopqrstuvwxyz'),
        arity = 0,
        new = index,
    ),

    'к':lambda index, stacks: AttrDict(
        doc = 'Top and tail (array → array)',
        call = digits(helpers.topandtail),
        arity = 1,
        new = index,
    ),

    'л':lambda index, stacks: AttrDict(
        doc = 'Behead an array (array → array)',
        call = helpers.behead,
        arity = 1,
        new = index,
    ),

    'ч':lambda index, stacks: AttrDict(
        doc = 'Yield \'#\'',
        call = nilad('#'),
        arity = 0,
        new = index,
    ),

    'ћ':lambda index, stacks: AttrDict(
        doc = 'Prime index (int → int)',
        call = iterable(helpers.prime_index),
        arity = 1,
        new = index,
    ),

    'ѕ':lambda index, stacks: AttrDict(
        doc = 'Greatest common divisor (GCD) (int → int → int)',
        call = math.gcd,
        arity = 2,
        new = index,
    ),

    'џ':lambda index, stacks: AttrDict(
        doc = 'Equality (any → any → bool)',
        call = operator.eq,
        arity = 2,
        new = index,
    ),

    'ц':lambda index, stacks: AttrDict(
        doc = 'Inequality (any → any → bool)',
        call = operator.ne,
        arity = 2,
        new = index,
    ),

    'в':lambda index, stacks: AttrDict(
        doc = 'Next prime (int → int)',
        call = iterate(partial(helpers.next_prev, func = 'prime', mode = 'next')),
        arity = 1,
        new = index,
    ),

    'б':lambda index, stacks: AttrDict(
        doc = 'Previous prime (int → int)',
        call = iterate(partial(helpers.next_prev, func = 'prime', mode = 'prev')),
        arity = 1,
        new = index,
    ),

    'н':lambda index, stacks: AttrDict(
        doc = 'Next Fibonnaci number (int → int)',
        call = iterate(partial(helpers.next_prev, func = 'fib', mode = 'next')),
        arity = 1,
        new = index,
    ),

    'м':lambda index, stacks: AttrDict(
        doc = 'Previous Fibonacci number (int → int)',
        call = iterate(partial(helpers.next_prev, func = 'fib', mode = 'prev')),
        arity = 1,
        new = index,
    ),

    'ђ':lambda index, stacks: AttrDict(
        doc = 'Next value in an infinite sequence (inf → int → int)',
        call = partial(helpers.next_prev, func = 'inf', mode = 'next'),
        arity = 2,
        new = index,
    ),

    'ж':lambda index, stacks: AttrDict(
        doc = 'Previous value in an infinite sequence (inf → int → int)',
        call = partial(helpers.next_prev, fund = 'inf', mode = 'prev'),
        arity = 2,
        new = index,
    ),

    '¤':lambda index, stacks: AttrDict(
        doc = 'Internal representation (any → str)',
        call = repr,
        arity = 1,
        new = index,
    ),

    'þ':lambda index, stacks: AttrDict(
        doc = 'Find the index in an infinite list (inf → any → int)',
        call = reverse(helpers.InfiniteList.index),
        arity = 2,
        new = index,
    ),

    '@':lambda index, stacks: AttrDict(
        doc = 'Take the first number of elements from an infinite list (inf → int → array)',
        call = reverse(helpers.InfiniteList.take),
        arity = 2,
        new = index,
    ),

    '%':lambda index, stacks: AttrDict(
        doc = 'Drop an element from the start of an infinite list (inf → inf)',
        call = helpers.InfiniteList.drop,
        arity = 1,
        new = index,
    ),

    '‰':lambda index, stacks: AttrDict(
        doc = 'Drop elements from the start of an infinite list (inf → int → inf)',
        call = reverse(helpers.InfiniteList.drop),
        arity = 2,
        new = index,
    ),

    '#':lambda index, stacks: AttrDict(
        doc = 'Restore any dropped elements (inf → inf)',
        call = helpers.InfiniteList.reset,
        arity = 1,
        new = index,
    ),

    '?':lambda index, stacks: AttrDict(
        doc = 'Infinitely cycle an array (array → inf)',
        call = helpers.cycle,
        arity = 1,
        new = index,
    ),
    
    '¿':lambda index, stacks: AttrDict(
        doc = 'Infinitely cycle an array, saving it as a sequence (array → str → inf)',
        call = helpers.cycle_ref,
        arity = 2,
        new = index,
    ),
    
    '!':lambda index, stacks: AttrDict(
        doc = 'Create an infinite list from a function (func → inf)',
        call = helpers.generator,
        arity = 1,
        new = index,
    ),
    
    '¡':lambda index, stacks: AttrDict(
        doc = 'Create an infinite list from a function, saving it as a sequence (func → str → inf)',
        call = helpers.generator_ref,
        arity = 2,
        new = index,
    ),

    '‹':lambda index, stacks: AttrDict(
        doc = 'Python-style monadic range (int → array)',
        call = iterate(range),
        arity = 1,
        new = index,
    ),

    '◊':lambda index, stacks: AttrDict(
        doc = 'Python-style dyadic range (int → int → array)',
        call = iterate(range),
        arity = 2,
        new = index,
    ),

    '›':lambda index, stacks: AttrDict(
        doc = 'Python-style triadic range (int → int → int → array)',
        call = iterate(range),
        arity = 3,
        new = index,
    ),

    '—':lambda index, stacks: AttrDict(
        doc = 'Yield the next element of an infinite list (inf → any)',
        call = next,
        arity = 1,
        new = index,
    ),

    '/':lambda index, stacks: AttrDict(
        doc = 'Yield the next element of an infinite list (inf → [inf → any])',
        call = peaceful(next),
        arity = 1,
        unpack = True,
        new = index,
    ),
    
    '\\':lambda index, stacks: AttrDict(
        doc = 'Yield the next element of an infinite list (inf → [any → inf])',
        call = nest(reversed, peaceful(next)),
        arity = 1,
        unpack = True,
        new = index,
    ),

    ':':lambda index, stacks: AttrDict(
        doc = 'Push the length of the stack, clearing the stack',
        call = tostack(len, False),
        arity = len(stacks[index]),
        new = index,
    ),

    ';':lambda index, stacks: AttrDict(
        doc = 'Wrap the stack in an array, clearing the stack',
        call = tostack(helpers.wrap, False),
        arity = len(stacks[index]),
        new = index,
    ),

    '^':lambda index, stacks: AttrDict(
        doc = 'Push the length of the stack',
        call = tostack(len, True),
        arity = len(stacks[index]),
        unpack = True,
        new = index,
    ),

    '*':lambda index, stacks: AttrDict(
        doc = 'Wrap the stack in an array',
        call = tostack(helpers.wrap, True),
        arity = len(stacks[index]),
        unpack = True,
        new = index,
    ),

    '+':lambda index, stacks: AttrDict(
        doc = 'Convert to the specified case (str → int → str)',
        call = reverse(helpers.cases),
        arity = 2,
        new = index,
    ),

    '_':lambda index, stacks: AttrDict(
        doc = 'Remove all save the top element of the stack',
        call = helpers.keep,
        arity = len(stacks[index]),
        new = index,
    ),

    '|':lambda index, stacks: AttrDict(
        doc = 'Halve (num → num)',
        call = iterate(partargs(operator.truediv, {2: 2})),
        arity = 1,
        new = index,
    ),

    '~':lambda index, stacks: AttrDict(
        doc = 'Integer halve (num → int)',
        call = iterate(partargs(operator.floordiv, {2: 2})),
        arity = 1,
        new = index,
    ),

    '<':lambda index, stacks: AttrDict(
        doc = 'Final digit of power (real → real → int)',
        call = iterate(partargs(pow, {3: 10})),
        arity = 2,
        new = index,
    ),

    '≤':lambda index, stacks: AttrDict(
        doc = 'Parity of power (real → real → int)',
        call = iterate(partargs(pow, {3: 2})),
        arity = 2,
        new = index,
    ),

    '«':lambda index, stacks: AttrDict(
        doc = 'Square (num → num)',
        call = iterate(partargs(pow, {2: 2})),
        arity = 1,
        new = index,
    ),

    '·':lambda index, stacks: AttrDict(
        doc = 'Convert to a left-justified grid (array → str)',
        call = helpers.grid,
        arity = 1,
        new = index,
    ),

    '»':lambda index, stacks: AttrDict(
        doc = 'Integer square root (real → int)',
        call = iterate(nest(int, math.sqrt)),
        arity = 1,
        new = index,
    ),

    '≥':lambda index, stacks: AttrDict(
        doc = 'Inverse factorial (int → int)',
        call = iterate(inverse(math.factorial)),
        arity = 1,
        new = index,
    ),

    '>':lambda index, stacks: AttrDict(
        doc = 'Inverse fibonacci (int → int)',
        call = iterate(inverse(helpers.fib)),
        arity = 1,
        new = index,
    ),

    '¶':lambda index, stacks: AttrDict(
        doc = 'Fibonacci (int → int)',
        call = iterate(helpers.fib),
        arity = 1,
        new = index,
    ),

    'ˋ':lambda index, stacks: AttrDict(
        doc = 'Swap the top two elements',
        call = helpers.swap,
        arity = 2,
        unpack = True,
        new = index,
    ),

    'ˆ':lambda index, stacks: AttrDict(
        doc = 'Rotate the top three elements',
        call = helpers.swap,
        arity = 3,
        unpack = True,
        new = index,
    ),

    '¨':lambda index, stacks: AttrDict(
        doc = 'Convert to a right-justified grid (array → str)',
        call = partial(helpers.grid, side = 'right'),
        arity = 1,
        new = index,
    ),

    '´':lambda index, stacks: AttrDict(
        doc = 'Rotate the top three elements the opposite direction',
        call = helpers.rotswap,
        arity = 3,
        unpack = True,
        new = index,
    ),

    'ˇ':lambda index, stacks: AttrDict(
        doc = 'GCD of a list (array → int)',
        call = partial(functools.reduce, math.gcd),
        arity = 1,
        new = index,
    ),

    '∞':lambda index, stacks: AttrDict(
        doc = 'Indexes of occurences (array → any → [int])',
        call = nest(helpers.truthy_indicies, operator.contains),
        arity = 2,
        new = index,
    ),

    '∂':lambda index, stacks: AttrDict(
        doc = 'Divides? (int → int → bool)',
        call = iterate(nest(partial(operator.eq, 0), operator.mod)),
        arity = 2,
        new = index,
    ),

    '×':lambda index, stacks: AttrDict(
        doc = 'Dot product (array → array → int)',
        call = nest(sum, partial(helpers.zipwith, operator.mul)),
        arity = 2,
        new = index,
    ),  

    '⁰':lambda index, stacks: AttrDict(
        doc = 'Equal to zero? (any → bool)',
        call = iterate(partial(operator.eq, 0)),
        arity = 1,
        new = index,
    ),

    '¹':lambda index, stacks: AttrDict(
        doc = 'Not equal to zero? (any → bool)',
        call = iterate(partial(operator.ne, 0)),
        arity = 1,
        new = index,
    ),

    # ≠†ƒ¬µﬁﬂ²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾÷₊₋₌₍₎
    
    # ª commands

    'ªÆ':lambda index, stacks: AttrDict(
        doc = 'Absolute value equality (iterative) (real → real → bool)',
        call = iterate(helpers.abs_inequality),
        arity = 2,
        new = index,
    ),

    'ªÅ':lambda index, stacks: AttrDict(
        doc = 'Absolute value greater than or equal to? (real → real → bool)',
        call = iterate(partial(helpers.abs_inequality, mode = 'ge')),
        arity = 2,
        new = index,
    ),

    'ªĄ':lambda index, stacks: AttrDict(
        doc = 'Absolute value greater than? (real → real → bool)',
        call = iterate(partial(helpers.abs_inequality, mode = 'gt')),
        arity = 2,
        new = index,
    ),

    'ªÇ':lambda index, stacks: AttrDict(
        doc = 'Absolute value less than or equal to? (real → real → bool)',
        call = iterate(partial(helpers.abs_inequality, mode = 'le')),
        arity = 2,
        new = index,
    ),

    'ªĎ':lambda index, stacks: AttrDict(
        doc = 'Absolute value less than? (real → real → bool)',
        call = iterate(partial(helpers.abs_inequality, mode = 'lt')),
        arity = 2,
        new = index,
    ),

    'ªÈ':lambda index, stacks: AttrDict(
        doc = 'Absolute value inequality (iterative) (real → real → bool)',
        call = iterate(partial(helpers.abs_inequality, mode = 'ne')),
        arity = 2,
        new = index,
    ),

    'ªΐ':lambda index, stacks: AttrDict(
        doc = 'Push the value of $y',
        call = nilad(variables['y']),
        arity = 0,
        new = index,
    ),

    'ªκ':lambda index, stacks: AttrDict(
        doc = 'Set the value of $y',
        call = partargs(helpers.assign, {1: variables, 2: 'y'}),
        arity = 1,
        new = index,
    ),

    'ªџ':lambda index, stacks: AttrDict(
        doc = 'Absolute value equality (real → real → bool)',
        call = helpers.abs_inequality,
        arity = 2,
        new = index,
    ),

    'ªц':lambda index, stacks: AttrDict(
        doc = 'Absolute value inequality (real → real → bool)',
        call = partial(helpers.abs_inequality, mode = 'ne'),
        arity = 2,
        new = index,
    ),

    # º commands

    'ºA':lambda index, stacks: AttrDict(
        doc = 'Yield Orst\'s code page',
        call = nilad(code_page),
        arity = 0,
        new = index,
    ),

    'ºΐ':lambda index, stacks: AttrDict(
        doc = 'Push the value of $z',
        call = nilad(variables['z']),
        arity = 0,
        new = index,
    ),

    'ºκ':lambda index, stacks: AttrDict(
        doc = 'Set the value of $z',
        call = partargs(helpers.assign, {1: variables, 2: 'z'}),
        arity = 1,
        new = index,
    ),

    ## Possible functions:
    
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog = './orst')

    a = 'store_true'

    getcode = parser.add_mutually_exclusive_group()
    getcode.add_argument('-f', '--file', help = 'Specifies that code be read from a file', action = a)
    getcode.add_argument('-c', '--cmd', '--cmdline', help = 'Specifies that code be read from the command line', action = a)
    
    parser.add_argument('-u', '--unicode', help = 'Use Unicode encoding', action = a)
    parser.add_argument('-a', '--answer', help = 'Outputs formatted answer', action = a)
    parser.add_argument('-l', '--length', help = 'Outputs the length of the program', action = a)
    parser.add_argument('-s', '--substitute', help = 'Substitute for common symbols', action = a)
    parser.add_argument('-d', '--direct', help = 'Include direct substitutes in substitution', action = a)
    parser.add_argument('-o', '--output', help = 'Output the whole stack', action = a)
    parser.add_argument('-p', '--parsed', help = 'Output the parsed code to STDERR', action = a)

    extra = parser.add_mutually_exclusive_group()
    extra.add_argument('--compress', help = 'Compress the given string', action = a)
    extra.add_argument('--find', help = 'Find commands by documentation', action = a)

    parser.add_argument('program')
    parser.add_argument('argv', nargs = '*', type = eval)
    settings = parser.parse_args()

    if settings.cmd:
        code = settings.program
        
    elif settings.file:
        
        with open(settings.program, mode = 'rb') as file:
            contents = file.read()
            
        if settings.unicode:
            code = contents.decode('utf-8')
        else:
            code = decode(contents)

    else:
        raise IOError('No code given')

    if settings.compress:
        import compressor
        for string in [code] + settings.argv:
            tokens = re.findall(r'[A-Z][a-z]+|[a-z]+ | [a-z]+|[A-Z]+|[a-z]+|.|\n', string)
            print(string, ':', compressor.compress_fast(tokens))
        sys.exit(0)

    elif settings.find:
        for phrase in [code] + settings.argv:
            for cmd in sorted(commands):
                doc = commands[cmd](0, [Stack([0])]).doc
                if doc and phrase.lower() in doc.lower():
                    print('\'{}\' relates to \'{}\''.format(phrase, cmd))
                    
            for cmd in full_help:
                if phrase.lower() in full_help[cmd].lower():
                    print('\'{}\' relates to {}'.format(phrase, cmd))
                
        sys.exit(0)

    if len(settings.argv) < 2:
        stks = 2
    else:
        stks = len(settings.argv)

    if settings.substitute or settings.direct:
        for old, new in subs.items():
            ret_new = commands.get(new)
            ret_old = commands.get(old)
            commands[old], commands[new] = ret_new, ret_old

        if settings.direct:
            for old, new in direct_subs.items():
                code = code.replace(old, new)

    processor = Processor(code, settings.argv, stks)
    ret = processor.execute(settings.parsed)
    out = processor.output(sep = '\n\n', flag = settings.output)

    if out:
        print(out, end = '')

    if settings.length:
        length = sum((code_page.index(i) // 256) + 1 for i in code)
        print('Length: {} bytes'.format(length), file = sys.stderr)

    if settings.answer:
        length = sum((code_page.index(i) // 256) + 1 for i in code)
        print('''# [Orst](https://github.com/cairdcoinheringaahing/Orst-Geo), {} bytes

    {}'''.format(length, code))
        
