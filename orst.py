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
ώ0123456789.-"¶
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
*+=,_|~ <≤«·»≥>ᴇ
ˋˆ¨´ˇ∞≠†∂ƒ∆¬≈µﬁﬂ
×⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾
÷₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎'''.replace('\n', '')

DIGITTRANS = str.maketrans('₁₂₃₄₅₆₇₈₉₀', '1234567890')
identity = lambda a: a
variables = {}

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

class Block:
    def __init__(self, code, glob):
        self.code = Processor(code[1:], *(glob or self.glob))
        self.glob = glob

    def __str__(self):
        return '{%s}' % ''.join(self.code.code)

    def __repr__(self):
        return '{%s}' % ''.join(self.code.code)

    def call(self, stacks):
        ret = self.code.execute(stacks)
        return ret

    __call__ = call

class Clean:
    def __init__(self, code, argv, stackcount):
        self.code = Processor(code[1:], argv, stackcount)

    def call(self, *args):
        if args:
            self.code.stacks[self.code.index].push(*args)
        
        ret = []
        for stk in self.code.execute().copy():
            stkc = stk.copy()
            stk.clear()
            if stkc:
                ret.append(stkc.pop())
                
        return ret.pop()

    def __str__(self):
        return '(%s)' % ''.join(self.code.code)

    def __repr__(self):
        return '(%s)' % ''.join(self.code.code)

    __call__ = call

class Processor:
    def __init__(self, code, args, stacks = 2, start = False):
        self.preserve = code
        self.code = tokeniser(code)
        self.stacks = [Stack(args or None) for _ in range(stacks)]
        self.args = args
        self.index = 0
        
        if start and args:
            var_order = ['x', 'y', 'z']
            for index, var in enumerate(var_order):
                variables[var] = args[index % len(args)]

    def __str__(self):
        return '''[
    {}
]'''.format('\n    '.join(map(str, self.stacks)))

    __repr__ = __str__

    def execute(self, rep_stacks = None):

        if rep_stacks is not None:
            print('>', rep_stacks)

        scollect = ''
        string = 0
        
        acollect = ''
        array = 0
        
        bcollect = ''
        block = 0
        
        ccollect = ''
        clean = 0

        for index, char in enumerate(self.code):
            if (len(char) > 1 or char.isdigit()) and not (string or array or block or clean):
                self.stacks[self.index].push(eval(char.translate(DIGITTRANS)))
                continue

            if char == '{':
                block += 1
            if char == '}':
                block -= 1
                if block == 0:
                    self.stacks[self.index].push(Block(bcollect, [self.args, len(self.stacks)]))
                    bcollect = ''
                    continue

            if char == '(':
                clean += 1
            if char == ')':
                clean -= 1
                if clean == 0:
                    count = self.stacks[self.index].peek()
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
                    self.stacks[self.index].push(eval(acollect + ']'))
                    acollect = ''
                    continue

            if char == '"':
                string ^= 1
                if not string:
                    self.stacks[self.index].push(scollect)
                    scollect = ''
                continue

            if string:
                scollect += char
                continue
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

            try:
                cmd, self.index = commands[char](self.index, self.stacks)
            except KeyError:
                print('Unknown command: "{}". Stack trace:\n{}'.format(char, repr(self)), file = sys.stderr)
                continue
            
            if cmd.arity >= 0:
                args = self.stacks[self.index].pop(count = cmd.arity, unpack = False)
                # ret = cmd.call(*args)
                try:
                    ret = cmd.call(*args)
                except Exception as e:
                    print('Error when running "{}": {}'.format(char, e), file = sys.stderr)
                    continue
            else:
                print('Error when running "{}"'.format(char), file = sys.stderr)
                continue

            if ret is None:
                continue
            
            if cmd.unpack:
                self.stacks[self.index].push(*ret)
            else:
                ret = simplify(ret)
                self.stacks[self.index].push(ret)

        # Garbage collection for uncollected syntax types
        for collect in [scollect, acollect, bcollect, ccollect]:
            if collect:
                self.stacks[self.index].push(collect)

        return self.stacks

    def output(self, sep = '\n'):
        strings = []
        out = list(filter(None, self.stacks))
        
        for stk in out:
            
            if len(stk) > 1:
                stk = '\n'.join(map(convert, stk))
                
            elif len(stk) == 1:
                stk = convert(stk[0])
                
            else:
                stk = ''
                
            strings.append(stk)
            
        return sep.join(strings).strip()

class Stack(list):
    def __init__(self, starters = None, mod = True):
        if starters is None:
            starters = [0]
        self.input = starters
        self.modular = mod

    def __repr__(self):
        final = []
        for elem in self:
            final += convert(elem)
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
        return super().__getitem__(index)

    def last(self, value):
        
        final = []
        
        for _ in range(value):
            final.append(self.peek(~_))
            
        return final
        
    def push(self, *values):
        for value in values:
            self.append(value)

    def peek(self, index=-1):
        return self[index]

    def pop(self, count = 1, indexes = (-1,), unpack = False):
        
        popped = []
        next_in = 0
        
        for _ in range(count):
            
            try:
                popped.append(super().pop(indexes[_] if _ < len(indexes) else -1))
            except IndexError:
                popped.append(self.input[next_in % len(self.input)])
                next_in += 1
                
        if unpack and popped:
            return popped if len(popped) > 1 else popped[0]
        return popped

def convert(value):
    if not isinstance(value, (int, float, complex, list, str)):
        try: value = list(value)
        except: pass
    return str(value)

def simplify(value, unpack = False):
    if isinstance(value, tuple):
        value = list(value)
        
    if isinstance(value, list):
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

    return value

def digit(string):
    regex = r'''(-?\z+)(((\k)|(\e-?)|(\j-?))(?(5)$|(\z+(\k\z+)?)))?$''' \
            .replace(r'\z', r'[\d₀₁₂₃₄₅₆₇₈₉]')                          \
            .replace(r'\e', r'[Eᴇ]')                                    \
            .replace(r'\j', r'[jј]')                                    \
            .replace(r'\k', r'[\.•]')
            
    return (not re.search(regex, string)) ^ 1

def group(array):
    tkns = ['']
    string = 0
    
    for elem in array:
        if elem == '"':
            string ^= 1
            if string:
                tkns.append('"')
            else:
                tkns[-1] += '"'
        else:
            if string:
                tkns[-1] += elem
            else:
                tkns.append(elem)

    tkns = list(filter(None, tkns))
    final = ['']
    index = 0
    while index < len(tkns):
        while index < len(tkns) and digit(tkns[index]):
            if not digit(final[-1]):
                final.append('')
            final[-1] += tkns[index]
            index += 1
        if index < len(tkns) and not digit(tkns[index]):
            final.append(tkns[index])
        index += 1
        
    return list(filter(None, final))

def tokeniser(string):
    regex = r'''^(-?\d+)(((\.)|(E-?)|(j-?))(?(5)$|(\d+(\.\d+)?)))?$'''.replace(r'\d', '[\d₀₁₂₃₄₅₆₇₈₉]')
    
    if not re.search(regex, string):
        if not re.search(regex[1:-1], string):
            return list(string)
        
        index = 0
        final = []
        temp = ''
        readstring = False
        
        while index < len(string):
            char = string[index]
            
            if char == '"':
                readstring ^= 1
                
            if char == '0' and string[index + 1] not in '.j':
                while char in '-1234567890.Ej' and not readstring:
                    temp += char
                    index += 1
                    try:
                        char = string[index]
                    except IndexError:
                        return final + [temp]
                
            if temp:
                final.append(temp)
                temp = ''
                
            final.append(char)
            index += 1
            
        if temp:
            final.append(temp)

        array = final

    else:
        
        tokens = re.findall(regex, string)
    
        if len(tokens) == 1:
            array = (
                        (
                            [tokens[0][0], tokens[0][1][0], tokens[0][1][1:]]
                            if
                                tokens[0][1][0] != '.'
                            else
                                [tokens[0][0] + tokens[0][1][0] + tokens[0][1][1:]]
                            )
                    ) if tokens[0][1] else list(tokens[0][:1])
        else:               
            array = [tokens[0][0] + '.' + tokens[1][0]] + ([tokens[1][1][0], tokens[1][1][1:]] if tokens[1][1] else [])

    return group(array)

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

def partargs(function, index_args, filler = 0):
    def inner(*args):
        args = list(args)
        for index, arg in index_args.items():
            args.insert(index - 1, arg)
        return function(*args)
    return inner

def dynamic_arity(stacks, current, minimum = 2):
    stk = stacks[current]
    arity = stk.pop(unpack = True)
    if arity == 0:
        arity = minimum - 1
    return arity + 1

def digits(function, indexes = None):
    def inner(*args):
        args = list(args)
        
        for index, arg in enumerate(args):
            perform = indexes is None or index in indexes
            if isinstance(arg, int) and perform:
                arg = helpers.to_base(arg)
            args[index] = arg
            
        return function(*args)
    return inner

def ranges(function, indexes = None):
    def inner(*args):
        args = list(args)

        for index, arg in enumerate(args):
            perform = indexes is None or index in indexes
            if isinstance(arg, int) and perform:
                arg = helpers.range(arg)
            args[index] = arg

        return function(*args)
    return inner

def iterate(function):
    def inner(*args):
        args = list(args)
        if len(args) == 1:
            arg = args[0]
            if not hasattr(arg, '__iter__'):
                arg = [arg]
            return map(function, arg)

        else:
            for index in range(len(args)):
                arg = args[index]
                if not hasattr(arg, '__iter__'):
                    arg = [arg]
                args[index] = arg
                
            length = max(map(len, args))
            args = map(itertools.repeat, args, itertools.cycle([length]))
            return map(function, *args)
    return inner

def nilad(value):
    def inner():
        return value
    return inner

listify = helpers.listify
partial = functools.partial

commands = {

    # 1 byte tokens

    'A':lambda index, stacks: (
        AttrDict(
            call = abs,
            arity = 1,
        ),
        index
    ),

    'À':lambda index, stacks: (
        AttrDict(
            call = operator.add,
            arity = 2,
        ),
        index
    ),

    'Á':lambda index, stacks: (
        AttrDict(
            call = operator.and_,
            arity = 2,
        ),
        index
    ),

    'Â':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.contains),
            arity = 2,
        ),
        index
    ),

    'Ä':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.countOf),
            arity = 2,
        ),
        index
    ),

    'Æ':lambda index, stacks: (
        AttrDict(
            call = operator.eq,
            arity = 2,
        ),
        index
    ),

    'Ã':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.floordiv),
            arity = 2,
        ),
        index
    ),

    'Å':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.ge),
            arity = 2,
        ),
        index
    ),

    'Ā':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.getitem),
            arity = 2,
        ),
        index
    ),

    'Ą':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.gt),
            arity = 2,
        ),
        index
    ),

    'B':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.indexOf),
            arity = 2,
        ),
        index
    ),

    'C':lambda index, stacks: (
        AttrDict(
            call = operator.inv,
            arity = 1,
        ),
        index
    ),

    'Ć':lambda index, stacks: (
        AttrDict(
            call = operator.is_,
            arity = 2,
        ),
        index
    ),

    'Č':lambda index, stacks: (
        AttrDict(
            call = operator.is_not,
            arity = 2,
        ),
        index
    ),

    'Ç':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.le),
            arity = 2,
        ),
        index
    ),

    'D':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.lshift),
            arity = 2,
        ),
        index
    ),

    'Ď':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.lt),
            arity = 2,
        ),
        index
    ),

    'Ð':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.mod),
            arity = 2,
        ),
        index
    ),

    'E':lambda index, stacks: (
        AttrDict(
            call = operator.mul,
            arity = 2,
        ),
        index
    ),

    'È':lambda index, stacks: (
        AttrDict(
            call = operator.ne,
            arity = 2,
        ),
        index
    ),

    'É':lambda index, stacks: (
        AttrDict(
            call = operator.neg,
            arity = 1,
        ),
        index
    ),

    'Ê':lambda index, stacks: (
        AttrDict(
            call = operator.not_,
            arity = 1,
        ),
        index
    ),

    'Ë':lambda index, stacks: (
        AttrDict(
            call = operator.or_,
            arity = 2,
        ),
        index
    ),

    'Ē':lambda index, stacks: (
        AttrDict(
            call = operator.pos,
            arity = 1,
        ),
        index
    ),

    'Ė':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.pow),
            arity = 2,
        ),
        index
    ),

    'Ě':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.rshift),
            arity = 2,
        ),
        index
    ),

    'Ę':lambda index, stacks: (
        AttrDict(
            call = reverse(helpers.setitem),
            arity = 3,
        ),
        index
    ),

    'F':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.sub),
            arity = 2,
        ),
        index
    ),

    'G':lambda index, stacks: (
        AttrDict(
            call = reverse(operator.truediv),
            arity = 2,
        ),
        index
    ),

    'H':lambda index, stacks: (
        AttrDict(
            call = bool,
            arity = 1,
        ),
        index
    ),

    'I':lambda index, stacks: (
        AttrDict(
            call = operator.xor,
            arity = 2,
        ),
        index
    ),

    'Ì':lambda index, stacks: (
        AttrDict(
            call = digits(all),
            arity = 1,
        ),
        index
    ),

    'Í':lambda index, stacks: (
        AttrDict(
            call = digits(any),
            arity = 1,
        ),
        index
    ),

    'Î':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.to_base, base = 2),
            arity = 1,
        ),
        index
    ),

    'Ï':lambda index, stacks: (
        AttrDict(
            call = nest(''.join, iterate(chr)),
            arity = 1,
        ),
        index
    ),

    'Ī':lambda index, stacks: (
        AttrDict(
            call = reverse(complex),
            arity = 2,
        ),
        index
    ),

    'Į':lambda index, stacks: (
        AttrDict(
            call = reverse(divmod),
            arity = 2,
        ),
        index
    ),

    'J':lambda index, stacks: (
        AttrDict(
            call = digits(enumerate),
            arity = 1,
        ),
        index
    ),

    'K':lambda index, stacks: (
        AttrDict(
            call = eval,
            arity = 1,
        ),
        index
    ),

    'L':lambda index, stacks: (
        AttrDict(
            call = exec,
            arity = 1,
        ),
        index
    ),

    'Ĺ':lambda index, stacks: (
        AttrDict(
            call = digits(filter),
            arity = 2,
        ),
        index
    ),

    'Ľ':lambda index, stacks: (
        AttrDict(
            call = float,
            arity = 1,
        ),
        index
    ),

    'Ł':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.to_base, base = 16),
            arity = 1,
        ),
        index
    ),

    'M':lambda index, stacks: (
        AttrDict(
            call = identity,
            arity = 1,
        ),
        index
    ),

    'N':lambda index, stacks: (
        AttrDict(
            call = nest(eval, input),
            arity = 0,
        ),
        index
    ),

    'Ń':lambda index, stacks: (
        AttrDict(
            call = int,
            arity = 1,
        ),
        index
    ),

    'Ň':lambda index, stacks: (
        AttrDict(
            call = digits(len),
            arity = 1,
        ),
        index
    ),

    'Ñ':lambda index, stacks: (
        AttrDict(
            call = digits(map),
            arity = 2,
        ),
        index
    ),

    'Ŋ':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.to_base, base = 8),
            arity = 1,
        ),
        index
    ),

    'O':lambda index, stacks: (
        AttrDict(
            call = digits(max),
            arity = 1,
        ),
        index
    ),

    'Ò':lambda index, stacks: (
        AttrDict(
            call = max,
            arity = 2,
        ),
        index
    ),

    'Ó':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.max),
            arity = 2,
        ),
        index
    ),

    'Ô':lambda index, stacks: (
        AttrDict(
            call = digits(min),
            arity = 1,
        ),
        index
    ),

    'Ö':lambda index, stacks: (
        AttrDict(
            call = min,
            arity = 2,
        ),
        index
    ),

    'Œ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.min),
            arity = 2,
        ),
        index
    ),

    'Õ':lambda index, stacks: (
        AttrDict(
            call = nest(partial(simplify, unpack = True), listify(iterate(ord))),
            arity = 1,
        ),
        index
    ),

    'Ø':lambda index, stacks: (
        AttrDict(
            call = pow,
            arity = 3,
        ),
        index
    ),

    'Ō':lambda index, stacks: (
        AttrDict(
            call = identify(print),
            arity = 1,
        ),
        index
    ),

    'P':lambda index, stacks: (
        AttrDict(
            call = identify(partial(print, stacks[index])),
            arity = 0,
        ),
        index
    ),

    'Q':lambda index, stacks: (
        AttrDict(
            call = quit,
            arity = 0,
            is_none = True,
        ),
        index
    ),

    'R':lambda index, stacks: (
        AttrDict(
            call = helpers.range,
            arity = 1,
        ),
        index
    ),

    'Ŕ':lambda index, stacks: (
        AttrDict(
            call = helpers.range,
            arity = 2,
        ),
        index
    ),

    'Ř':lambda index, stacks: (
        AttrDict(
            call = helpers.range,
            arity = 3,
        ),
        index
    ),

    'S':lambda index, stacks: (
        AttrDict(
            call = digits(reversed),
            arity = 1,
        ),
        index
    ),

    'Ś':lambda index, stacks: (
        AttrDict(
            call = round,
            arity = 1,
        ),
        index
    ),

    'Š':lambda index, stacks: (
        AttrDict(
            call = round,
            arity = 2,
        ),
        index
    ),

    'T':lambda index, stacks: (
        AttrDict(
            call = helpers.read,
            arity = 1,
        ),
        index
    ),

    'Ť':lambda index, stacks: (
        AttrDict(
            call = helpers.write,
            arity = 2,
        ),
        index
    ),

    'Ŧ':lambda index, stacks: (
        AttrDict(
            call = helpers.append_write,
            arity = 2,
        ),
        index
    ),

    'U':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.deduplicate),
            arity = 1,
        ),
        index
    ),

    'Ù':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.deduplicate_predicate),
            arity = 2,
        ),
        index
    ),

    'Ú':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.sort),
            arity = 1,
        ),
        index
    ),

    'Û':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(helpers.sort)),
            arity = 2,
        ),
        index
    ),

    'Ü':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(helpers.sort)),
            arity = 3,
        ),
        index
    ),

    'Ů':lambda index, stacks: (
        AttrDict(
            call = ranges(sum),
            arity = 1,
        ),
        index
    ),

    'Ū':lambda index, stacks: (
        AttrDict(
            call = digits(sum),
            arity = 1,
        ),
        index
    ),

    'V':lambda index, stacks: (
        AttrDict(
            call = str,
            arity = 1,
        ),
        index
    ),

    'W':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.zip),
            arity = 1,
        ),
        index
    ),

    'Ŵ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.zip),
            arity = 2,
        ),
        index
    ),

    'X':lambda index, stacks: (
        AttrDict(
            call = random.uniform,
            arity = 2,
        ),
        index
    ),

    'Y':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.choice),
            arity = 1,
        ),
        index
    ),

    'Ý':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.shuffle),
            arity = 1,
        ),
        index
    ),

    'Ŷ':lambda index, stacks: (
        AttrDict(
            call = random.randint,
            arity = 2,
        ),
        index
    ),

    'Ÿ':lambda index, stacks: (
        AttrDict(
            call = random.random,
            arity = 0,
        ),
        index
    ),

    'Z':lambda index, stacks: (
        AttrDict(
            call = random.randrange,
            arity = 1,
        ),
        index
    ),

    'Ź':lambda index, stacks: (
        AttrDict(
            call = random.randrange,
            arity = 2,
        ),
        index
    ),

    'Ž':lambda index, stacks: (
        AttrDict(
            call = random.randrange,
            arity = 3,
        ),
        index
    ),

    'Ż':lambda index, stacks: (
        AttrDict(
            call = digits(random.sample, (0,)),
            arity = 2,
        ),
        index
    ),

    'a':lambda index, stacks: (
        AttrDict(
            call = math.acos,
            arity = 1,
        ),
        index
    ),

    'à':lambda index, stacks: (
        AttrDict(
            call = math.acosh,
            arity = 1,
        ),
        index
    ),

    'á':lambda index, stacks: (
        AttrDict(
            call = math.asin,
            arity = 1,
        ),
        index
    ),

    'â':lambda index, stacks: (
        AttrDict(
            call = math.asinh,
            arity = 1,
        ),
        index
    ),

    'ä':lambda index, stacks: (
        AttrDict(
            call = math.atan,
            arity = 1,
        ),
        index
    ),

    'æ':lambda index, stacks: (
        AttrDict(
            call = math.atan2,
            arity = 2,
        ),
        index
    ),

    'ã':lambda index, stacks: (
        AttrDict(
            call = math.atanh,
            arity = 1,
        ),
        index
    ),

    'å':lambda index, stacks: (
        AttrDict(
            call = math.ceil,
            arity = 1,
        ),
        index
    ),

    'ā':lambda index, stacks: (
        AttrDict(
            call = math.cos,
            arity = 1,
        ),
        index
    ),

    'ą':lambda index, stacks: (
        AttrDict(
            call = math.cosh,
            arity = 1,
        ),
        index
    ),

    'b':lambda index, stacks: (
        AttrDict(
            call = math.degrees,
            arity = 1,
        ),
        index
    ),

    'c':lambda index, stacks: (
        AttrDict(
            call = nilad(math.e),
            arity = 0,
        ),
        index
    ),

    'ć':lambda index, stacks: (
        AttrDict(
            call = math.exp,
            arity = 1,
        ),
        index
    ),

    'č':lambda index, stacks: (
        AttrDict(
            call = math.factorial,
            arity = 1,
        ),
        index
    ),

    'ç':lambda index, stacks: (
        AttrDict(
            call = math.floor,
            arity = 1,
        ),
        index
    ),

    'd':lambda index, stacks: (
        AttrDict(
            call = math.gamma,
            arity = 1,
        ),
        index
    ),

    'ď':lambda index, stacks: (
        AttrDict(
            call = math.hypot,
            arity = 2,
        ),
        index
    ),

    'ð':lambda index, stacks: (
        AttrDict(
            call = math.log,
            arity = 1,
        ),
        index
    ),

    'e':lambda index, stacks: (
        AttrDict(
            call = math.log,
            arity = 2,
        ),
        index
    ),

    'è':lambda index, stacks: (
        AttrDict(
            call = partargs(math.log, {2: 10}),
            arity = 1,
        ),
        index
    ),

    'é':lambda index, stacks: (
        AttrDict(
            call = partargs(math.log, {2: 2}),
            arity = 1,
        ),
        index
    ),

    'ê':lambda index, stacks: (
        AttrDict(
            call = nilad(math.pi),
            arity = 0,
        ),
        index
    ),

    'ë':lambda index, stacks: (
        AttrDict(
            call = math.radians,
            arity = 1,
        ),
        index
    ),

    'ē':lambda index, stacks: (
        AttrDict(
            call = math.sin,
            arity = 1,
        ),
        index
    ),

    'ė':lambda index, stacks: (
        AttrDict(
            call = math.sinh,
            arity = 1,
        ),
        index
    ),

    'ě':lambda index, stacks: (
        AttrDict(
            call = math.sqrt,
            arity = 1,
        ),
        index
    ),

    'ę':lambda index, stacks: (
        AttrDict(
            call = math.tan,
            arity = 1,
        ),
        index
    ),

    'f':lambda index, stacks: (
        AttrDict(
            call = math.tanh,
            arity = 1,
        ),
        index
    ),

    'g':lambda index, stacks: (
        AttrDict(
            call = nilad(2 * math.pi),
            arity = 0,
        ),
        index
    ),

    'h':lambda index, stacks: (
        AttrDict(
            call = math.trunc,
            arity = 1
        ),
        index
    ),

    'i':lambda index, stacks: (
        AttrDict(
            call = cmath.phase,
            arity = 1,
        ),
        index
    ),

    'ì':lambda index, stacks: (
        AttrDict(
            call = cmath.polar,
            arity = 1,
        ),
        index
    ),

    'í':lambda index, stacks: (
        AttrDict(
            call = cmath.rect,
            arity = 2,
        ),
        index
    ),

    'î':lambda index, stacks: (
        AttrDict(
            call = partial(digits(helpers.reduce), accumulate = True),
            arity = 2,
        ),
        index
    ),

    'ï':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(itertools.combinations, (0,))),
            arity = 2,
        ),
        index
    ),

    'ī':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(itertools.combinations_with_replacement, (0,))),
            arity = 2,
        ),
        index
    ),

    'į':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.dropwhile),
            arity = 2,
        ),
        index
    ),

    'j':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.filterfalse),
            arity = 2,
        ),
        index
    ),

    'k':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.groupby),
            arity = 2,
        ),
        index
    ),

    'l':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.permutations),
            arity = 1,
        ),
        index
    ),

    'ĺ':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(itertools.permutations, (0,))),
            arity = 2,
        ),
        index
    ),

    'ľ':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.product),
            arity = 1,
        ),
        index
    ),

    'ł':lambda index, stacks: (
        AttrDict(
            call = nest(partial(map, list), digits(itertools.product)),
            arity = 2,
        ),
        index
    ),

    'm':lambda index, stacks: (
        AttrDict(
            call = reverse(digits(itertools.repeat, (0,))),
            arity = 2,
        ),
        index
    ),

    'n':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.starmap),
            arity = 2,
        ),
        index
    ),

    'ń':lambda index, stacks: (
        AttrDict(
            call = digits(itertools.takewhile),
            arity = 2,
        ),
        index
    ),

    'ň':lambda index, stacks: (
        AttrDict(
            call = functools.partial,
            arity = dynamic_arity(stacks, index),
        ),
        index
    ),

    'ñ':lambda index, stacks: (
        AttrDict(
            call = stacks[index].push,
            arity = 1,
        ),
        index
    ),

    'ŋ':lambda index, stacks: (
        AttrDict(
            call = stacks[index].pop,
            arity = 0,
        ),
        index
    ),

    'o':lambda index, stacks: (
        AttrDict(
            call = helpers.exec,
            arity = len(stacks[index]),
        ),
        index
    ),

    'ò':lambda index, stacks: (
        AttrDict(
            call = helpers.exec,
            arity = 1,
        ),
        index
    ),

    'ó':lambda index, stacks: (
        AttrDict(
            call = helpers.exec,
            arity = 2,
        ),
        index
    ),

    'ô':lambda index, stacks: (
        AttrDict(
            call = helpers.exec,
            arity = 3,
        ),
        index
    ),

    'ö':lambda index, stacks: (
        AttrDict(
            call = helpers.exec,
            arity = dynamic_arity(stacks, index, minimum = 1),
        ),
        index
    ),

    'œ':lambda index, stacks: (
        AttrDict(
            call = complex.conjugate,
            arity = 1,
        ),
        index
    ),

    'ø':lambda index, stacks: (
        AttrDict(
            call = partargs(getattr, {2: 'real'}),
            arity = 1,
        ),
        index
    ),

    'ō':lambda index, stacks: (
        AttrDict(
            call = partargs(getattr, {2: 'imag'}),
            arity = 1,
        ),
        index
    ),

    'õ':lambda index, stacks: (
        AttrDict(
            call = float.as_integer_ratio,
            arity = 1,
            unpack = True,
        ),
        index
    ),

    'p':lambda index, stacks: (
        AttrDict(
            call = float.is_integer,
            arity = 1,
        ),
        index
    ),

    'q':lambda index, stacks: (
        AttrDict(
            call = int.bit_length,
            arity = 1,
        ),
        index
    ),

    'r':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.reduce),
            arity = 2,
        ),
        index
    ),

    'ŕ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.append),
            arity = 2,
        ),
        index
    ),

    'ř':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.extend),
            arity = 2,
        ),
        index
    ),

    's':lambda index, stacks: (
        AttrDict(
            call = digits(list.index, (0,)),
            arity = 3,
        ),
        index
    ),

    'ß':lambda index, stacks: (
        AttrDict(
            call = digits(list.index, (0,)),
            arity = 4,
        ),
        index
    ),

    'ś':lambda index, stacks: (
        AttrDict(
            call = digits(list.pop),
            arity = 1,
        ),
        index
    ),

    'š':lambda index, stacks: (
        AttrDict(
            call = digits(list.pop, (0,)),
            arity = 2,
        ),
        index
    ),

    't':lambda index, stacks: (
        AttrDict(
            call = digits(list.insert, (0,)),
            arity = 3,
        ),
        index
    ),

    'ť':lambda index, stacks: (
        AttrDict(
            call = str.capitalize,
            arity = 1,
        ),
        index
    ),

    'ŧ':lambda index, stacks: (
        AttrDict(
            call = str.center,
            arity = 2,
        ),
        index
    ),

    'u':lambda index, stacks: (
        AttrDict(
            call = str.center,
            arity = 3,
        ),
        index
    ),

    'ù':lambda index, stacks: (
        AttrDict(
            call = str.count,
            arity = 2,
        ),
        index
    ),

    'ú':lambda index, stacks: (
        AttrDict(
            call = str.count,
            arity = 3,
        ),
        index
    ),

    'û':lambda index, stacks: (
        AttrDict(
            call = str.count,
            arity = 4,
        ),
        index
    ),

    'ü':lambda index, stacks: (
        AttrDict(
            call = str.endswith,
            arity = 2,
        ),
        index
    ),

    'ů':lambda index, stacks: (
        AttrDict(
            call = str.endswith,
            arity = 3,
        ),
        index
    ),

    'ū':lambda index, stacks: (
        AttrDict(
            call = str.endswith,
            arity = 4,
        ),
        index
    ),

    'v':lambda index, stacks: (
        AttrDict(
            call = str.find,
            arity = 2,
        ),
        index
    ),

    'w':lambda index, stacks: (
        AttrDict(
            call = str.find,
            arity = 3,
        ),
        index
    ),

    'ŵ':lambda index, stacks: (
        AttrDict(
            call = str.find,
            arity = 4,
        ),
        index
    ),

    'x':lambda index, stacks: (
        AttrDict(
            call = str.isalnum,
            arity = 1,
        ),
        index
    ),

    'y':lambda index, stacks: (
        AttrDict(
            call = str.isalpha,
            arity = 1,
        ),
        index
    ),

    'ý':lambda index, stacks: (
        AttrDict(
            call = str.isdigit,
            arity = 1,
        ),
        index
    ),

    'ŷ':lambda index, stacks: (
        AttrDict(
            call = str.islower,
            arity = 1,
        ),
        index
    ),

    'ÿ':lambda index, stacks: (
        AttrDict(
            call = str.isupper,
            arity = 1,
        ),
        index
    ),

    'z':lambda index, stacks: (
        AttrDict(
            call = str.join,
            arity = 2,
        ),
        index
    ),

    'ź':lambda index, stacks: (
        AttrDict(
            call = str.ljust,
            arity = 2,
        ),
        index
    ),

    'ž':lambda index, stacks: (
        AttrDict(
            call = str.ljust,
            arity = 3,
        ),
        index
    ),

    'ż':lambda index, stacks: (
        AttrDict(
            call = str.lower,
            arity = 1,
        ),
        index
    ),

    'Α':lambda index, stacks: (
        AttrDict(
            call = str.lstrip,
            arity = 1,
        ),
        index
    ),

    'Ά':lambda index, stacks: (
        AttrDict(
            call = str.lstrip,
            arity = 2,
        ),
        index
    ),

    'Β':lambda index, stacks: (
        AttrDict(
            call = str.replace,
            arity = 3,
        ),
        index
    ),

    'Γ':lambda index, stacks: (
        AttrDict(
            call = str.replace,
            arity = 4,
        ),
        index
    ),

    'Δ':lambda index, stacks: (
        AttrDict(
            call = str.rjust,
            arity = 2,
        ),
        index
    ),

    'Ε':lambda index, stacks: (
        AttrDict(
            call = str.rjust,
            arity = 3,
        ),
        index
    ),

    'Έ':lambda index, stacks: (
        AttrDict(
            call = str.rstrip,
            arity = 1,
        ),
        index
    ),

    'Ζ':lambda index, stacks: (
        AttrDict(
            call = str.rstrip,
            arity = 2,
        ),
        index
    ),

    'Η':lambda index, stacks: (
        AttrDict(
            call = str.split,
            arity = 1,
        ),
        index
    ),

    'Ή':lambda index, stacks: (
        AttrDict(
            call = str.split,
            arity = 2,
        ),
        index
    ),

    'Θ':lambda index, stacks: (
        AttrDict(
            call = str.startswith,
            arity = 2,
        ),
        index
    ),

    'Ι':lambda index, stacks: (
        AttrDict(
            call = str.startswith,
            arity = 3,
        ),
        index
    ),

    'Ί':lambda index, stacks: (
        AttrDict(
            call = str.startswith,
            arity = 4,
        ),
        index
    ),

    'Κ':lambda index, stacks: (
        AttrDict(
            call = str.strip,
            arity = 1,
        ),
        index
    ),

    'Λ':lambda index, stacks: (
        AttrDict(
            call = str.strip,
            arity = 2,
        ),
        index
    ),

    'Μ':lambda index, stacks: (
        AttrDict(
            call = str.upper,
            arity = 1,
        ),
        index
    ),

    'Ν':lambda index, stacks: (
        AttrDict(
            call = helpers.to_base,
            arity = 1,
        ),
        index
    ),

    'Ξ':lambda index, stacks: (
        AttrDict(
            call = helpers.duplicate,
            arity = 1,
            unpack = True,
        ),
        index
    ),

    'Ο':lambda index, stacks: (
        AttrDict(
            call = helpers.increment,
            arity = 1,
        ),
        index
    ),

    'Ό':lambda index, stacks: (
        AttrDict(
            call = helpers.decrement,
            arity = 1,
        ),
        index
    ),

    'Π':lambda index, stacks: (
        AttrDict(
            call = helpers.totient,
            arity = 1,
        ),
        index
    ),

    'Ρ':lambda index, stacks: (
        AttrDict(
            call = partial(digits(from_base), base = 2),
            arity = 1,
        ),
        index
    ),

    'Σ':lambda index, stacks: (
        AttrDict(
            call = partial(operator.sub, 1),
            arity = 1,
        ),
        index
    ),

    'Τ':lambda index, stacks: (
        AttrDict(
            call = digits(from_base),
            arity = 1,
        ),
        index
    ),

    'Υ':lambda index, stacks: (
        AttrDict(
            call = digits(from_base),
            arity = 2,
        ),
        index
    ),

    'Ύ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.all_equal),
            arity = 1,
        ),
        index
    ),

    'Φ':lambda index, stacks: (
        AttrDict(
            call = helpers.flatten,
            arity = 1,
        ),
        index
    ),

    'Χ':lambda index, stacks: (
        AttrDict(
            call = partial(operator.mul, 2),
            arity = 1,
        ),
        index
    ),

    'Ψ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.increments),
            arity = 1,
        ),
        index
    ),

    'Ω':lambda index, stacks: (
        AttrDict(
            call = helpers.isprime,
            arity = 1,
        ),
        index
    ),

    'Ώ':lambda index, stacks: (
        AttrDict(
            call = helpers.prime_product,
            arity = 1,
        ),
        index
    ),

    'α':lambda index, stacks: (
        AttrDict(
            call = helpers.factors,
            arity = 1,
        ),
        index
    ),

    'ά':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.factors, proper = True),
            arity = 1,
        ),
        index
    ),

    'β':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.factors, prime = True),
            arity = 1,
        ),
        index
    ),

    'γ':lambda index, stacks: (
        AttrDict(
            call = partargs(helpers.factors, {2: True, 3: True}),
            arity = 1,
        ),
        index
    ),

    'δ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.product),
            arity = 1,
        ),
        index
    ),

    'ε':lambda index, stacks: (
        AttrDict(
            call = helpers.if_statement,
            arity = 3,
        ),
        index
    ),

    'έ':lambda index, stacks: (
        AttrDict(
            call = helpers.if_statement,
            arity = 2,
        ),
        index
    ),

    'ζ':lambda index, stacks: (
        AttrDict(
            call = helpers.ifnot_statement,
            arity = 2,
        ),
        index
    ),

    'η':lambda index, stacks: (
        AttrDict(
            call = nilad(stacks[index].input),
            arity = 0,
        ),
        index
    ),

    'ή':lambda index, stacks: (
        AttrDict(
            call = nilad(stacks[index].input),
            arity = 0,
            unpack = True,
        ),
        index
    ),

    'θ':lambda index, stacks: (
        AttrDict(
            call = helpers.nfind,
            arity = 2,
        ),
        index
    ),

    'ι':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.nfind, tail = True),
            arity = 2,
        ),
        index
    ),

    'ί':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.nfind, head = True),
            arity = 2,
        ),
        index
    ),

    'ΐ':lambda index, stacks: (
        AttrDict(
            call = nilad(variables['x']),
            arity = 0,
        ),
        index
    ),

    'κ':lambda index, stacks: (
        AttrDict(
            call = partargs(helpers.assign, {1: variables, 2: 'x'}),
            arity = 1,
        ),
        index
    ),

    'λ':lambda index, stacks: (
        AttrDict(
            call = nest(sum, helpers.factors),
            arity = 1,
        ),
        index
    ),

    'μ':lambda index, stacks: (
        AttrDict(
            call = variables.get,
            arity = 1,
        ),
        index
    ),

    'ν':lambda index, stacks: (
        AttrDict(
            call = parargs(helpers.assign, {1: variables}),
            arity = 2,
        ),
        index
    ),

    'ξ':lambda index, stacks: (
        AttrDict(
            call = partargs(operator.mod, {2: 2}),
            arity = 1,
        ),
        index
    ),

    'ο':lambda index, stacks: (
        AttrDict(
            call = helpers.wrap,
            arity = 1,
        ),
        index
    ),

    'ό':lambda index, stacks: (
        AttrDict(
            call = ranges(helpers.product),
            arity = 1,
        ),
        index
    ),

    'π':lambda index, stacks: (
        AttrDict(
            call = partial(pow, 10),
            arity = 1,
        ),
        index
    ),

    'σ':lambda index, stacks: (
        AttrDict(
            call = partial(pow, 2),
            arity = 1,
        ),
        index
    ),

    'ς':lambda index, stacks: (
        AttrDict(
            call = nest(helpers.rle, helpers.prime_product),
            arity = 1,
        ),
        index
    ),

    'τ':lambda index, stacks: (
        AttrDict(
            call = digits(helpers.rle),
            arity = 1,
        ),
        index
    ),

    'υ':lambda index, stacks: (
        AttrDict(
            call = nest(partial(map, helpers.tail), helpers.rle, helpers.prime_product),
            arity = 1,
        ),
        index
    ),

    'ύ':lambda index, stacks: (
        AttrDict(
            call = partial(operator.mul, 2),
            arity = 1,
        ),
        index
    ),

    'ΰ':lambda index, stacks: (
        AttrDict(
            call = partial(helpers.nfind, helpers.isprime),
            arity = 1,
        ),
        index
    ),
    
}
# φχψωώ

test = True

if __name__ == '__main__' and test:
    code = 'ΰ'
    argv = [5]

    proc = Processor(code, argv, start = True)
    ret = proc.execute()
    out = proc.output(sep = '\n\n')
    
    if out:
        print(out)
    if out == code:
        print('', 'QUINE', sep = '\n')

if __name__ == '__main__' and not test:
    
    parser = argparse.ArgumentParser(prog = './orst')

    a = 'store_true'

    getcode = parser.add_mutually_exclusive_group()
    getcode.add_argument('-f', '--file', help = 'Specifies that code be read from a file', action = a)
    getcode.add_argument('-c', '--cmd', '--cmdline', help = 'Specifies that code be read from the command line', action = a)

    parser.add_argument('-u', '--unicode', help = 'Use Unicode encoding', action = a)

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

    if len(settings.argv) < 2:
        stks = 2
    else:
        stks = len(settings.argv)
        
    processor = Processor(code, settings.argv, stks)

        
