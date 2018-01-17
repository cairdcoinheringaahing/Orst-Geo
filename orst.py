import operator
import re

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
ώ1234567890.-" ¶
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
°◊•—/\:;”“„‚'’‘`
^*+=,_|~<≤«·»≥>ᴇ
ˋˆ¨´ˇ∞≠†∂ƒ∆¬≈µﬁﬂ
⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾×
₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎÷'''.replace('\n', '')

def overload(arguments, type_mapping, ordered = True):
    types = tuple(map(type, arguments))
    if not ordered: valid = any(t in type_mapping.keys() for t in itertools.permutations(types))
    else: valid = types in type_mapping.keys()
    
    if valid:
        return type_mapping[types](*arguments)
    if 'default' in type_mapping.keys():
        return type_mapping['default'](*arguments)

def repeat(block, integer):
    if type(block) == Block:
        for _ in range(integer):
            block.call()

def tokeniser(string):
    regex = r'''^(-?\d+)(((\.)|(E-?)|(j-?))(?(5)$|(\d+(\.\d+)?)))?$'''
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
        return final
        
    tokens = re.findall(regex, string)
    if len(tokens) == 1:
        return (
                    (
                        [tokens[0][0], tokens[0][1][0], tokens[0][1][1:]]
                        if
                            tokens[0][1][0] != '.'
                        else
                            [tokens[0][0] + tokens[0][1][0] + tokens[0][1][1:]]
                        )
                ) if tokens[0][1] else list(tokens[0][:1])
                    
    return [tokens[0][0] + '.' + tokens[1][0]] + ([tokens[1][1][0], tokens[1][1][1:]] if tokens[1][1] else [])

class AttrDict:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

class Block:
    def __init__(self, code):
        self.code = Processor(code[1:-1])

    def call(self, stacks):
        self.code.load_stacks(stacks)
        return self.code.execute()

class Clean:
    def __init__(self, code, stackcount):
        self.code = Processor(code[1:-1], stacks = stackcount)

    def call(self):
        return self.code.execute()

class Processor:
    def __init__(self, code, inputs = None, variables = None, stacks = 2):
        if inputs is None:
            inputs = []
        if variables is None:
            variables = {}

        self.varaibles = variables
        self.stacks = [Stack(inputs or None) for _ in range(stacks)]
        self.index = 0

    def load_stacks(self, array):
        while len(self.stacks) < len(array):
            self.stacks.append(Stack())
        while len(self.stacks) > len(array):
            array.append([])
            
        for i, values in enumerate(array):
            self.stacks[i].push(*values)

    def execute(self):

        scollect = ''
        string = 0
        
        acollect = ''
        array = 0

        bcollect = ''
        block = 0

        ccollect = ''
        clean = 0

        for char in tokeniser(code):
            # Handle the numeric types
            if (len(char) > 1 or char.isdigit()) and not (scollect or acollect or bcollect):
                self.stacks[self.index].push(eval(char))
                continue

            # The four syntax types: string ( " ), array ( [] ), block ( {} ) and clean ( () )
            if char == '"':
                string ^= 1
                if not string:
                    self.stacks[self.index].push(scollect)
                    scollect = ''
                continue

            if char == '[':
                array += 1
            if char == ']':
                array -= 1
                if array == 0:
                    self.stacks[self.index].push(eval(acollect + ']'))
                    acollect = ''

            if char == '{':
                block += 1
            if char == '}':
                block -= 1
                if block == 0:
                    self.stacks[self.index].push(Block(bcollect))
                    bcollect = ''

            if char == '(':
                clean += 1
            if char == ')':
                clean -= 1
                if clean == 0:
                    self.stacks[self.index].push(Clean(ccollect, self.stacks[self.index].pop()))
                    ccollect = ''

            # Time to collect the characters needed in the syntax
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

            # Now, move on to the actual commands
            cmd = commands[char] if char in commands else AttrDict(call = lambda z: z, arity = 1)
            arguments = self.stacks[self.index].pop(cmd.arity)
            self.stacks[self.index].push(cmd.call(*arguments))

        # Garbage collection for uncollected syntax types
        for collect in [scollect, acollect, bcollect]:
            if collect:
                self.stacks[self.index].push(collect)

        return list(filter(lambda a: list(filter(None, a)), self.stacks))

class Stack(list):
    def __init__(self, starters=None):
        if starters is None:
            starters = [0]
        self.input = starters
        
    def push(self, *values):
        for value in values:
            self.append(value)

    def peek(self, index=-1):
        return self[index]

    def pop(self, count=1, indexes=(-1,) ):
        popped = []
        next_in = 0
        for _ in range(count):
            try:
                popped.append(super().pop(indexes[_] if _ < len(indexes) else -1))
            except IndexError:
                popped.append(self.input[next_in % len(self.input)])
                next_in += 1
        return popped

commands = {

    'A':AttrDict(
            arity = 2,
            call = lambda x, y: overload((x, y), {
                    (Block, int): repeat,
                    (Clean, int): repeat,
                    'default'   : operator.add,
                })
    ),

}

if __name__ == '__main__':
    code = input('Enter code: ')
    inputs = eval(input('Enter the input list: '))
    variables = eval(input('Enter the variables dictionary: '))
    stacks = int(input('Enter number of stacks: '))
    process = Processor(code, inputs, variables, stacks)
    results = process.execute()
    print(results)

            
