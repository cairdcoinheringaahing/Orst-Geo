import itertools
import operator
import sys

import orst

def parse(code):
    perm = int(code.split('\n')[0])
    code = '\n'.join(code.split('\n')[1:])
    lines = ['']
    count = 4
    nl = code.count('\n') + 1

    while nl % 4:
        code += '\n'
        nl += 1
    code = code.rstrip()

    for ln in code.split('\n'):
        count -= 1
        lines[-1] += ln + '\n'

        if count == 0:
            lines.append('')
            count = 4

    return perm, lines

def integers(lines):
    ret = []
    for mul, line in enumerate(lines):
        line = line.split('\n')[:-1]
        while len(line) < 4:
            line.append('')
        line = sum(list(map(list, zip(*map(str.ljust, line, itertools.repeat(20))))), [])
        chars = sorted(set(line))

        if len(chars) > 2:
             print('Only two distinct characters may be used in the grid', file = sys.stderr)
             sys.exit(1)

        ret.extend(list(map(chars.index, line)))

    return list(filter(None, itertools.starmap(operator.mul, enumerate(ret, 1))))

def to_int(array):
    total = p = 0
    for elem in array[::-1]:
        total += elem * 2320 ** p
        p += 1
    return total

def to_array(num):
    digits = []
    while num:
        num, mod = divmod(num, 512)
        digits.append(mod)
    return digits

def run_orst(array, argv):
    code = ''
    for num in array:
        code += orst.code_page[num]

    proc = orst.Processor(code, argv, max(2, len(argv)))
    ret = proc.execute()
    out = proc.output(sep = '\n\n')

    return out

def main(code, argv):
    p, code = parse(code)
    perm_gen = itertools.permutations(integers(code))
    for _ in range(p + 1):
        ints = next(perm_gen)

    return run_orst(to_array(to_int(ints)), argv)

if __name__ == '__main__':
    code = sys.argv[1]
    try:
        code = open(code).read()
    except:
        pass
    argv = sys.argv[2:]
    print(main(code, argv))
