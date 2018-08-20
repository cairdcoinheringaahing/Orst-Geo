Orst and Geo are two languages very closely related to one another, and so have been included in this repo together

# Orst

The standard Hello, World! program is simply

    "Hello, World!"
    
However, we can shorten this by using Orst's builtin string compression with

    "BΡÁζŠ(ﬁ“
    
## Invocation

If the code is held in a file, `prog`, use the following invocation to fun the program

    $ python3 orst.py --file --unicode prog
    
With any additional inputs added to the end of the call. Otherwise, the code can be run be replacing
`--file` with `--cmd` and directly passing the code, rather than a file.

# Geo

Geo stands for Grid Encoded Orst, and the name describes the language. Essentially, we have a grid of 2320 cells in fashion of a Leyes grid. For those who don't know what that is, imagine normal squared paper, and add 3 horizontal lines inside each square.

Each cell either contains a space or another character. This can be any other character than a space, but must be consistent throughtout the entire program. The indexes of cells containing the non-space character are recorded and used to base a base-2320 number (e.g. `[1, 2, 3] => 1 × 2320⁰ + 2 × 2320¹ + 3 × 2320² => 16151841`) This number is then encoded in base 512, and the corrosponding "digits" are used to index into the Orst code page. Finally, the resulting characters are concatenated and run as Orst.

In addition to this grid format, Geo also has a number at the start of the program. This number indicates which permutation of the indexes should be encoded in base-2320, as they are in ascending order by default, limiting the number of available programs.

## Invocation

Unlike Orst, Geo has no command line customisation and simply is run with the following command, assuming the file is in `prog`:

    $ python3 geo.py prog
    
You can also bypass the use of a file by simply inputting the code directly. However, programs tend to span many lines, so this isn't recommended. The standard Hello, World! program is quite long, weighing in at 1922 bytes:

```
3733
 O                  
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
      O             
                    
                   O
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
            O       
       O            
                    
                    
                    
                    
          O         
                    
                    
                    
          O         
                    
     O
```

## Orst to Geo

As the process of converting Orst code to Geo is very convoluted and close to impossible to do by hand for very long Orst programs, the file `orst2geo.py` takes a line of input, representing the Orst code to be run, and outputs the corrosponding Geo code
