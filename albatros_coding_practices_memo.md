# Good coding practices memo

If you're feeling enthusiastic about writing clean Python, checkout the [pep 8 style guide](https://peps.python.org/pep-0008/).

NB, there are many valid reasons why you might not abide by the following code of conduct. However, always keep in mind that if you write clean code, manage your source control appropriately, and use environments properly, you will save people in the future, including you, *countless hours* of stress and confusion. Those people might be you. *Write code as though the person who will inherit your codebase is a violent psychopath and knows your address.*


**Pythonic Good Behaviour**
- No "copy/paste" programming.
- Refactor before things get ugly; avoid repeating the same thing (with small variations), including between scripts (not just within files)
- No "magic numbers", i.e. literal constants, except the one time you assign it to a variable. (At the *very least*, add a comment! When you use magic numbers so that people cleaning up your mess later on know what it's supposed to do.)
- Python does not support named constants, but if you are to name a variable that is not supposed to change, the Pythonic convention is to capitalize that variable name. That way, the reader knows the value is supposed to be constant and unchanging. *Never overwrite a capitalized variable after it's been initialized.*
- Comments, but not useless ones. In the heat of the moment, it's better to over-comment or comment badly than not comment at all.
- Virtual environments for library compatibility. It's easy not to bother setting up a virtual environment, but as the codebase grows and ages, it will eventually catch up, and you'll fall into a dependency conflict nightmare. Setting up python v-envs is actually very quick and painless once you get used to it.
- State machine coding: use state variables and a structured approach; avoid the "many randomly declared flags and variables" catastrophe
- Never idle-loop without a sleep (i.e. if you are waiting for something, insert a sleep)
- Tests. Tests are *really important*. Unit tests and integration tests.
- Variable names. The length of a variable name should give the reader a rough indication of its scope.
- Python is not strongly typed, but python3 does support type-hinting. If you write a function with many arguments or with potentially vague input & output types, please either use the type hinting or mention the types in a docstring. 
- Readability is extremely important. All of the above good practices are good because they improve readability. So aesthetics and variable naming conventions count too. It's important to be concise so as not to take up too much screen real estate, but it's also important not to name important variables
- Length of line: If your lines are much longer than 80 characters, think about whether it needs to be and try to make them shorter. (Black will do this for you when it can by splitting up your line into multiple lines, but this takes up more space & screen real estate too! Sometimes this is unnecessary because you can re-word it to make it more concise.)
- Length of function: Your function shouldn't be 500 lines long. A few hundred at most. Always aim for legibility. 

**Source control**
- Don't commit and push to the master/main branch. Make commits on a separate branch, submit a pull request (PR), and ask someone to review your changes before merging.
- Informative & concise commit messages.
- Reasonably scoped commits. i.e. don't push a commit that edits every file, and don't commit the deletion of two white spaces in one file just to clean your staging area (that's what the `--amend` option is for `commit -m 'message' --amend`)
- Don't *ever* force a push unless you're certain that the commit(s) you're over-writing are your own *and* that you're up to date with the remote branch. 
- Don't push large files (e.g. images, videos, jupyter notebooks, binaries); this will slow everyone down when they pull/merge/rebase.
- Know when to rebase and when to merge a commit. Rebase makes your commit history cleaner; merge preserves more information. If in doubt, it's safer to merge; however (in my workflow), rebasing is more common.



## Examples

(1) Explicit, type-hinted, properly-named variables, well documented.
```python
import numpy as np
import os
PATH_TO_BASEBAND_DATA_2022 = '/path/to/corr/data/2022'

def specific_operation(
        arr1 : np.ndarray,
        arr2 : np.ndarray,
) -> np.ndarray:
    """Description of the function.

    Detailed, multiline description of what the function does.

    Parameters
    ----------
    arr1 : np.ndarray
        Explain arr1
    arr2 : np.ndarray
        Explain arr2

    Returns
    -------
    np.ndarray
        Explain output. 
    """
    # main function body here
    for i in range(4):
        # small loop here
    return arr_out

for fname in os.listdir(PATH_TO_BASEBAND_DATA_2022):
    # load file
    specific_operation(arr1, arr2)
```

Implicit, no typing, poorly chosen variable names, no documentation, constant not capitalized. 
```python
import numpy
import os
path = '/path/to/corr/data/2022'

def f(a,b):
    # main function body here
    for iteration_number_count_xyz in range(4):
        # small loop here
    return arr_out

for x in os.listdir(path):
    # load file
    specific_operation(arr1, arr2)
```



(2) To make things more beautiful...

<img width="867" alt="Screenshot 2023-08-07 at 11 58 13 AM" src="https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/6335d615-d3c7-4788-8256-83da0339af77">

It's sometimes worth *explicitly* importing submodules to accomplish beauty. 

<img width="692" alt="Screenshot 2023-08-07 at 11 56 28 AM" src="https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/7d23f5bf-681f-4bf6-9991-19b980b6506d">

(3) Shorter doesn't always imply more beautiful. For instance, which of the two following snippets are easier to read?

```python
def func(i: int) -> float:
    out = i*i + 5         # square it and add five
    out = np.sqrt(out/10) # div by 10 and take the square root
    return out            # return

def is_sqaure(i: int) -> bool:
    if i in (1,4,9):      # all the squares under 10
        return True
    return False

my_list = []
for i in range(0,10, 2):  # iterate through even numbers
    if is_square(i):      # if square add it to list
        my_list.append(func(i))
```

or 

```python
my_list = [np.sqrt((i**2+5)/10) for i in range(10,2) if np.sqrt(i)==int(np.sqrt(i))]
```

This is a bit of a trick question. The former is easier to read but takes up much more space on the screen. You'll need to use your judgement to decide which is more appropriate.

In the former code snippet, a sin was committed. Can you spot it? 

<details><summary><b>Answer</b></summary>

We hard-coded the list of squares in `is_square`, which means the function is deceptive and relies on us not looking beyond 10. We should either rename the function to something like `is_square_below_ten`, or at the least warn the reader in a comment. 
</details>

## Documentation

Writing good documentation is as tricky as writing good scientific writing. You have to balance brevity and completeness. Writing helps you organize and improve your thoughts. Similarly, writing good documentation helps you organize and improve your code. 


## Whimsical Stuff

Here's a short poem about writing python, most of the stuff applies to all code. 

*Beautiful is better than ugly.<br/>
Explicit is better than implicit.<br/>
Simple is better than complex.<br/>
Complex is better than complicated.<br/>
Flat is better than nested.<br/>
Sparse is better than dense.<br/>
Readability counts.<br/>
Special cases aren't special enough to break the rules.<br/>
Although practicality beats purity.<br/>
Errors should never pass silently.<br/>
Unless explicitly silenced.<br/>
In the face of ambiguity, refuse the temptation to guess.<br/>
There should be one-- and preferably only one --obvious way to do it.<br/>
Although that way may not be obvious at first unless you're Dutch.<br/>
Now is better than never.<br/>
Although never is often better than *right* now.<br/>
If the implementation is hard to explain, it's a bad idea.<br/>
If the implementation is easy to explain, it may be a good idea.<br/>
Namespaces are one honking great idea -- let's do more of those!*

Another, poem penned by yours truly:

<i>
Every piece of code is more than just a piece of code,<br/>
It's a piece of code plus several human brains<br/>
That understand how it works, that's the mode, <br/>
If you are one of those human brains, <br/>
It's up to you to transmit the knowledge <br/>
That's in your head to future human brains <br/>
That need to run or read or rework your code.<br/>
<br/>
A piece of code is a living organism,<br/>
It's like a tree or a Chrysanthemum,<br/>
You plant the seed and watch it grow,<br/>
If you don't take good care of it, it will wither,<br/>
And eventually die of sorrow,<br/>
Someone else will have to come hither,<br/>
And plant a seed anew, or at great pains,<br/>
Resuscitate your woeful brain-child.<br/>
</i>
