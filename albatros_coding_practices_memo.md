
Most of these are about readability. 

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

### Procedures

[Eamon, enlighten us here]


### Beautiful is better than ugly
Sometimes this means shorter is better than longer. 

<img width="867" alt="Screenshot 2023-08-07 at 11 58 13 AM" src="https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/6335d615-d3c7-4788-8256-83da0339af77">

It's sometimes useful to *explicitly* import submodules to accomplish beauty. 

<img width="692" alt="Screenshot 2023-08-07 at 11 56 28 AM" src="https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/7d23f5bf-681f-4bf6-9991-19b980b6506d">

But shorter doesn't always mean more beautiful. For instance, which of the two following snippets is easier to read?

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

This is a bit of a trick question. The former is easier to read but takes up much more space on the screen. You'll need to use your judgement.

However, in the former code snippet, a sin was committed. Can you spot it? 

<detials><summary>Answer</summary>

We hard-coded the list of squares in `is_square`, which means the function is deceptive and relies on us not looking beyond 10. We should either rename the function to something like `is_square_below_ten`, or at the least warn the reader in a comment. 

</details>

### Don't repeat yourself
If you find yourself copy-pasting something many times, you should probably put that into a function. 

### Consistency

Here's an example of inconsistent string formatting. 

![Screenshot 2023-07-20 at 6 22 40 PM](https://github.com/ALBATROS-Experiment/albatros_analysis/assets/21654151/79ed544f-2e79-4a2d-a56f-95e8ac5d24f2)

In the heat of a deployment mission, you may find yourself writing ugly code, and that's *fine*. However, if you see things can easily be tidied up to help future readers of your code have a better, smoother experience reading it, a little cleanup commit can go a long way. It will lower the cognitive load on anyone reading and editing your codebase.

Here's a short poem I wrote about my feelings on good coding practices:

However, I've also heard it said that "foolish consistency is the hobgoblin of little minds". Not sure what to make of that... 

## Documentation

Writing good documentation is as tricky as writing good scientific writing. You have to balance brevity and completeness. 

## Short poem

<i>
Every piece of code is not just a piece of code,<br/>
It's a piece of code plus one or several human brains<br/>
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

## Refs
- [pep 8 style guide](https://peps.python.org/pep-0008/)


