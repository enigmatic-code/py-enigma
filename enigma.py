#!/usr/bin/env python -t
###############################################################################
#
# File:         enigma.py
# RCS:          $Header: $
# Description:  Useful routines for solving Enigma Puzzles
# Author:       Jim Randell
# Created:      Mon Jul 27 14:15:02 2009
# Modified:     Sat Aug 24 11:57:40 2019 (Jim Randell) jim.randell@gmail.com
# Language:     Python
# Package:      N/A
# Status:       Free for non-commercial use
#
# (c) Copyright 2009-2019, Jim Randell, all rights reserved.
#
###############################################################################
# -*- mode: Python; py-indent-offset: 2; -*-

"""
A collection of useful code for solving New Scientist Enigma (and similar) puzzles.

The latest version is available at <http://www.magwag.plus.com/jim/enigma.html>.

Currently this module provides the following functions and classes:

arg                    - extract an argument from the command line
args                   - extract a list of arguments from the command line
base2int               - convert a string in the specified base to an integer
base_digits            - get/set digits used in numerical base conversion
bit_permutations       - generate bit permutations
C                      - combinatorial function (nCk)
cached                 - decorator for caching functions
cbrt                   - the (real) cube root of a number
choose                 - choose a sequence of values satisfying some functions
chunk                  - go through an iterable in chunks
compare                - comparator function
concat                 - concatenate a list of values into a string
coprime_pairs          - generate coprime pairs
cslice                 - cumulative slices of an array
csum                   - cumulative sum
diff                   - sequence difference
digit_map              - create a map of digits to corresponding integer values
digrt                  - the digital root of a number
divc                   - ceiling division
divf                   - floor division
div                    - exact division
divisor                - generate the divisors of a number
divisor_pairs          - generate pairs of divisors of a number
divisors               - the divisors of a number
divisors_pairs         - generate pairs of divisors of a number
drop_factors           - reduce a number by removing factors
egcd                   - extended gcd
factor                 - the prime factorisation of a number
factorial              - factorial function
farey                  - generate Farey sequences of coprime pairs
fib                    - generate fibonacci sequences
filter2                - partition an iterator into values that satisfy a predicate, and those that do not
filter_unique          - partition an iterator into values that are unique, and those that are not
find                   - find the index of an object in an iterable
find_max               - find the maximum value of a function
find_min               - find the minimum value of a function
find_value             - find where a function has a specified value
find_zero              - find where a function is zero
first                  - return items from the start of an iterator
flatten                - flatten a list of lists
flattened              - fully flatten a nested structure
format_recurring       - output the result from recurring()
fraction               - convert numerator / denominator to lowest terms
gcd                    - greatest common divisor
grid_adjacency         - adjacency matrix for an n x m grid
hypot                  - calculate hypotenuse
icount                 - count the number of elements of an iterator that satisfy a predicate
int2base               - convert an integer to a string in the specified base
int2roman              - convert an integer to a Roman Numeral
int2words              - convert an integer to equivalent English words
intc                   - ceiling conversion of float to int
intf                   - floor conversion of float to int
invmod                 - multiplicative inverse of n modulo m
ipartitions            - partition a sequence with repeated values by index
irange                 - inclusive range iterator
iroot                  - integer kth root function
is_cube                - check a number is a perfect cube
is_distinct            - check a value is distinct from other values
is_duplicate           - check to see if value (as a string) contains duplicate characters
is_pairwise_distinct   - check all arguments are distinct
is_power               - check if n = i^k for some integer i
is_power_of            - check if n = k^i for some integer i
is_prime               - simple prime test
is_prime_mr            - Miller-Rabin fast prime test
is_roman               - check a Roman Numeral is valid
is_square              - check a number is a perfect square
is_triangular          - check a number is a triangular number
isqrt                  - intf(sqrt(x))
join                   - concatenate strings
lcm                    - lowest common multiple
M                      - multichoose function (nMk)
mgcd                   - multiple gcd
multiply               - the product of numbers in a sequence
nconcat                - concatenate single digits into an integer
nreverse               - reverse the digits in an integer
nsplit                 - split an integer into single digits
number                 - create an integer from a string ignoring non-digits
P                      - permutations function (nPk)
partitions             - partition a sequence of distinct values into tuples
pi                     - float approximation to pi
poly_*                 - routines manipulating polynomials, wrapped as Polynomial
prime_factor           - generate terms in the prime factorisation of a number
printf                 - print with interpolated variables
pythagorean_triples    - generate Pythagorean triples
reciprocals            - generate reciprocals that sum to a given fraction
recurring              - decimal representation of fractions
repeat                 - repeatedly apply a function to a value
repdigit               - number consisting of repeated digits
roman2int              - convert a Roman Numeral to an integer
split                  - split a value into characters
sprintf                - interpolate variables into a string
sqrt                   - the (positive) square root of a number
subseqs                - sub-sequences of an iterable
subsets                - generate subsequences of an iterator
substitute             - substitute symbols for digits in text
substituted_expression - a substituted expression (alphametic/cryptarithm) solver
substituted_sum        - a solver for substituted sums
T, tri                 - T(n) is the nth triangular number
tau                    - tau(n) is the number of divisors of n
timed                  - decorator for timing functions
timer                  - a Timer object
trirt                  - the (positive) triangular root of a number
tuples                 - generate overlapping tuples from a sequence
uniq                   - unique elements of an iterator
unpack                 - return a function that unpacks its arguments
update                 - return an updated copy of an object

Accumulator            - a class for accumulating values
CrossFigure            - a class for solving cross figure puzzles
Delay                  - a class for the delayed evaluation of a function
DominoGrid             - a class for solving domino grid puzzles
Football               - a class for solving football league table puzzles
MagicSquare            - a class for solving magic squares
Polynomial             - a class for manipulating Polynomials
Primes                 - a class for creating prime sieves
SubstitutedDivision    - a class for solving substituted long division sums
SubstitutedExpression  - a class for solving general substituted expression (alphametic/cryptarithm) problems
SubstitutedSum         - a class for solving substituted addition sums
Timer                  - a class for measuring elapsed timings
"""

# Python 3 style print() and division
from __future__ import print_function, division

__author__ = "Jim Randell <jim.randell@gmail.com>"
__version__ = "2019-08-24"

__credits__ = """Brian Gladman, contributor"""

import sys
import os

import operator
import math
import functools
import itertools
import collections
import copy

# maybe use the "six" module for some of this stuff
if sys.version_info[0] == 2:
  # Python 2.x
  if sys.version_info[1] < 7:
    print("[enigma.py] WARNING: Python {v} is very old. Things may not work.".format(v=sys.version.split(None, 1)[0]))
  _python = 2
  range = xrange
  reduce = reduce
  basestring = basestring
  raw_input = raw_input
  Sequence = collections.Sequence
elif sys.version_info[0] > 2:
  # Python 3.x
  _python = 3
  range = range
  reduce = functools.reduce
  basestring = str
  raw_input = input
  if sys.version_info[1] > 6:
    # Python 3.7
    # not: [[ Sequence = collections.abc.Sequence ]]
    from collections.abc import Sequence
  else:
    Sequence = collections.Sequence

# useful constants
enigma = sys.modules[__name__]
nl = "\n"
pi = math.pi
two_pi = 2.0 * pi
inf = float('+inf')

# add attributes to a function (to use as static variables)
# (but for better performance use global variables)
def static(**kw):
  def decorate(fn):
    for (k, v) in kw.items():
      setattr(fn, k, v)
    return fn
  return decorate

# useful routines for solving Enigma puzzles

# like cmp() in Python 2, but results are always -1, 0, +1.
# vs can be a triple of values to return instead, default corresponds to (-1, 0, +1)
def compare(a, b, vs=None):
  """
  return -1 if a < b, 0 if a == b and +1 if a > b.

  >>> compare(42, 0)
  1
  >>> compare(0, 42)
  -1
  >>> compare(42, 42)
  0
  >>> compare('evil', 'EVIL')
  1
  """
  r = (b < a) - (a < b)
  return (vs[r + 1] if vs else r)

# it's probably quicker (and shorter) to just use:
#   X not in args
# rather than:
#   is_distinct(X, args)
def is_distinct(value, *args):
  """
  check <value> is distinct from values in <args>

  >>> is_distinct(0, 1, 2, 3)
  True
  >>> is_distinct(2, 1, 2, 3)
  False
  >>> is_distinct('h', 'e', 'l', 'l', 'o')
  True
  """
  return value not in args

distinct = is_distinct

# <s> has <n> distinct values
def distinct_values(s, n=None):
  if n is None: n = len(s)
  return len(set(s)) == n

def seq_all_different(s):
  seen = set()
  for x in s:
    if x in seen: return False
    seen.add(x)
  return True

# same as distinct_values(args), or distinct_values(args, len(args))
def is_pairwise_distinct(*args):
  """
  check all arguments are pairwise distinct

  >>> is_pairwise_distinct(0, 1, 2, 3)
  True
  >>> is_pairwise_distinct(2, 1, 2, 3)
  False
  >>> is_pairwise_distinct('h', 'e', 'l', 'l', 'o')
  False

  """
  # this gives the same result as: distinct_values(args, None)
  # it's probably faster to use a builtin...
  #return len(set(args)) == len(args)
  ## even through the following may do fewer tests:
  #for i in range(len(args) - 1):
  #  if args[i] in args[i + 1:]: return False
  #return True
  return seq_all_different(args)

pairwise_distinct = is_pairwise_distinct

def seq_all_same(s, **kw):
  """
  >>> seq_all_same([1, 2, 3])
  False
  >>> seq_all_same([1, 1, 1, 1, 1, 1])
  True
  >>> seq_all_same([1, 1, 1, 1, 1, 1], value=4)
  False
  >>> seq_all_same(Primes(expandable=True))
  False
  """
  i = iter(s)
  try:
    v = kw['value']
  except KeyError:
    try:
      v = next(i)
    except StopIteration:
      return True
  return all(x == v for x in i)

# same as distinct_values(args, 1)
def all_same(*args):
  """
  check all arguments have the same value

  >>> all_same(1, 2, 3)
  False

  >>> all_same(1, 1, 1, 1, 1, 1)
  True

  >>> all_same()
  True
  """
  return seq_all_same(args)

all_different = is_pairwise_distinct

def ordered(*args, **kw):
  """
  return args as a tuple in order.

  this is useful for making a key for a dictionary.

  >>> ordered(2, 1, 3)
  (1, 2, 3)
  >>> ordered(2, 1, 3, reverse=1)
  (3, 2, 1)
  """
  return tuple(sorted(args, **kw))

# I would prefer join() to be a method of sequences: s.join(sep='')
# but for now we define a utility function
def join(s, sep=''):
  """
  join the items in sequence <s> as strings, separated by separator <sep>.

  the default separator is the empty string so you can just use:

    join(s)

  instead of:

    ''.join(s)

  >>> join(['a', 'b', 'cd'])
  'abcd'
  >>> join(['a', 'b', 'cd'], sep=',')
  'a,b,cd'
  >>> join([5, 700, 5])
  '57005'
  """
  return str.join(sep, (str(x) for x in s))

def concat(*args, **kw):
  """
  return a string consisting of the concatenation of the elements of
  the sequence <args>. the elements will be converted to strings
  (using str(x)) before concatenation.

  you can use it instead of str.join() to join non-string lists by
  specifying a 'sep' argument.

  >>> concat('h', 'e', 'l', 'l', 'o')
  'hello'
  >>> concat(1, 2, 3, 4, 5)
  '12345'
  >>> concat(1, 2, 3, 4, 5, sep=',')
  '1,2,3,4,5'
  """
  sep = kw.get('sep', '')
  if len(args) == 1:
    try:
      return str.join(sep, (str(x) for x in args[0]))
    except TypeError:
      pass
    except:
      raise
  return str.join(sep, (str(x) for x in args))


def nconcat(*digits, **kw):
  """
  return an integer consisting of the concatenation of the list
  <digits> of single digits

  the digits can be specified as individual arguments, or as a single
  argument consisting of a sequence of digits.

  >>> nconcat(1, 2, 3, 4, 5)
  12345
  >>> nconcat([1, 2, 3, 4, 5])
  12345
  >>> nconcat(13, 14, 10, 13, base=16)
  57005
  >>> nconcat(123,456,789, base=1000)
  123456789
  >>> nconcat([127, 0, 0, 1], base=256)
  2130706433
  """
  # in Python3 [[ def nconcat(*digits, base=10): ]] is allowed instead
  base = kw.get('base', 10)
  fn = lambda x: reduce(lambda a, b: a * base + b, x, 0)
  if len(digits) == 1:
    try:
      return fn(digits[0])
    except TypeError:
      pass
    except:
      raise
  return fn(digits)
  # or: (slower, and only works with digits < 10)
  #return int(concat(*digits), base=base)

def nsplit(n, k=None, base=10):
  """
  split an integer into digits (using base <base> representation)

  if <k> is specified it gives the number of digits to return, if the
  number has too few digits the the result is zero padded at the beginning,
  if the number has too many digits then the result includes only the
  rightmost digits.

  the sign of the integer is ignored.

  >>> nsplit(12345)
  (1, 2, 3, 4, 5)
  >>> nsplit(57005, base=16)
  (13, 14, 10, 13)
  >>> nsplit(123456789, base=1000)
  (123, 456, 789)
  >>> nsplit(2130706433, base=256)
  (127, 0, 0, 1)
  >>> nsplit(7, 3)
  (0, 0, 7)
  >>> nsplit(111 ** 2, 3)
  (3, 2, 1)
  """
  if n < 0: n = -n
  ds = list()
  while True:
    (n, r) = divmod(n, base)
    ds.insert(0, r)
    if k is None:
      if n == 0: break
    else:
      if k < 2: break
      k -= 1
  return tuple(ds)


def nreverse(n, base=10):
  """
  reverse an integer (using base <base> representation)

  >>> nreverse(12345)
  54321
  >>> nreverse(-12345)
  -54321
  >>> nreverse(0xedacaf, base=16) == 0xfacade
  True
  >>> nreverse(100)
  1
  """
  if n < 0:
    return -nreverse(-n, base=base)
  else:
    return nconcat(reversed(nsplit(n, base=base)), base=base)

from fnmatch import fnmatch

# match a value (as a string) to a template
def match(v, t):
  """
  match a value (as a string) to a template (see fnmatch.fnmatch).

  to make matching numbers easier if the template starts with a minus
  sign ('-') then so must the value. if the template starts with a
  plus sign ('+') then the value must not start with a minus sign. so
  to match a 4-digit positive number use '+????' as a template.

  >>> match("abcd", "?b??")
  True
  >>> match("abcd", "a*")
  True
  >>> match("abcd", "?b?")
  False
  >>> match(1234, '+?2??')
  True
  >>> match(-1234, '-??3?')
  True
  >>> match(-123, '???3')
  True
  """
  v = str(v)
  if t.startswith('-'):
    if v.startswith('-'):
      (v, t) = (v[1:], t[1:])
    else:
      return False
  elif t.startswith('+'):
    if v.startswith('-'):
      return False
    else:
      t = t[1:]
  return fnmatch(v, t)

def number(s, base=10):
  """
  make an integer from a string, ignoring non-digit characters

  >>> number('123,456,789')
  123456789
  >>> number('100,000,001')
  100000001
  >>> number('-1,024')
  -1024
  >>> number('DEAD.BEEF', base=16) == 0xdeadbeef
  True
  """
  return base2int(s, base=base, strip=1)


def split(x, fn=None):
  """
  split <x> into characters (which may be subsequently post-processed by <fn>).

  >>> split('hello')
  ['h', 'e', 'l', 'l', 'o']
  >>> split(12345)
  ['1', '2', '3', '4', '5']
  >>> list(split(12345, int))
  [1, 2, 3, 4, 5]
  """
  return list(map(fn, str(x))) if fn else list(str(x))


# or you can use itertools.izip_longest(*[iter(l)]*n) for padded chunks
def chunk(s, n=2, fn=tuple):
  """
  iterate through iterable <s> in chunks of size <n>.

  (for overlapping tuples see tuples())

  >>> list(chunk(irange(1, 8)))
  [(1, 2), (3, 4), (5, 6), (7, 8)]
  >>> list(chunk(irange(1, 8), 3))
  [(1, 2, 3), (4, 5, 6), (7, 8)]
  """
  i = iter(s)
  while True:
    s = fn(itertools.islice(i, 0, n))
    if not s: break
    yield s


def diff(a, b, *rest):
  """
  return the subsequence of <a> that excludes elements in <b>.

  >>> diff((1, 2, 3, 4, 5), (3, 5, 2))
  (1, 4)
  >>> join(diff('newhampshire', 'wham'))
  'nepsire'
  """
  if rest: b = set(b).union(*rest)
  return tuple(x for x in a if x not in b)


# unique combinations:
# like uniq(combinations(s, k) but more efficient
def uC(s, k):
  if k == 0:
    yield ()
  else:
    seen = set()
    for (i, x) in enumerate(s):
      if x not in seen:
        for t in uC(s[i + 1:], k - 1): yield (x,) + t
        seen.add(x)

def ucombinations(s, k=None):
  s = list(s)
  if k is None: k = len(s)
  return uC(s, k)

# a (partial) multiset implementation with len() that counts the number of elements
# the multiset is implemented as a dict mapping <item> -> <count>
# it can be used as an alternative to collections.Counter
class multiset(dict):

  # the multiset s is passed in as one of:
  #  a dict of <item> -> <count> values
  #  a sequence of (<item>, <count>) values
  #  a sequence of <item> values
  def __init__(self, v=None, **kw):
    dict.__init__(self)
    if v is None:
      pass
    elif isinstance(v, dict):
      # from a dict
      for (x, n) in v.items(): self.add(x, n)
    else:
      # from a sequence
      for x in v:
        try:
          self.add(*x)
        except TypeError:
          self.add(x)
    # add in any keyword items
    for (x, n) in kw.items(): self.add(x, n)

  # count all elements in the multiset
  # (for number of unique elements use: [[ len(s.keys()) ]])
  def __len__(self):
    return sum(self.values())

  # all elements of the multiset
  # (for unique elements use: [[ s.keys() ]])
  def __iter__(self):
    for (k, v) in self.items():
      for _ in range(v):
        yield k

  # add an item
  def add(self, item, count=1):
    try:
      count += self[item]
      if count == 0:
        del self[item]
        return
    except KeyError:
      pass
    if count < 0: raise ValueError(sprintf("negative count: {item} -> {count}"))
    if count > 0: self[item] = count

  # remove an item
  def remove(self, item, count=1):
    if count != 0:
      self.add(item, -count)

  # like self.items(), but in value order
  def most_common(self, n=None):
    s = sorted(self.items(), key=lambda t: t[::-1], reverse=True)
    return (s if n is None else s[:n])

  # provide some useful operations on multisets

  # update self with some other multisets (item counts are summed)
  def update(self, *rest):
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      for (item, count) in m.items(): self.add(item, count)
    return self

  # combine self and some other multisets (item counts are summed)
  def combine(self, *rest):
    return multiset(self).update(*rest)

  # union update of self and some other multiset (maximal item counts are retained)
  def union_update(self, *rest):
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      for (item, count) in m.items(): self[item] = max(count, self.get(item, 0))
    return self

  # union of self and some other multiset (maximal item counts are retained)
  def union(self, *rest):
    return multiset(self).union_update(*rest)

  # intersection of self and some other multisets (minimal item counts are retained)
  def intersection(self, *rest):
    r = multiset(self)
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      r = multiset((item, min(count, r.get(item, 0))) for (item, count) in m.items())
    return r

  # differences between self and m
  # return (self - m, m - self)
  def differences(self, m):
    if not isinstance(m, dict): m = multiset(m)
    (d1, d2) = (multiset(), multiset())
    for item in set(self.keys()).union(m.keys()):
      count = self.get(item, 0) - m.get(item, 0)
      if count > 0:
        d1.add(item, count)
      elif count < 0:
        d2.add(item, -count)
    return (d1, d2)

  # difference between self and m
  # (m may contain items that are not in self, they are ignored)
  def difference(self, m):
    return self.differences(m)[0]

  # is multiset m a subset of self?
  def issuperset(self, m):
    (d1, d2) = self.differences(m)
    return not d2

  # is multiset m a superset of self?
  def issubset(self, m):
    (d1, d2) = self.differences(m)
    return not d1

  # absolute difference in item counts of the two multisets
  def symmetric_difference(self, m):
    (d1, d2) = self.differences(m)
    return d1.update(d2)

def mcombinations(s, k=None):
  s = sorted(multiset(s))
  if k is None: k = len(s)
  return uC(s, k)

# multiset permutations:
# a bit like uniq(permutations(s, k)) but more efficient, however items
# will not be generated in the same order
#
# there are more sophisticated algorithms, but this one does the job:
#
#  >>> with Timer(): icount(uniq(subsets("mississippi", select="P")))
#  107899
#  [timing] elapsed time: 68.9407666s (68.94s)
#
#  >>> with Timer(): icount(subsets("mississippi", select="mP"))
#  107899
#  [timing] elapsed time: 0.5661372s (566.14ms)
#
def mP(d, n):
  if n == 0:
    yield ()
  else:
    for (k, v) in d.items():
      if v > 0:
        d[k] -= 1
        for t in mP(d, n - 1): yield (k,) + t
        d[k] += 1

def mpermutations(s, k=None):
  s = multiset(s)
  if k is None: k = len(s)
  return mP(s, k)


# subsets (or subseqs) wraps various itertools methods (which can save an import)
@static(select_fn=dict(), prepare_fn=dict())
def subsets(s, size=None, min_size=0, max_size=None, select='C', prepare=None):
  """
  generate tuples representing the subsequences of a (finite) iterator.

  'min_size' and 'max_size' can be used to limit the size of the
  subsequences or 'size' can be specified to produce subsequences of a
  particular size.

  the way the elements of the subsequences are selected can be
  controlled with the 'select' parameter:
     'C' = combinations (default),
     'P' = permutations,
     'R' = combinations with replacement,
     'M' = product,
     'uC' = unique combinations
     'mC' = multiset combinations
     'mP' = multiset permutations
  or you can provide your own function.

  aliases: subseqs(), powerset().

  >>> list(subsets((1, 2, 3)))
  [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

  >>> list(subsets((1, 2, 3), size=2, select='C'))
  [(1, 2), (1, 3), (2, 3)]

  >>> list(subsets((1, 2, 3), size=2, select='P'))
  [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

  >>> list(subsets((1, 2, 3), size=2, select='R'))
  [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

  >>> list(subsets((1, 2, 3), size=2, select='M'))
  [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
  """
  if prepare is None: prepare = (list if callable(select) else subsets.prepare_fn.get(select, list))
  s = prepare(s)
  # choose appropriate size parameters
  if size is not None:
    if callable(size): size = size(s)
    min_size = max_size = size
  elif max_size is None:
    max_size = len(s)
  else:
    if callable(min_size): min_size = min_size(s)
    if callable(max_size): max_size = max_size(s)
  # choose an appropriate select function
  if not callable(select): select = subsets.select_fn[select]
  # generate the subsets
  for k in irange(min_size, max_size):
    for x in select(s, k): yield tuple(x)

# provide selection functions (where available)
# [[ maybe 'R' should be 'M', and 'M' should be 'X' ]]
for (k, v, p) in (
    ('C', getattr(itertools, 'combinations', None), None),
    ('P', getattr(itertools, 'permutations', None), None),
    ('R', getattr(itertools, 'combinations_with_replacement', None), None),
    ('M', (lambda fn: ((lambda s, k: fn(s, repeat=k)) if fn else None))(getattr(itertools, 'product', None)), None),
    ('uC', uC, None),
    ('mC', uC, (lambda s: sorted(multiset(s)))),
    ('mP', mP, multiset),
  ):
  if v:
    subsets.select_fn[k] = v
    setattr(subsets, k, v)
    if p: subsets.prepare_fn[k] = p

# aliases
powerset = subsets
subseqs = subsets


# like filter() but also returns the elements that don't satisfy the predicate
# see also partition() recipe from itertools documentation
# (but note that itertools.partition() returns (false, true) lists)
def filter2(p, i, fn=list):
  """
  use a predicate to partition an iterable into those elements that
  satisfy the predicate, and those that do not.

  alias: partition()

  >>> filter2(lambda n: n % 2 == 0, irange(1, 10))
  ([2, 4, 6, 8, 10], [1, 3, 5, 7, 9])
  """
  t = list((x, p(x)) for x in i)
  return (fn(x for (x, v) in t if v), fn(x for (x, v) in t if not v))

# alias if you prefer the term partition (but don't confuse it with partitions())
partition = filter2

def identity(x):
  """
  the identity function: identity(x) == x
  """
  return x

def filter_unique(s, f=identity, g=identity):
  """
  for every object, x, in sequence, s, consider the map f(x) -> g(x)
  and return a partition of s into those objects where f(x) implies a
  unique value for g(x), and those objects where f(x) implies multiple
  values for g(x).

  returns the partition of the original sequence as
  (<unique values>, <non-unique values>)

  See: Enigma 265 <https://enigmaticcode.wordpress.com/2015/03/14/enigma-265-the-parable-of-the-wise-fool/#comment-4167>

  alias: partition_unique()

  "If I told you the first number you could deduce the second"
  >>> filter_unique([(1, 1), (1, 3), (2, 1), (3, 1), (3, 2), (3, 3)], (lambda v: v[0]))[0]
  [(2, 1)]

  "If I told you the first number you could not deduce if the second was odd or even"
  >>> filter_unique([(1, 1), (1, 3), (2, 1), (3, 1), (3, 2), (3, 3)], (lambda v: v[0]), (lambda v: v[1] % 2))[1]
  [(3, 1), (3, 2), (3, 3)]
  """
  u = collections.defaultdict(set)
  r = collections.defaultdict(list)
  for x in s:
    i = f(x)
    u[i].add(g(x))
    r[i].append(x)
  (r1, r2) = ([], [])
  for (k, v) in u.items():
    (r1 if len(v) == 1 else r2).extend(r[k])
  return (r1, r2)

# alias if you prefer the term partition (but don't confuse it with partitions())
partition_unique = filter_unique


def unpack(fn):
  """
  Turn a function that takes multiple parameters into a function that
  takes a tuple of those parameters.

  To some extent this can be used to work around the removal of
  parameter unpacking in Python 3 (PEP 3113):

  In Python 2.7 we could write:

    > fn = lambda (x, y): is_square(x * x + y * y)
    > fn((3, 4))
    5

  but this is not allowed in Python 3.

  Instead we can write:

    > fn = unpack(lambda x, y: is_square(x * x + y * y))
    > fn((3, 4))
    5

  to provide the same functionality.

  >>> fn = unpack(lambda x, y: is_square(x * x + y * y))
  >>> list(filter(fn, [(1, 2), (2, 3), (3, 4), (4, 5)]))
  [(3, 4)]
  """
  return lambda args: fn(*args)


# count the number of occurrences of a predicate in an iterator
def icount(i, p=None, t=None):
  """
  count the number of elements in iterator <i> that satisfy predicate <p>,
  the termination limit <t> controls how much of the iterator we visit,
  so we don't have to count all occurrences.

  So, to find if exactly <n> elements of <i> satisfy <p> use:

  icount(i, p, n + 1) == n

  which is what icount_exactly(i, p, n) does.

  This will examine all elements of <i> to verify there are exactly 4 primes
  less than 10:
  >>> icount_exactly(irange(1, 10), is_prime, 4)
  True

  But this will stop after testing 73 (the 21st prime):
  >>> icount_exactly(irange(1, 100), is_prime, 20)
  False

  To find if at least <n> elements of <i> satisfy <p> use:

  icount(i, p, n) == n

  This is what icount_at_least(i, p, n) does.

  The following will stop testing at 71 (the 20th prime):
  >>> icount_at_least(irange(1, 100), is_prime, 20)
  True

  To find if at most <n> elements of <i> satisfy <p> use:

  icount(i, p, n + 1) < n + 1

  This is what icount_at_most(i, p, n) does.

  The following will stop testing at 73 (the 21st prime):
  >>> icount_at_most(irange(1, 100), is_prime, 20)
  False

  If p is not specified a function that always returns True is used,
  so you can use this function to count the number of items in a (finite) iterator:

  >>> icount(Primes(1000))
  168

  """
  if p is None:
    if t is None:
      if hasattr(i, '__len__'):
        return len(i)
      else:
        # a quick way to count an iterable
        d = collections.deque(enumerate(i, start=1), maxlen=1)
        return (d[0][0] if d else 0)
    else:
      p = (lambda x: True)
  n = 0
  for x in i:
    if p(x):
      n += 1
      if n == t: break
  return n

# icount recipes
icount_exactly = lambda i, p, n: icount(i, p, n + 1) == n
icount_at_least = lambda i, p, n: icount(i, p, n) == n
icount_at_most = lambda i, p, n: icount(i, p, n + 1) < n + 1

# find: like index(), but return -1 instead of throwing an error
def find(s, v):
  """
  find the first index of a value in a sequence, return -1 if not found.

  for a string it works like str.find() (for single characters)
  >>> find('abc', 'b')
  1
  >>> find('abc', 'z')
  -1

  but it also works on lists
  >>> find([1, 2, 3], 2)
  1
  >>> find([1, 2, 3], 7)
  -1

  and on iterators in general (don't try this with a non-prime value)
  >>> find(Primes(), 10007)
  1229

  Note that this function works by attempting to use the index() method
  of the sequence. If it implements index() in a non-compatible way
  this function won't work.
  """
  try:
    return s.index(v)
  except ValueError:
    return -1
  except AttributeError:
    pass
  for (i, x) in enumerate(s):
    if x == v: return i
  return -1


def _partitions(s, n):
  """
  partition a sequence <s> of distinct elements into subsequences of length <n>.

  <s> should be sequenceable type (tuple, list, str).
  <n> should be a factor of the size of the sequence.

  >>> list(_partitions((1, 2, 3, 4), 2))
  [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
  """
  if not(len(s) > n):
    yield (s,)
  else:
    for x in itertools.combinations(s[1:], n - 1):
      p = (s[0],) + tuple(x)
      for ps in _partitions(diff(s[1:], x), n):
        yield (p,) + ps


def ipartitions(s, n):
  """
  partition a sequence by index.

  >>> list(ipartitions((1, 0, 1, 0), 2))
  [((1, 0), (1, 0)), ((1, 1), (0, 0)), ((1, 0), (0, 1))]
  """
  for p in _partitions(tuple(range(len(s))), n):
    yield tuple(tuple(s[i] for i in x) for x in p)


def partitions(s, n, pad=0, value=None, distinct=None):
  """
  partition a sequence <s> into subsequences of length <n>.

  if <pad> is True then the sequence will be padded (using <value>)
  until it's length is a integer multiple of <n>.

  if sequence <s> contains distinct elements then <distinct> can be
  set to True, if it is not set then <s> will be examined for repeated
  elements.

  >>> list(partitions((1, 2, 3, 4), 2))
  [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
  """
  if not isinstance(s, (tuple, list, str)): s = tuple(s)
  (d, r) = divmod(len(s), n)
  if r != 0:
    if not pad: raise ValueError("invalid sequence length {l} for {n}-tuples".format(l=len(s), n=n))
    s = tuple(s) + (value,) * (n - r)
  if d == 0 or (d == 1 and r == 0):
    yield (s,)
  else:
    if distinct is None: distinct = is_pairwise_distinct(*s)
    fn = (_partitions if distinct else ipartitions)
    # or in Python 3: [[ yield from fn(s, n) ]]
    for p in fn(s, n): yield p


# see: [ https://enigmaticcode.wordpress.com/2017/05/17/tantalizer-482-lapses-from-grace/#comment-7169 ]
# choose: choose values from <vs> satisfying <fns> in turn
# distinct - true if values must be distinct
# s - initial sequence (that supports 'copy()' and 'append()')
def choose(vs, fns, s=None, distinct=0):
  """
  choose values from <vs> satisfying <fns> in turn.

  if all values are acceptable then a value of None can be passed in <fns>.

  set 'distinct' if all values should be distinct.

  >>> list(choose([1, 2, 3], [None, (lambda a, b: abs(a - b) == 1), (lambda a, b, c: abs(b - c) == 1)]))
  [[1, 2, 1], [1, 2, 3], [2, 1, 2], [2, 3, 2], [3, 2, 1], [3, 2, 3]]
  """
  if s is None: s = list()
  # are we done?
  if not fns:
    yield s
  else:
    # choose the next value
    fn = fns[0]
    for v in vs:
      if not(distinct) or v not in s:
        s1 = copy.copy(s)
        s1.append(v)
        if fn is None or fn(*s1):
          # choose the rest [[Python 3: yield from ...]]
          for z in choose(vs, fns[1:], s1, distinct): yield z


def first(i, count=1, skip=0, fn=list):
  """
  return the first <count> items in iterator <i> (skipping the initial
  <skip> items) as a list.

  this would be a way to find the first 10 primes:
  >>> first((n for n in irange(1, inf) if is_prime(n)), count=10)
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  """
  r = itertools.islice(i, skip, skip + count)
  return (r if fn is None else fn(r))


def repeat(fn, v=0):
  """
  generate repeated applications of function <fn> to value <v>.

  >>> first(repeat(lambda x: x + 1), 10)
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  """
  while True:
    yield v
    v = fn(v)

def uniq(i, fn=None):
  """
  generate unique values from iterator <i> (maintaining order).

  >>> list(uniq([5, 7, 0, 0, 5]))
  [5, 7, 0]
  >>> list(uniq('mississippi'))
  ['m', 'i', 's', 'p']
  >>> list(uniq([1, 2, 3, 4, 5, 6, 7, 8, 9], fn=lambda x: x % 3))
  [1, 2, 3]
  """
  seen = set()
  for x in i:
    r = (x if fn is None else fn(x))
    if r not in seen:
      yield x
      seen.add(r)

def uniq1(i, fn=None):
  """
  generate unique consecutive values from iterator <i> (maintaining
  order), where values are compared using <fn>.

  this function assumes that common elements are generated in <i>
  together, so it only needs to track the last value.

  essentially it collapses repeated consecutive values to a single
  value.

  >>> list(uniq1((1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5)))
  [1, 2, 3, 4, 5]
  >>> list(uniq1('mississippi'))
  ['m', 'i', 's', 'i', 's', 'i', 'p', 'i']
  """
  i = iter(i)
  x = next(i)
  yield x
  seen = (x if fn is None else fn(x))
  for x in i:
    r = (x if fn is None else fn(x))
    if r != seen:
      yield x
      seen = r

# we use math.pow in sqrt(), cbrt() rather than ** to avoid generating complex numbers

_F13 = 1.0 / 3.0

def cbrt(x):
  """
  Return the cube root of a number (as a float).

  >>> cbrt(27.0)
  3.0
  >>> cbrt(-27.0)
  -3.0
  """
  return (-math.pow(-x, _F13) if x < 0 else math.pow(x, _F13))

def prime_factor(n):
  """
  generate (<prime>, <exponent>) pairs in the prime factorisation of positive integer <n>.

  no pairs are returned for 1 (or for non-positive integers).

  >>> list(prime_factor(60))
  [(2, 2), (3, 1), (5, 1)]
  >>> list(prime_factor(factorial(12)))
  [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
  """
  # to test the correct domain of the function we could use:
  #if not(isinstance(n, numbers.Integral) and n > 0): raise ValueError("expecting positive integer")
  if n > 1:
    i = 2
    # generate a list of deltas: 1, 2, then 2, 4, repeatedly
    (ds, js) = ((1, 2, 2, 4), (1, 2, 3, 2))
    j = 0
    while i * i <= n:
      e = 0
      while True:
        (d, m) = divmod(n, i)
        if m > 0: break
        e += 1
        n = d
      if e > 0: yield (i, e)
      i += ds[j]
      j = js[j]
    # anything left is prime
    if n > 1: yield (n, 1)

# maybe should be called factors() or factorise()
def factor(n, fn=prime_factor):
  """
  return a list of the prime factors of positive integer <n>.

  for integers less than 1, None is returned.

  The <fn> parameter is used to generate the prime factors of the
  number. (Defaults to using prime_factor()).

  >>> factor(101)
  [101]
  >>> factor(1001)
  [7, 11, 13]
  >>> factor(12)
  [2, 2, 3]
  >>> factor(125)
  [5, 5, 5]
  """
  if n < 1: return None
  factors = []
  for (p, e) in fn(n):
    factors.extend([p] * e)
  return factors

# divsors are based on:
#
#  (a, b) is a divisor pair of n iff: a, b in [0, n] and a.b = n
#
# the set of divisors is the set of numbers that appear in the divisor pairs
#
# and these are returned in order,
#
# so:
#
#  divisor pairs 6 = (1, 6) (2, 3); divisors 6 = (1, 2, 3, 6)
#  divisor pairs 4 = (1, 4) (2, 2); divisors 4 = (1, 2, 4)
#  divisor pairs 2 = (1, 2); divisors 2 = (1, 2)
#  divisor pairs 1 = (1, 1); divisors 1 = (1)
#  divisor pairs 0 = (0, 0); divisors 0 = (0)

def divisor_pairs(n):
  """
  generate divisors (a, b) of positive integer n, such that a =< b and a * b = n.

  the pairs are generated in order of increasing <a>.

  if you only want a few small divisors, this routine is OK, otherwise
  you are probably better using divisors_pairs().

  >>> list(divisor_pairs(36))
  [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6)]
  >>> list(divisor_pairs(101))
  [(1, 101)]
  """
  if n == 0:
    yield (0, 0)
    return
  a = 1
  while True:
    (b, r) = divmod(n, a)
    if a > b: break
    if r == 0: yield (a, b)
    a += 1

def divisor(n):
  """
  generate divisors of positive integer <n> in numerical order.

  >>> list(divisor(36))
  [1, 2, 3, 4, 6, 9, 12, 18, 36]
  >>> list(divisor(101))
  [1, 101]
  """
  bs = list()
  for (a, b) in divisor_pairs(n):
    yield a
    if a < b:
      bs.insert(0, b)
    else:
      break
  for b in bs:
    yield b


def multiples(ps):
  """
  given a list of (<m>, <n>) pairs, return all numbers that can be formed by multiplying
  together the <m>s, with each <m> occurring up to <n> times.

  the multiples are returned as a sorted list

  the practical upshot of this is that the divisors of a number <x> can be found using
  the expression: multiples(prime_factor(x))

  >>> multiples(prime_factor(180))
  [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180]
  """
  s = [1]
  for (m, n) in ps:
    t = list()
    p = m
    for _ in range(n):
      t.extend(x * p for x in s)
      p *= m
    s.extend(t)
  s.sort()
  return s


def divisors(n, fn=prime_factor):
  """
  return the divisors of positive integer <n> as a sorted list.

  >>> divisors(36)
  [1, 2, 3, 4, 6, 9, 12, 18, 36]
  >>> divisors(101)
  [1, 101]
  """
  if n == 0: return [0]
  return multiples(fn(n))


def divisors_pairs(n, fn=prime_factor):
  """
  generate divisors pairs (a, b) with a =< b, such that a * b = n.

  pairs are generated in order, by determining the factors of n.

  this is probably faster than divisor_pairs() if you want all divisors.
  """
  if n == 0:
    yield (0, 0)
    return
  for a in divisors(n, fn=fn):
    b = n // a
    if a > b: break
    yield (a, b)


def is_prime(n):
  """
  return True if the positive integer <n> is prime.

  Note: for numbers up to 2^64 is_prime_mr() is a fast, accurate prime
  test. (And for larger numbers it is probabilistically accurate).

  >>> is_prime(101)
  True
  >>> is_prime(1001)
  False

  """
  for (p, e) in prime_factor(n):
    return p == n
  return False

prime = is_prime


# Miller-Rabin primality test
# modified from a contribution by Brian Gladman

import random

def _is_composite(a, d, n, s):
  if a == 0: return 0
  x = pow(a, d, n)
  if x == 1:
    return 0
  for _ in range(s):
    if x == n - 1:
      return 0
    x = (x * x) % n
  # definitely composite
  return 1

def is_prime_mr(n, r=0):
  """
  Miller-Rabin primality test for <n>.
  <r> is the number of random extra rounds performed for large numbers

  return value:
    0 = the number is definitely not prime (definitely composite for n > 1)
    1 = the number is probably prime
    2 = the number is definitely prime

  for numbers less than 2^64, the prime test is completely accurate,
  and deterministic, the extra rounds are not performed.

  for larger numbers <r> additional rounds are performed, and if the
  number cannot be found to be composite a value of 1 (probably prime)
  is returned. confidence can be increased by using more additional
  rounds.

  >>> is_prime_mr(288230376151711813)
  2
  >>> is_prime_mr(316912650057057350374175801351)
  1
  >>> is_prime_mr(332306998946228968225951765070086171)
  0
  """
  # 0, 1 = not prime
  if n < 2:
    return 0

  # 2, 3 = definitely prime
  if n < 4:
    return 2

  # all other primes have a residue mod 6 of 1 or 5
  x = n % 6
  if x != 1 and x != 5:
    return 0

  # compute 2^s.d = n - 1
  d = n - 1
  s = (d & -d).bit_length() - 1
  d >>= s

  # bases from: https://miller-rabin.appspot.com/
  # we use 3 sets of bases:
  # 1 base = [9345883071009581737] is completely accurate for n < 341531 (about 2^18)
  # 2 bases = [336781006125, 9639812373923155] (2 bases) is completely accurate for n < 1050535501 (about 2^30)
  # 7 bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022] is completely accurate for n < 2^64

  # 1 base is completely accurate for n < 341531
  if n < 341531:
    return (0 if _is_composite(9345883071009581737 % n, d, n, s) else 2)

  # 2 bases are completely accurate for n < 1050535501
  if n < 1050535501:
    return (0 if _is_composite(336781006125 % n, d, n, s) or _is_composite(9639812373923155 % n, d, n, s) else 2)

  # test remaining numbers with the 7 base set
  if any(_is_composite(a % n, d, n, s) for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022)):
    # definitely composite
    return 0

  # the 7 base set is completely accurate for n < 2^64:
  if n < 0x10000000000000000:
    # definitely prime
    return 2

  # for larger numbers run further prime tests as specified
  if r > 0 and any(_is_composite(random.randrange(2, n - 1), d, n, s) for _ in range(r)):
    # definitely composite
    return 0

  # otherwise, probably prime
  return 1


def tau(n, fn=prime_factor):
  """
  count the number of divisors of a positive integer <n>.

  tau(n) = len(divisors(n)) (but faster)

  >>> tau(factorial(12))
  792
  """
  return multiply(e + 1 for (_, e) in fn(n))


def is_square_free(n, fn=prime_factor):
  """
  a positive integer is "square free" if it is not divisibly by
  a perfect square greater than 1.

  >>> is_square_free(8596)
  False
  >>> is_square_free(8970)
  True
  """
  return n > 0 and all(e == 1 for (_, e) in fn(n))


def farey(n):
  """
  generate the Farey sequence F(n) - the sequence of coprime
  pairs (a, b) where 0 < a < b =< n. pairs are generated
  in numerical order when considered as fractions a/b.

  the pairs (0, 0) and (1, 1) usually present at the start
  and end of the sequence are not generated by this function.

  >>> list(p for p in farey(20) if sum(p) == 20)
  [(1, 19), (3, 17), (7, 13), (9, 11)]
  """
  (a, b, c, d) = (0, 1, 1, n)
  while d > 1:
    k = (n + b) // d
    (a, b, c, d) = (c, d, k * c - a, k * d - b)
    yield (a, b)

def coprime_pairs(n=None, order=0):
  """
  generate coprime pairs (a, b) with 0 < a < b =< n.

  the list is complete and no element appears more than once.

  if n is not specified then pairs will be generated indefinitely.

  if n is specified then farey() can be used instead to generate
  coprime pairs in numerical order (when considered as fractions).

  if order=1 is specified then the pairs will be produced in order.

  >>> sorted(p for p in coprime_pairs(20) if sum(p) == 20)
  [(1, 19), (3, 17), (7, 13), (9, 11)]
  >>> list(coprime_pairs(6, order=1))
  [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5), (1, 6), (5, 6)]
  """
  fn = ((lambda p: p[0] <= n) if n else (lambda p: True))
  if order:
    # use a heap to order the pairs
    from heapq import heapify, heappush, heappop
    ps = list()
    heapify(ps)
    _push = heappush
    _pop = heappop
  else:
    # just use a list
    ps = list()
    _push = lambda ps, p: ps.append(p)
    _pop = lambda ps: ps.pop(0)
  for p in ((2, 1), (3, 1)):
    if fn(p): _push(ps, p)
  while ps:
    (b, a) = _pop(ps)
    yield (a, b)
    for p in ((2 * b - a, b), (2 * a + b, a), (2 * b + a, b)):
      if fn(p): _push(ps, p)

# Pythagorean Triples:
# see: https://en.wikipedia.org/wiki/Formulas_for_generating_Pythagorean_triples

# generate primitive pythagorean triples (x, y, z) with hypotenuse not exceeding Z
# if Z is None, then triples will be generated indefinitely
# if order is True, then triples will be returned in order
def _pythagorean_primitive(Z=None, order=0):
  fn = ((lambda z: z <= Z) if Z is not None else (lambda z: True))
  if order:
    # use a heap
    from heapq import heapify, heappush, heappop
    ts = list()
    heapify(ts)
    _push = heappush
    _pop = heappop
  else:
    # just use a list
    ts = list()
    _push = lambda s, t: s.append(t)
    _pop = lambda s: s.pop(0)
  # initial triple
  if fn(5): _push(ts, (5, 4, 3))
  while ts:
    (c, b, a) = _pop(ts)
    yield (a, b, c)
    # my original formulation (using only addition/subtraction)
    (a2, b2, c2) = (a + a, b + b, c + c)
    c3 = c2 + c
    for (z, y, x) in (
      (c3 + b2 - a2, c2 + b - a2, c2 + b2 - a),
      (c3 + b2 + a2, c2 + b + a2, c2 + b2 + a),
      (c3 - b2 + a2, c2 - b + a2, c2 - b2 + a),
    ):
      if fn(z): _push(ts, ((z, x, y) if y < x else (z, y, x)))
    ## alternatively: brian's (more compact) formulation
    #t = 2 * (a + b + c)
    #(u, v, w) = (t - 4 * b, t, t - 4 * a)
    #for (z, y, x) in ((u + c, u + b, u - a), (v + c, v - b, v - a), (w + c, w - b, w + a)):
    #  if fn(z): _push(ts, ((z, x, y) if y < x else (z, y, x)))

# generate pythagorean triples (x, y, z) with hypotenuse not exceeding Z
def _pythagorean_all(Z, order=0):
  if order:
    # use a heap to save the multiples
    from heapq import heapify, heappush, heappop
    ms = list()
    heapify(ms)
    for (x, y, z) in _pythagorean_primitive(Z, order=1):
      # return any saved multiples less than (x, y, z)
      while ms and ms[0] < (z, y, x):
        yield heappop(ms)[::-1]
      # return (x, y, z)
      yield (x, y, z)
      # add in any new multiples
      for k in irange(2, Z // z):
        heappush(ms, (k * z, k * y, k * x))
    # return any remaining multiples
    while ms:
      yield heappop(ms)[::-1]
  else:
    # return the multiples with the primitives
    for (x, y, z) in _pythagorean_primitive(Z, order=0):
      for k in irange(1, Z // z):
        yield (k * x, k * y, k * z)

# generate pythagorean triples
# n - specifies the maximum hypotenuse allowed
# primitive - if set only primitive triples are generated
# order - if set triples are generated in order
# if primitive is False, then a value for n must be specified
def pythagorean_triples(n=None, primitive=0, order=0):
  """
  generate pythagorean triples (x, y, z) where x < y < z and x^2 + y^2 = z^2.

  n - maximum allowed hypotenuse (z)
  primitive - if set only primitive triples are generated
  order - if set triples are generated in order

  order is by shortest z, then shortest y, then shortest x
  (i.e. reverse lexicographic)

  if 'primitive' is set, then n can be None, and primitive triples
  will be generated indefinitely (although it will eventually run out
  of memory)

  >>> list(pythagorean_triples(20, primitive=0, order=1))
  [(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (12, 16, 20)]

  >>> list(pythagorean_triples(20, primitive=1, order=1))
  [(3, 4, 5), (5, 12, 13), (8, 15, 17)]

  >>> icount(pythagorean_triples(10000, primitive=1))
  1593

  >>> icount(pythagorean_triples(10000, primitive=0))
  12471
  """
  if primitive:
    # primitive only triples
    return _pythagorean_primitive(n, order)
  else:
    # include non-primitive
    assert n is not None
    return _pythagorean_all(n, order)


def fib(*s, **kw):
  """
  generate Fibonacci type sequences.

  The initial k terms are provided as sequence s, subsequent terms are
  calculated as a function of the preceeding k terms.

  The default function being 'sum', but a different function can be
  specified using the 'fn' parameter (which should be a function that
  takes a sequence of k terms and computes the appropriate value).

  Standard Fibonacci numbers (OEIS A000045):
  >>> first(fib(0, 1), 10)
  [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

  Lucas numbers (OEIS A000032):
  >>> first(fib(2, 1), 10)
  [2, 1, 3, 4, 7, 11, 18, 29, 47, 76]

  Tribonacci numbers (OEIS A001590):
  >>> first(fib(0, 1, 0), 10)
  [0, 1, 0, 1, 2, 3, 6, 11, 20, 37]

  Powers of 2 (using addition):
  >>> first(fib(1, fn=unpack(lambda x: x + x)), 10)
  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  """
  fn = kw.get('fn', sum)
  s = list(s)
  while True:
    s.append(fn(s))
    yield s.pop(0)


# if we don't overflow floats (happens around 2**53) this works...
#   def is_power(n, m):
#     i = int(n ** (1.0 / m) + 0.5)
#     return (i if i ** m == n else None)
# but here we use a binary search, which should work on arbitrary large integers
#
# NOTE: that this will return 0 if n = 0 and None if n is not a perfect k-th power,
# so [[ power(n, k) ]] will evaluate to True only for positive n
# if you want to allow n to be 0 you should check: [[ power(n, k) is not None ]]

def iroot(n, k):
  """
  compute the largest integer x such that x^k =< n.

  i.e. x is the integer k-th root of n.

  it is the exact root if: (x ** k == n)
  (which is what is_power() does)
  """
  # binary search
  if n < 0 or k < 1: return
  if n >> k == 0: return int(n > 0)
  a = 1 << ((n.bit_length() - 1) // k)
  b = a << 1
  #assert (a ** k <= n and b ** k > n)
  # if this assertion fails we need:
  #while not(b ** k > n): (a, b) = (b, b << 1)
  while b - a > 1:
    r = (a + b) // 2
    x = r ** k
    if x < n:
      a = r
    elif x > n:
      b = r
    else:
      return r
  return a


def is_power(n, k):
  """
  check positive integer <n> is a perfect <k>th power of some integer.

  if <n> is a perfect <k>th power, returns the integer <k>th root.
  if <n> is not a perfect <k>th power, returns None.

  >>> is_power(49, 2)
  7
  >>> is_power(49, 3) is not None
  False
  >>> is_power(0, 2)
  0
  >>> n = (2 ** 60 + 1)
  >>> (is_power(n ** 2, 2) is not None, is_power(n ** 2 + 1, 2) is not None)
  (True, False)
  >>> (is_power(n ** 3, 3) is not None, is_power(n ** 3 + 1, 3) is not None)
  (True, False)
  """
  r = iroot(n, k)
  return (r if r ** k == n else None)


def sqrt(a, b=None):
  """
  the (real) square root of a / b (or just a if b is None)

  >>> sqrt(9)
  3.0
  >>> sqrt(9, 4)
  1.5
  """
  # / is operator.truediv() here
  return math.sqrt(a if b is None else a / b)


# calculate intf(sqrt(n))
# see: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Binary_numeral_system_.28base_2.29
def isqrt(n):
  """
  calculate intf(sqrt(n)).

  >>> isqrt(9)
  3
  >>> isqrt(15)
  3
  >>> isqrt(16)
  4
  >>> isqrt(17)
  4
  """
  if n < 0: return None
  if n < 4: return int(n > 0)

  r = 0
  k = n.bit_length() - 2
  b = 1 << (k + (k & 1))

  #assert not(b > n)
  ## if this assertion fails we need
  #while b > n: b >>= 2

  while b:
    if n >= r + b:
      n -= r + b
      r = (r >> 1) + b
    else:
      r >>= 1
    b >>= 2

  return r

# it would be more Pythonic to encapsulate is_square in a class with the initialisation
# in __init__, and the actual call in __call__, and then instantiate an object to be
# the is_square() function (i.e. [[ is_square = _is_square_class(80) ]]), but it is
# more efficient (and perhaps more readable) to just use normal variables, although
# if you're using PyPy the class based version is just as fast (if not slightly faster)
@static(mod=None, residues=None, reject=None)
def is_square(n):
  """
  check positive integer <n> is a perfect square.

  if <n> is a perfect square, returns the integer square root.
  if <n> is not a perfect square, returns None.

  >>> is_square(49)
  7
  >>> is_square(50) is not None
  False
  >>> is_square(0)
  0
  """
  if n < 0: return None
  if n < 2: return n
  # early rejection: check <square> mod <some value> against a precomputed cache
  # e.g. <square> mod 80 = 0, 1, 4, 9, 16, 20, 25, 36, 41, 49, 64, 65 (rejects 88% of numbers)
  if is_square.reject[n % is_square.mod]: return None
  # otherwise use isqrt and check the result
  r = isqrt(n)
  return (r if r * r == n else None)

# experimentally 80, 48, 72, 32 are good values (24, 16 also work OK)
is_square.mod = 80
is_square.residues = set((i * i) % is_square.mod for i in range(is_square.mod))
is_square.reject = list(i not in is_square.residues for i in range(is_square.mod))

def is_not_none(fn):
  """
  turn a function into a predicate that is the equivalent of:

    fn(<args>) is not None

  >>> is_square(0)
  0
  >>> is_square(0) == False
  True
  >>> p = is_not_none(is_square)
  >>> p(0)
  True
  """
  return (lambda *args, **kw: fn(*args, **kw) is not None)

is_square_p = is_not_none(is_square)

def is_cube(n):
  """
  check positive integer <n> is a perfect cube.

  >>> is_cube(27)
  3
  >>> is_cube(49) is not None
  False
  >>> is_cube(0)
  0
  """
  return is_power(n, 3)

is_cube_p = is_not_none(is_cube)


# keep the old names as aliases
power = is_power
cube = is_cube
square = is_square


def drop_factors(n, k):
  """
  remove factors of <k> from <n>.

  return (i, m) where n = (m)(k^i) such that m is not divisible by k
  """
  i = 0
  while n > 1:
    (d, r) = divmod(n, k)
    if r != 0: break
    i += 1
    n = d
  return (i, n)

def is_power_of(n, k):
  """
  check <n> is a power of <k>.

  returns <m> such that pow(k, m) = n or None.

  >>> is_power_of(128, 2)
  7
  >>> is_power_of(1, 2)
  0
  >>> is_power_of(0, 2) is None
  True
  >>> is_power_of(0, 0)
  1
  """
  if n == 0: return (1 if k == 0 else None)
  if n == 1: return 0
  if k < 2: return None
  (i, m) = drop_factors(n, k)
  return (i if m == 1 else None)


def T(n):
  """
  T(n) is the nth triangular number.

  T(n) = n * (n + 1) // 2.

  >>> T(1)
  1
  >>> T(100)
  5050
  """
  return n * (n + 1) // 2

tri = T


def trirt(x):
  """
  return the triangular root of <x> (as a float)

  >>> trirt(5050)
  100.0
  >>> round(trirt(2), 8)
  1.56155281
  """
  return 0.5 * (sqrt(8 * x + 1) - 1.0)


def is_triangular(n):
  """
  check positive integer <n> is a triangular number.

  if <n> is a triangular number, returns <i> such that T(i) == n.
  if <n> is not a triangular number, returns None.

  >>> is_triangular(5050)
  100
  >>> is_triangular(49) is not None
  False
  """
  i = int(trirt(n) + 0.5)
  return (i if i * (i + 1) == 2 * n else None)


def digrt(n, base=10):
  """
  return the digital root of positive integer <n>.

  >>> digrt(123456789)
  9
  >>> digrt(sum([1, 2, 3, 4, 5, 6, 7, 8, 9]))
  9
  >>> digrt(factorial(100))
  9
  """
  return (0 if n == 0 else int(1 + (n - 1) % (base - 1)))


def repdigit(n, d=1, base=10):
  """
  return a number consisting of the digit <d> repeated <n> times, in base <base>

  default digit is d=1 (to return repunits)
  default base is base=10

  >>> repdigit(6)
  111111
  >>> repdigit(6, 7)
  777777
  >>> repdigit(6, 7, base=16)
  7829367
  >>> repdigit(6, 7, base=16) == 0x777777
  True
  """
  assert 0 <= d < base
  return d * (base ** n - 1) // (base - 1)


def hypot(*v):
  """
  return hypotenuse of a right angled triangle with shorter sides <a> and <b>.

  hypot(a, b) = sqrt(a**2 + b**2)

  >>> hypot(3.0, 4.0)
  5.0
  """
  return sqrt(sum(x * x for x in v))


def intf(x):
  """
  floor conversion (largest integer not greater than x) of float to int.

  >>> intf(1.0)
  1
  >>> intf(1.5)
  1
  >>> intf(-1.5)
  -2
  """
  return int(x) if x > 0 or x.is_integer() else int(x) - 1

floor = intf


def intc(x):
  """
  ceiling conversion (smallest integer not less than x) of float to int.

  >>> intc(1.0)
  1
  >>> intc(1.5)
  2
  >>> intc(-1.5)
  -1
  """
  return int(x) if x < 0 or x.is_integer() else int(x) + 1

ceil = intc


def divf(a, b):
  """
  floor division. (equivalent to Python's a // b).

  >>> divf(100, 10)
  10
  >>> divf(101, 10)
  10
  >>> divf(101.1, 10)
  10.0
  >>> divf(4.5, 1)
  4.0
  """
  return a // b


def divc(a, b):
  """
  ceiling division.

  >>> divc(100, 10)
  10
  >>> divc(101, 10)
  11
  >>> divc(101.1, 10)
  11.0
  >>> divc(4.5, 1)
  5.0
  """
  return -(-a // b)

cdiv = divc


def div(a, b):
  """
  returns (a // b) if b exactly divides a, otherwise None

  >>> div(1001, 13)
  77
  >>> bool(div(101, 13))
  False
  >>> div(42, 0) is None
  True
  """
  if b == 0: return None
  (d, r) = divmod(a, b)
  if r != 0: return None
  return d

is_multiple = div

def fdiv(a, b):
  """
  float result of <a> divided by <b>.

  >>> fdiv(3, 2)
  1.5

  >>> fdiv(9, 3)
  3.0
  """
  return float(a) / float(b)



def is_duplicate(*s):
  """
  check to see if arguments (as strings) contain duplicate characters.

  >>> is_duplicate("hello")
  True
  >>> is_duplicate("world")
  False
  >>> is_duplicate(99 ** 2)
  False
  """
  s = join(s)
  return len(set(s)) != len(s)
  # or using regexps
  #return True if re.search(r'(.).*\1', str(s)) else False

duplicate = is_duplicate

def multiply(s):
  """
  return the product of the sequence <s>.

  >>> multiply(range(1, 7))
  720
  >>> multiply([2] * 8)
  256
  """
  return reduce(operator.mul, s, 1)

# product is the original name for multiply, but it's been renamed
# to avoid name clashes with itertools.product.
product = multiply


def gcd(a, b):
  """
  greatest common divisor (on positive integers).

  >>> gcd(123, 456)
  3
  >>> gcd(5, 7)
  1
  """
  while b:
    (a, b) = (b, a % b)
  return a


def lcm(a, b):
  """
  lowest common multiple (on positive integers).

  >>> lcm(123, 456)
  18696
  >>> lcm(5, 7)
  35
  """
  return (a // gcd(a, b)) * b


# Extended Euclidean Algorithm
def egcd(a, b):
  """
  Extended Euclidean Algorithm (on positive integers).

  returns integers (x, y, g) = egcd(a, b) where ax + by = g = gcd(a, b)

  Note that x and y are not necessarily positive integers.

  >>> egcd(120, 23)
  (-9, 47, 1)
  """
  ## recursively...
  #if b == 0: return (1, 0, a)
  #(q, r) = divmod(a, b)
  #(s, t, g) = egcd(b, r)
  #return (t, s - q * t, g)
  #
  # or iteratively...
  (x0, x1) = (1, 0)
  (y0, y1) = (0, 1)
  while b:
    (q, r) = divmod(a, b)
    (a, b, x0, x1, y0, y1) = (b, r, x1, x0 - q * x1, y1, y0 - q * y1)
  return (x0, y0, a)

# multiplicative inverse mod m
def invmod(n, m):
  """
  return the multiplicative inverse of n mod m (or None if there is no inverse)

  e.g. the inverse of 2 (mod 9) is 5, as (2 * 5) % 9 = 1
  >>> invmod(2, 9)
  5
  """
  (x, y, g) = egcd(n, m)
  return ((x % m) if g == 1 else None)

# multiple GCD
def mgcd(a, *rest):
  """
  GCD of multiple (two or more) integers.

  >>> mgcd(123, 456)
  3
  >>> mgcd(123, 234, 345, 456, 567, 678, 789)
  3
  >>> mgcd(11, 37, 228)
  1
  >>> mgcd(56, 65, 671)
  1
  """
  return reduce(gcd, rest, a)

# multiple LCM
def mlcm(a, *rest):
  """
  LCM of multiple (two or more) integers.

  >>> mlcm(2, 3, 5, 9)
  90
  """
  return reduce(lcm, rest, a)

def is_coprime(a, b):
  return gcd(a, b) == 1


# for those times when fractions.Fraction is overkill
def fraction(a, b):
  """
  return the numerator and denominator of the fraction a/b in lowest terms

  >>> fraction(286, 1001)
  (2, 7)
  """
  g = gcd(a, b)
  return (a // g, b // g)


def factorial(a, b=1):
  """
  return a!/b!.

  >>> factorial(6)
  720
  >>> factorial(10, 7)
  720
  """
  r = math.factorial(a)
  return (r if b == 1 else r // math.factorial(b))


def P(n, k):
  """
  permutations functions: n P k.

  the number of ordered k-length selections from n elements
  (elements can only be used once).

  >>> P(10, 3)
  720
  """
  if k > n:
    return 0
  else:
    return math.factorial(n) // math.factorial(n - k)


def C(n, k):
  """
  combinatorial function: n C k.

  the number of unordered k-length selections from n elements
  (elements can only be used once).

  >>> C(10, 3)
  120
  """
  if k > n:
    return 0
  else:
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

# NOTE: this corresponds to [[ select='R' ]] in subsets(), not [[ select='M' ]]
def M(n, k):
  """
  multichoose function: n M k.

  the number of unordered k-length selections from n elements where
  elements may be repeated.

  >>> M(10, 3)
  220
  """
  return C(n + k - 1, k)


def recurring(a, b, recur=0, base=10, digits=None):
  """
  find recurring representation of the fraction <a> / <b> in the specified base.
  return strings (<integer-part>, <non-recurring-part>, <recurring-part>)
  if you want rationals that normally terminate represented as non-terminating set <recur>

  >>> recurring(1, 7)
  ('0', '', '142857')
  >>> recurring(3, 2)
  ('1', '5', '')
  >>> recurring(3, 2, recur=1)
  ('1', '4', '9')
  >>> recurring(5, 17, base=16)
  ('0', '', '4B')
  """
  # check input fraction
  if b == 0 or (recur and a == 0):
    raise ValueError("invalid input fraction: {a} / {b} [recur={recur}]".format(a=a, b=b, recur=bool(recur)))
  # sort out arguments
  neg = 0
  if b < 0: (a, b) = (-a, -b)
  if a < 0: (a, neg) = (-a, 1)
  # the integer part
  (i, a) = divmod(a, b)
  if recur and a == 0: (i, a) = (i - 1, b)
  # record dividends
  r = dict()
  s = ''
  n = 0
  while True:
    try:
      # have we had this dividend before?
      j = r[a]
      (i, nr, rr) = (int2base(i, base, digits=digits), s[:j], s[j:])
      if neg and (nr or rr or i != '0'): i = '-' + i
      return (i, nr, rr)
    except KeyError:
      # no, we haven't
      r[a] = n
      n += 1
      (d, a) = divmod(base * a, b)
      if recur and a == 0: (d, a) = (d - 1, b)
      if not(d == a == 0):
        # add to the digit string
        s += int2base(d, base, digits=digits)

# Python 3.6: ...(*args, dp='.')
def format_recurring(*args, **kw):
  """
  format the (i, nr, rr) return from recurring() as a string

  >>> format_recurring(recurring(1, 7))
  '0.(142857)...'
  >>> format_recurring(recurring(3, 2))
  '1.5'
  >>> format_recurring(recurring(3, 2, recur=1))
  '1.4(9)...'
  >>> format_recurring(recurring(5, 17, base=16))
  '0.(4B)...'
  """
  dp = kw.get('dp', '.')
  if len(args) == 1: args = args[0]
  (i, nr, rr) = args
  rr = ('(' + rr + ')...' if rr else '')
  return (i + dp + nr + rr if nr or rr else i)

# see: Enigma 348
def reciprocals(k, b=1, a=1, m=1, g=0):
  """
  generate k whole numbers (d1, d2, ..., dk) such that 1/d1 + 1/d2 + ... + 1/dk = a/b
  the numbers are generated as an ordered list
  m = minimum allowed number
  g = minimum allowed gap between numbers

  e.g. sums of 3 reciprocals that sum to 1
  1/2 + 1/3 + 1/6 = 1
  1/2 + 1/4 + 1/4 = 1
  1/3 + 1/3 + 1/3 = 1
  >>> list(reciprocals(3, 1))
  [[2, 3, 6], [2, 4, 4], [3, 3, 3]]
  """
  # are we done?
  if k == 1:
    (d, r) = divmod(b, a)
    if r == 0 and not(d < m):
      yield [d]
  else:
    # find a suitable reciprocal
    for d in irange(m, divf(k * b, a)):
      if not(b < a * d): continue
      # and solve for the remaining fraction
      for ds in reciprocals(k - 1, b * d, a * d - b, d + g, g):
        yield [d] + ds


# command line arguments

# fetch command line arguments from sys
@static(argv=None)
def get_argv(force=0, args=None):
  if force or argv.argv is None: argv.argv = (args if args is not None else sys.argv[1:])
  return argv.argv

# alias
argv = get_argv

# might have been better to use: arg(n, fn=identity, default=None, argv=None)
# if 'p' is in PY_ENIGMA, then we will prompt
# if 'v' is in PY_ENIGMA, then we will print values
def arg(v, n, fn=identity, prompt=None, argv=None):
  """
  if command line argument <n> is specified return fn(argv[n])
  otherwise return default value <v>

  if argv is None (the default), then the value of sysv.argv[1:] is used

  >>> arg(42, 0, int, argv=['56'])
  56
  >>> arg(42, 1, int, argv=['56'])
  42
  """
  if argv is None: argv = get_argv()
  r = (fn(argv[n]) if n < len(argv) else v)
  if 'p' in _PY_ENIGMA:
    if not prompt: prompt = "value"
    s = raw_input(sprintf("arg{n}: {prompt} [{r}] > ")).strip()
    if s: r = fn(s)
  if 'v' in _PY_ENIGMA:
    if not prompt: prompt = "value"
    printf("[arg{n}: {prompt} = {r!r}]")
  return r

# get a list of similar arguments
# if no arguments are collected the value of <vs> is returned
def args(vs, n, fn=identity, prompt=None, argv=None):
  if argv is None: argv = get_argv()
  rs = (list(fn(v) for v in argv[n:]) or vs)
  if 'p' in _PY_ENIGMA:
    if not prompt: prompt = "values"
    s = raw_input(sprintf("args: {prompt} [{rs}] > ")).strip()
    if s: rs = list(fn(v) for v in re.split(r',\s*|\s+', s))
  if 'v' in _PY_ENIGMA:
    if not prompt: prompt = "values"
    printf("[args: {prompt} = {rs!r}]")
  return rs

# printf / sprintf variable interpolation
# (see also the "say" module)

# this works in all version of Python
def __sprintf(fmt, vs):
  return fmt.format(**vs)

# Python3 has str.format_map(vs)
def __sprintf3(fmt, vs):
  return fmt.format_map(vs)

# in Python v3.6.x we are getting f"..." strings which can do this job
#
# NOTE: you lose the ability to do this:
#
# printf("... {d[x]} ...", d={ 'x': 42 })  ->  "... 42 ..."
#
# instead you have to do this:
#
# printf("... {d['x']} ...", d={ 'x': 42 })  ->  "... 42 ..."
#
# but you gain the ability to use arbitrary expressions:
#
# printf("... {a} + {b} = {a + b} ...", a=2, b=3)  ->  "... 2 + 3 = 5 ..."

def __sprintf36(fmt, vs):
  return eval('f' + repr(fmt), vs)

@static(fn=None)
def _sprintf(fmt, vs, frame):
  # first try using the locals of the frame
  d = frame.f_locals
  if vs: d = update(d, vs)
  try:
    return _sprintf.fn(fmt, d)
  except (NameError, KeyError):
    pass
  # if that fails, try adding in the globals too
  d = update(frame.f_globals, frame.f_locals)
  if vs: d = update(d, vs)
  return _sprintf.fn(fmt, d)

_sprintf.fn = __sprintf
if _python > 2: _sprintf.fn = __sprintf3
if sys.version_info[0:2] > (3, 5): _sprintf.fn = __sprintf36

# print with variables interpolated into the format string
def sprintf(fmt='', **kw):
  """
  interpolate local variables and any keyword arguments into the format string <fmt>.

  >>> (a, b, c) = (1, 2, 3)
  >>> sprintf("a={a} b={b} c={c}")
  'a=1 b=2 c=3'
  >>> sprintf("a={a} b={b} c={c}", c=42)
  'a=1 b=2 c=42'
  """
  return _sprintf(fmt, kw, sys._getframe(1))

# print with local variables interpolated into the format string
def printf(fmt='', **kw):
  """
  print format string <fmt> with interpolated local variables and keyword arguments.

  the final newline can be suppressed by ending the string with '\'.

  >>> (a, b, c) = (1, 2, 3)
  >>> printf("a={a} b={b} c={c}")
  a=1 b=2 c=3
  >>> printf("a={a} b={b} c={c}", c=42)
  a=1 b=2 c=42
  """
  s = _sprintf(fmt, kw, sys._getframe(1))
  d = dict() # flush=1
  if s.endswith('\\'): (s, d['end']) = (s[:-1], '')
  print(s, **d)



# useful as a decorator for caching functions (@cached).
# NOTE: functools.lru_cached() can be used as an alternative in Python from v3.2
def cached(f):
  """
  return a cached version of function <f>.
  """
  c = dict()
  @functools.wraps(f)
  def _cached(*k):
    try:
      #printf("[{f.__name__}: cache hit, {k}")
      return c[k]
    except KeyError:
      r = c[k] = f(*k)
      #printf("[{f.__name__}: {k} -> {r}]")
      return r
  return _cached


# inclusive range iterator
@static(inf=inf) # so b=irange.inf can be used
def irange(a, b=None, step=1):
  """
  a range iterator that includes both integer endpoints, <a> and <b>.

  if only one endpoint is specified then this is taken as the highest
  value, and a lowest value of 1 is used (so irange(n) produces n
  integers from 1 to n).

  if <b> is specified as inf (or -inf for negative steps) the iterator
  generate will values indefinitely.

  >>> list(irange(1, 9))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(irange(9, 1, step=-1))
  [9, 8, 7, 6, 5, 4, 3, 2, 1]
  >>> list(irange(0, 10, step=3))
  [0, 3, 6, 9]
  >>> list(irange(9))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  """
  if step == 0: raise ValueError("irange: step cannot be 0")
  if b == inf:
    if step < 0: return range(0)
  elif b == -inf:
    if step > 0: return range(0)
  else:
    if b is None: (a, b) = (1, a)
    return range(a, b + (1 if step > 0 else -1), step)
  return itertools.count(start=a, step=step)


# flatten a list of lists
def flatten(s, fn=list):
  """
  flatten a list of lists (actually an iterator of iterators).

  >>> flatten([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> flatten(((1, 2), (3, 4, 5), (6, 7, 8, 9)), fn=tuple)
  (1, 2, 3, 4, 5, 6, 7, 8, 9)
  >>> flatten([['abc'], ['def', 'ghi']])
  ['abc', 'def', 'ghi']
  """
  return fn(j for i in s if i is not None for j in i)

# do we flatten this?
def _flatten_test(s):
  # don't flatten strings
  if isinstance(s, basestring):
    return None
  # do flatten other sequences
  if isinstance(s, Sequence):
    return s
  # otherwise don't flatten
  return None

# a generator for flattening a sequence
def _flattened(s, depth, test):
  d = (None if depth is None else depth - 1)
  for i in s:
    j = (test(i) if depth is None or depth > 0 else None)
    if j is None:
      yield i
    else:
      for k in _flattened(j, d, test): yield k

# fully flatten a nested structure
# (<fn> has been renamed to <test>)
def flattened(s, depth=None, test=_flatten_test, fn=None):
  """
  fully flatten a nested structure <s> (to depth <depth>, default is to fully flatten).

  <test> can be used to determine how objects are flattened, it should return either
  - None, if the object is not to be flattened, or
  - an iterable of objects representing one level of flattening
  default behaviour is to flatten sequences other than strings

  >>> list(flattened([[1, [2, [3, 4, [5], [[]], [[6, 7], 8], [[9]]]], []]]))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(flattened([[1, [2, [3, 4, [5], [[]], [[6, 7], 8], [[9]]]], []]], depth=3))
  [1, 2, 3, 4, [5], [[]], [[6, 7], 8], [[9]]]
  >>> list(flattened([['abc'], ['def', 'ghi']]))
  ['abc', 'def', 'ghi']
  >>> flattened(42)
  42
  """
  if test is None: test = fn
  z = (test(s) if depth is None or depth > 0 else None)
  if z is None:
    return s
  else:
    return _flattened(z, depth, test)


# return a copy of object s, but with value <v> at index <k> for (k, v) in <ps>
# <ps> can be a sequence of (k, v) pairs, or a sequence of keys, in which case
# the values should be given in <vs>
def update(s, ps=(), vs=None):
  """
  create an updated version of object <s> which is the same as <s>
  except that the value at index <k> is <v> for the keys and values
  provided in <ps> and <vs>.

  <ps> can either be a sequence of (<key>, <value>) pairs, or <ps> can
  be a sequence of keys and <vs> the corresponding sequence of values.

  >>> update([0, 1, 2, 3], [(2, 'foo')])
  [0, 1, 'foo', 3]

  >>> update(dict(a=1, b=2, c=3), 'bc', (4, 9)) == dict(a=1, b=4, c=9)
  True
  """
  if vs is not None: ps = zip(ps, vs)
  try:
    # use copy() method if available
    s = s.copy()
  except AttributeError:
    # otherwise create a new object initialised from the old one
    s = type(s)(s)
  try:
    # use update() method if available
    s.update(ps)
  except AttributeError:
    # otherwise update the pairs individually
    for (k, v) in ps:
      s[k] = v
  # return the new object
  return s

# adjacency matrix for an n (columns) x m (rows) grid
# entries are returned as lists in case you want to modify them before use
def grid_adjacency(n, m, deltas=None, include_adjacent=1, include_diagonal=0, include_self=0):
  """
  this function generates the adjacency matrix for a grid with n
  columns and m rows, represented by a linear array of size n*m

  the element in the (i, j)th position in the grid is at index (i + n*j)
  in the array

  it returns an array, where the entry at index k is a list of indices
  into the linear array that are adjacent to the square at index k.

  the default behaviour is to treat the squares immediately N, S, E, W
  of the target square as being adjacent, although this can be controlled
  with the 'deltas' parameter, it can be specified as a list of (x, y)
  deltas to use instead.

  if 'deltas' is not specified the 'include_adjacent',
  'include_diagonal' and 'include_self' flags are used to specify
  which squares are adjacent to the target square.
  'include_adjacent' includes the N, S, E, W squares
  'include_diagonal' includes the NW, NE, SW, SE squares
  'include_self' includes the square itself

  >>> grid_adjacency(2, 2)
  [[1, 2], [0, 3], [0, 3], [1, 2]]
  >>> sorted(grid_adjacency(3, 3)[4])
  [1, 3, 5, 7]
  >>> sorted(grid_adjacency(3, 3, include_diagonal=1)[4])
  [0, 1, 2, 3, 5, 6, 7, 8]
  """
  # if deltas aren't provided use standard deltas
  if deltas is None:
    deltas = list()
    if include_adjacent: deltas.extend([(0, -1), (-1, 0), (1, 0), (0, 1)])
    if include_diagonal: deltas.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])
    if include_self: deltas.append((0, 0))
  # construct the adjacency matrix
  t = n * m
  r = [None] * t
  for y in range(0, m):
    for x in range(0, n):
      s = list()
      for (dx, dy) in deltas:
        (x1, y1) = (x + dx, y + dy)
        if not(x1 < 0 or y1 < 0 or x1 + 1 > n or y1 + 1 > m):
          s.append(x1 + y1 * n)
      r[x + y * n] = s
  return r


# cumulative sum
# (see also itertools.accumulate() from Python 3.2)
def csum(i, s=0, fn=operator.add):
  """
  generate an iterator that is the cumulative sum of an iterator.

  >>> list(csum(irange(1, 10)))
  [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
  >>> list(csum(irange(1, 10), fn=operator.mul, s=1))
  [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
  >>> list(csum('python', s=''))
  ['p', 'py', 'pyt', 'pyth', 'pytho', 'python']
  """
  for x in i:
    s = fn(s, x)
    yield s


# cumulative slice
def cslice(x):
  """
  generate an iterator that is the cumulative slices of an array.

  >>> list(cslice([1, 2, 3]))
  [[1], [1, 2], [1, 2, 3]]
  >>> list(cslice('python'))
  ['p', 'py', 'pyt', 'pyth', 'pytho', 'python']
  """
  i = 0
  while i < len(x):
    i += 1
    yield x[:i]


# overlapping tuples from a sequence
def tuples(s, n=2):
  """
  generate overlapping <n>-tuples from sequence <s>.

  (for non-overlapping tuples see chunk()).

  >>> list(tuples('12345'))
  [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]
  >>> list(tuples(irange(1, 5), 3))
  [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
  """
  i = iter(s)
  t = list()
  try:
    # collect the first tuple
    for _ in range(n):
      t.append(next(i))
    while True:
      # return the tuple
      yield tuple(t)
      # move the next value in to the tuple
      t.pop(0)
      t.append(next(i))
  except StopIteration:
    pass

def contains(seq, subseq):
  """
  return the position in <seq> that <subseq> occurs as a contiguous subsequence
  or -1 if it is not found

  >>> contains("abcdefghijkl", "def")
  3
  >>> contains("abcdefghijkl", "hik")
  -1
  >>> contains(Primes(), [11, 13, 17, 19])
  4
  >>> contains([1, 2, 3], [1, 2, 3])
  0
  >>> contains([1, 2, 3], [])
  0
  """
  subseq = tuple(subseq)
  n = len(subseq)
  if n == 0: return 0
  k = 0
  i = 1
  for x in seq:
    if x == subseq[k]:
      k += 1
      if k == n: return i - n
    else:
      k = 0
    i += 1
  return -1

# subseqs: generate the subsequences of an iterator -> replaced by subsets()

# bit permutations
# see: https://enigmaticcode.wordpress.com/2017/05/20/bit-twiddling/
def bit_permutations(a, b=None):
  """
  generate numbers in order that have the same number of bits set in
  their binary representation.

  numbers start at <a> and are generated while they are smaller than
  <b>.

  to generate all numbers with k bits start start with:

    a = 2 ** k - 1
    a = (1 << k) - 1

  >>> list(bit_permutations(3, 20))
  [3, 5, 6, 9, 10, 12, 17, 18]
  >>> first(bit_permutations(1), 11)
  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  """
  if a == 0:
    yield a
    return
  while b is None or a < b:
    yield a
    t = (a | (a - 1)) + 1
    a = t | ((((t & -t) // (a & -a)) >> 1) - 1)

###############################################################################

# Numerical approximations

# a simple record type class for results
# (Python 3.3 introduced types.SimpleNamespace)
class Record(object):

  def update(self, **vs):
    "update values in a record"
    self.__dict__.update(vs)

  # __init__ is the same as update
  __init__ = update

  def __iter__(self):
    d = self.__dict__
    for k in sorted(d.keys()):
      yield (k, d[k])

  def __repr__(self):
    return self.__class__.__name__ + '(' + join((k + '=' + repr(v) for (k, v) in self), sep=', ') + ')'

  def map(self):
    return self.__dict__


# a golden section search minimiser
# f - function to minimise
# a, b - bracket to search
# t - tolerance
# m - metric
def gss_minimiser(f, a, b, t=1e-9, m=None):
  # apply any metric
  fn = (f if m is None else lambda x: m(f(x)))
  R = 0.5 * (sqrt(5.0) - 1.0)
  C = 1.0 - R
  (x1, x2) = (R * a + C * b, C * a + R * b)
  (f1, f2) = (fn(x1), fn(x2))
  while b - a > t:
    if f1 > f2:
      (a, x1, f1) = (x1, x2, f2)
      x2 = C * a + R * b
      f2 = fn(x2)
    else:
      (b, x2, f2) = (x2, x1, f1)
      x1 = R * a + C * b
      f1 = fn(x1)
  return (Record(v=x1, fv=f(x1), t=t) if f1 < f2 else Record(v=x2, fv=f(x2), t=t))


find_min = gss_minimiser
find_min.__name__ = 'find_min'
find_min.__doc__ = """
  find the minimum value of a (well behaved) function over an interval.

  f = function to minimise (should take a single float argument)
  a, b = the interval to minimise over (a < b)
  t = the tolerance to work to
  m = the metric we want to minimise (default is None = the value of the function)

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is minimised
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_min(lambda x: (x - 2) ** 2, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
"""

# NOTE: using functools.partial and setting __name__ and __doc__ doesn't work (in Python 2.7 and 3.3)
# see: http://bugs.python.org/issue12790
def find_max(f, a, b, t=1e-9):
  """
  find the maximum value of a (well behaved) function over an interval.

  f = function to maximise (should take a single float argument)
  a, b = the interval to maximise over (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is maximised
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_max(lambda x: 9 - (x - 2) ** 2, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  """
  return gss_minimiser(f, a, b, t=t, m=lambda x: -x)

def find_zero(f, a, b, t=1e-9, ft=1e-6):
  """
  find the zero of a (well behaved) function over an interval.

  f = function to find the zero of (should take a single float argument)
  a, b = the interval to maximise over (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is zero
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_zero(lambda x: x ** 2 - 4, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  >>> r = find_zero(lambda x: x ** 2 + 4, 0.0, 10.0) # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
    ...
  ValueError: Value not found
  """
  r = find_min(f, a, b, t, m=abs)
  if ft < abs(r.fv): raise ValueError("Value not found")
  r.ft = ft
  return r

def find_value(f, v, a, b, t=1e-9, ft=1e-6):
  """
  find the value of a (well behaved) function over an interval.

  f = function to find the value of (should take a single float argument)
  a, b = the interval to search over (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is the specified value
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_value(lambda x: x ** 2 + 4, 8.0, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  """
  r = find_zero(lambda x: f(x) - v, a, b, t, ft)
  r.fv += v
  return r


###############################################################################

# Roman Numerals
# the following is good for numbers in the range 1 - 4999

_romans = (
  # X bar = 10000
  # V bar = 5000
  # (<numeral>, <value>, <max-repeats>)
  ('M',  1000, 4),
  ('CM',  900, 1),
  ('D',   500, 1),
  ('CD',  400, 1),
  ('C',   100, 3),
  ('XC',   90, 1),
  ('L',    50, 1),
  ('XL',   40, 1),
  ('X',    10, 3),
  ('IX',    9, 1),
  ('V',     5, 1),
  ('IV',    4, 1),
  ('I',     1, 4)
)


def int2roman(x):
  """
  return a representation of an integer <x> (from 1 to 4999) as a Roman Numeral

  >>> int2roman(4)
  'IV'
  >>> int2roman(1999)
  'MCMXCIX'
  """
  x = int(x)
  if not(0 < x < 5000): raise ValueError("integer out of range: {x}".format(x=x))
  s = list()
  for (n, i, m) in _romans:
    (d, r) = divmod(x, i)
    if d < 1: continue
    s.append(n * d)
    x = r
  return join(s)


def roman2int(x):
  """
  return the integer value of a Roman Numeral <x>.

  >>> roman2int('IV')
  4
  >>> roman2int('IIII')
  4
  >>> roman2int('MCMXCIX')
  1999
  """
  x = str(x).upper()
  p = r = 0
  for (n, i, m) in _romans:
    (l, c) = (len(n), 0)
    while x[p:p + l] == n and c < m:
      r += i
      p += l
      c += 1
  if p < len(x): raise ValueError("invalid Roman numeral: {x}".format(x=x))
  return r


def is_roman(x):
  """
  check if a Roman Numeral <x> is valid.

  >>> is_roman('IV')
  True
  >>> is_roman('IIII')
  True
  >>> is_roman('XIVI')
  False
  """
  x = str(x).upper()
  if x == 'IIII': return True
  try:
    i = roman2int(x)
  except ValueError:
    return False
  return int2roman(i) == x


# (default) digits for use in converting bases
_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

@static(digits=_DIGITS)
def base_digits(*args):
  """
  get or set the default string of digits used to represent numbers.

  with no arguments the current string of digits is returned.

  with an argument the current string is set, and the new string
  returned, if the argument is None (or any non-True value) then
  the standard default string is used (0-9 then A-Z).

  NOTE: this is a global setting and will affect all subsequent
  base conversions.
  """
  assert len(args) < 2
  if args: base_digits.digits = (args[0] or _DIGITS)
  return base_digits.digits

def int2base(i, base=10, width=None, pad=None, group=None, sep=",", digits=None):
  """
  convert an integer <i> to a string representation in the specified base <base>.

  if the <width> parameter is specified the number of digits will be
  padded to value of <width> using the <pad> character. if <width> is
  positive pad characters will be added on the left, if negative they
  are added on the right. The default pad character is the digit 0.

  if the <group> parameter is specified the digits are grouped into
  blocks of <group> digits and separated by the string <sep> (this
  happens after the digits are padded to any specified <width>). if
  <group> is positive the groups start from the right, if negative
  they start from the left.

  By default this routine only handles single digits up 36 in any given base,
  but the <digits> parameter can be specified to give the symbols for larger bases.

  >>> int2base(-42)
  '-42'
  >>> int2base(3735928559, base=16)
  'DEADBEEF'
  >>> int2base(-3735928559, base=16, digits='0123456789abcdef')
  '-deadbeef'
  >>> int2base(190, base=3, digits='zyx')
  'xyzzy'
  >>> int2base(29234652, base=36)
  'HELLO'
  >>> int(int2base(123456, base=14), base=14)
  123456
  >>> int2base(84, base=2, width=9, group=3, sep="_")
  '001_010_100'
  """
  assert base > 1, "invalid base {base}".format(base=base)
  if digits is None: digits = base_digits()
  if i == 0:
    r = digits[0]
  elif i < 0:
    return '-' + int2base(-i, base=base, width=width, pad=pad, group=group, sep=sep, digits=digits)
  else:
    r = list()
    while i > 0:
      (i, n) = divmod(i, base)
      r.insert(0, digits[n])
    r = join(r)
  if width is not None:
    if pad is None: pad = digits[0]
    r = (r.rjust(width, pad) if width > 0 else r.ljust(-width, pad))
  if group is not None:
    (s, group) = ((-1, group) if group > 0 else (1, -group))
    r = join((join(x) for x in chunk(r[::s], group)), sep=sep[::s])[::s]
  return r

def base2int(s, base=10, strip=0, digits=None):
  """
  convert a string representation of an integer in the specified base to an integer.

  >>> base2int('-42')
  -42
  >>> base2int('xyzzy', base=3, digits='zyx')
  190
  >>> base2int('HELLO', base=36)
  29234652
  """
  assert base > 1, "invalid base {base}".format(base=base)
  if digits is None: digits = base_digits()
  if len(digits) > base:
    digits = digits[:base]
  if s == digits[0]:
    return 0
  elif s.startswith('-'):
    return -base2int(s[1:], base=base, strip=strip, digits=digits)
  i = 0
  for d in s:
    try:
      v = digits.index(d)
    except ValueError as e:
      if strip: continue
      e.args = ("invalid digit for base {base}: {s}".format(base=base, s=s),)
      raise

    i *= base
    i += v
  return i

def digit_map(a=0, b=9, digits=None):
  """
  create a map (dict) mapping individual digits to their numerical value.

  the symbols used for the digits can be provided, otherwise the default
  list set via base_digits() is used

  >>> digit_map(1, 3) == { '1': 1, '2': 2, '3': 3 }
  True
  """
  if digits is None: digits = base_digits()
  return dict((digits[i], i) for i in irange(a, b))


_numbers = {
  0: 'zero',
  1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
  10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 15: 'fifteen', 18: 'eighteen',
}

_tens = {
  1: 'teen', 2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'
}

def int2words(n, scale='short', sep='', hyphen=' '):
  """
  convert an integer <n> to a string representing the number in English.

  scale - 'short' (for short scale numbers), or 'long' (for long scale numbers)
  sep - separator between groups
  hyphen - separator between tens and units

  >>> int2words(1234)
  'one thousand two hundred and thirty four'
  >>> int2words(-7)
  'minus seven'
  >>> int2words(factorial(13))
  'six billion two hundred and twenty seven million twenty thousand eight hundred'
  >>> int2words(factorial(13), sep=',')
  'six billion, two hundred and twenty seven million, twenty thousand, eight hundred'
  >>> int2words(factorial(13), sep=',', hyphen='-')
  'six billion, two hundred and twenty-seven million, twenty thousand, eight hundred'
  >>> int2words(factorial(13), scale='long')
  'six thousand two hundred and twenty seven million twenty thousand eight hundred'
  """
  return _int2words(int(n), scale, sep, hyphen)

def _int2words(n, scale='short', sep='', hyphen=' '):
  if n in _numbers:
    return _numbers[n]
  if n < 0:
    return 'minus ' + _int2words(-n, scale, sep, hyphen)
  if n < 20:
    return _numbers[n % 10] + _tens[1]
  if n < 100:
    (d, r) = divmod(n, 10)
    x = _tens[d]
    if r == 0: return x
    return x + hyphen + _numbers[r]
  if n < 1000:
    (d, r) = divmod(n, 100)
    x = _int2words(d, scale, sep, hyphen) + ' hundred'
    if r == 0: return x
    return x + ' and ' + _int2words(r, scale, sep, hyphen)
  if n < 1000000:
    (d, r) = divmod(n, 1000)
    x = _int2words(d, scale, sep, hyphen) + ' thousand'
    if r == 0: return x
    if r < 100: return x + ' and ' + _int2words(r, scale, sep, hyphen)
    return x + sep + ' ' + _int2words(r, scale, sep, hyphen)
  try:
    return __int2words(n, scale, sep, hyphen)
  except IndexError:
    raise ValueError('Number too large')

# from http://en.wikipedia.org/wiki/Names_of_large_numbers
_larger = [
  'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion',
  'nonillion', 'decillion', 'undecillion', 'duodecillion', 'tredecillion', 'quattuordecillion', 'quindecillion',
  'sexdecillion', 'septendecillion', 'octodecillion', 'novemdecillion', 'vigintillion', 'unvigintillion',
  'duovigintillion', 'tresvigintillion', 'quattuorvigintillion', 'quinquavigintillion', 'sesvigintillion',
  'septemvigintillion', 'octovigintillion', 'novemvigintillion', 'trigintillion', 'untrigintillion',
  'duotrigintillion', 'trestrigintillion', 'quattuortrigintillion', 'quinquatrigintillion', 'sestrigintillion',
  'septentrigintillion', 'octotrigintillion', 'noventrigintillion', 'quadragintillion',
]

def __int2words(n, scale='short', sep='', hyphen=' '):
  """
  convert a large integer (one million or greater) to a string
  representing the number in English, using short or long scale.

  >>> __int2words(10 ** 12, scale='short')
  'one trillion'
  >>> __int2words(10 ** 12, scale='long')
  'one billion'
  """
  if scale == 'short':
    (g, p, k) = (3, 1000, 2)
  elif scale == 'long':
    (g, p, k) = (6, 1000000, 1)
  else:
    raise ValueError('Unsupported scale type: ' + scale)
  i = (len(str(n)) - 1) // g
  (d, r) = divmod(n, p ** i)
  x = _int2words(d, scale, sep, hyphen) + ' ' + _larger[i - k]
  if r == 0: return x
  if r < 100: return x + ' and ' + _int2words(r, scale, sep, hyphen)
  return x + sep + ' ' + _int2words(r, scale, sep, hyphen)

# convert an integer to BCD (binary coded decimal)
# same as: nconcat(nsplit(n, base=10), base=16)
def int2bcd(n, base=10, bits_per_digit=4):
  """
  convert integer n into BCD (Binary Coded Decimal)

  the base and bits_per_integer can be specified (if desired)

  >>> int2bcd(123456)
  1193046
  >>> int2bcd(123456) == 0x123456
  True
  """
  s = 1
  if n < 0: (s, n) = (-1, -n)
  r = k = 0
  while True:
    (n, x) = divmod(n, base)
    r += (x << k)
    if n == 0: break
    k += bits_per_digit
  return s * r

###############################################################################

# specialised classes:

###############################################################################

# Delayed Evaluation

# delayed evaluation (see also lazypy)
class Delay(object):

  # set to force immediate evaluation
  immediate = 0

  def __init__(self, fn, *args, **kw):
    """
    create a delayed evaluation promise for fn(*args, **kw).

    Note that if you want the arguments themselves to be lazily evaluated
    you will need to use:

      Delay(lambda: fn(expr1, expr2, opt1=expr3, opt2=expr4))

    rather than:

      Delay(fn, expr1, expr2, opt1=expr3, opt2=expr4)

    example:

      x = Delay(expensive, 1)
      x.evaluated --> False
      x.value --> returns expensive(1)
      x.evaluated --> True
      x.value --> returns expensive(1) again, without re-evaluating it
      x.reset()
      x.evaluated --> False
      x.value --> returns expensive(1), but re-evaluates it
    """
    self.fn = fn
    self.args = args
    self.kw = kw
    self.evaluated = False
    if self.immediate: self.evaluate()

  def __repr__(self):
    return self.__class__.__name__ + '(value=' + (repr(self.value) if self.evaluated else '<delayed>') + ')'

  def evaluate(self):
    self.value = self.fn(*(self.args), **(self.kw))
    self.evaluated = True
    return self.value

  def reset(self):
    del self.value
    self.evaluated = False

  def __getattr__(self, key):
    if key == 'value':
      return self.evaluate()
    else:
      raise AttributeError()

  @classmethod
  def iter(self, *args):
    """
    create an iterator that takes a sequence of Delay() objects (or
    callable objects with no arguments) and evaluates and returns each
    one as next() is called.

    i = Delay.iter(Delay(expensive, 1), Delay(expensive, 2), Delay(expensive, 3))
    next(i) --> evaluates and returns expensive(1)
    next(i) --> evaluates and returns expensive(2)
    next(i) --> evaluates and returns expensive(3)
    """
    # use x.value for delay objects, and just try and call everything else
    return ((x.value if type(x) is self else x()) for x in args)

###############################################################################

# Value Accumulator


class Accumulator(object):

  """
  A value accumulator.

  >>> a = Accumulator(fn=max)
  >>> for x in (6, 5, 4, 7, 3, 1): a.accumulate(x)
  >>> a.value
  7
  >>> a = Accumulator()
  >>> for x in irange(1, 9): a.accumulate(x)
  >>> a.value
  45
  >>> fdiv(a.value, a.count)
  5.0
  """

  def __init__(self, fn=operator.add, value=None, data=None, count=0):
    """
    create an Accumulator.

    The accumulation function and initial value can be specified.
    """
    self.fn = fn
    self.value = value
    self.data = data
    self.count = count


  def __repr__(self):
    return 'Accumulator(value=' + repr(self.value) + ', data=' + repr(self.data) + ', count=' + str(self.count) + ')'

  def accumulate(self, v=1):
    """
    Accumulate a value.

    If the current value is None then this value replaces the current value.
    Otherwise it is combined with the current value using the accumulation
    function which is called as fn(<current-value>, v).
    """
    self.count += 1
    self.value = (v if self.value is None else self.fn(self.value, v))


  def accumulate_data(self, v, data, t=None):
    """
    Accumulate a value, and check the accumulated value against a target value,
    and if it matches record the data parameter.

    You can use this to record data where some function of the data is at an
    extremum value.
    """
    self.accumulate(v)
    if self.value == (v if t is None else t): self.data = data

  def accumulate_from(self, s):
    """
    Accumulate values from iterable object <s>.
    """
    for v in s:
      self.accumulate(v)
    return self

  def accumulate_data_from(self, s, value=0, data=1):
    """
    Accumulate values and data from iterable object <s>.

    <value>, <data> can be an index into elements from <s>
    or a function to extract the appropriate value from an element.
    """
    fn = lambda i: (lambda x: x[i])
    if not callable(value): value = fn(value)
    if not callable(data): data = fn(data)
    for x in s:
      self.accumulate_data(value(x), data(x))
    return self


###############################################################################

# Routines for dealing with polynomials

# represent polynomial a + bx + cx^2 + dx^3 + ... as:
#
#   [a, b, c, d, ...]
#
# so the polynomial can be reconstructed as:
#
#   sum(c * pow(x, i) for (i, x) in enumerate(poly))
#

# make a polynomial from (exponent, coefficient) pairs
# (we can use enumerate() to reverse the process)
def poly_from_pairs(ps, p=None):
  if p is None: p = []
  for (e, c) in ps:
    if c != 0:
      x = e + 1 - len(p)
      if x > 0: p.extend([0] * x)
      p[e] += c
  return poly_trim(p)

# remove extraneous zero coefficients
def poly_trim(p):
  while len(p) > 1 and p[-1] == 0: p.pop()
  return p

# we can multiply two polynomials
def poly_mul(p, q):
  return poly_from_pairs(
    ((i + j, a * b) for (i, a) in enumerate(p) for (j, b) in enumerate(q)),
    [0] * (len(p) + len(q) - 1)
  )

poly_zero = [0]
poly_unit = [1]

# and multiply any number of polynomials
def poly_multiply(*ps):
  r = poly_unit
  for p in ps:
    r = poly_mul(r, p)
  return r

# and raise a polynomial to a (positive) integer power
def poly_pow(p, n):
  r = poly_unit
  while n > 0:
    (n, m) = divmod(n, 2)
    if m: r = poly_mul(r, p)
    if n: p = poly_mul(p, p)
  return r

# add two polynomials
def poly_add(p, q):
  return poly_from_pairs(enumerate(p), list(q))

# add any number of polynomials
def poly_sum(*ps):
  r = poly_zero
  for p in ps:
    r = poly_add(r, p)
  return r

# map a function over the coefficients of a polynomial
def poly_map(p, fn):
  return poly_trim(list(fn(x) for x in p))

# negate a polynomial
def poly_neg(p):
  return list(-c for c in p)

# subtract two polynomials
def poly_sub(p, q):
  return poly_add(p, poly_neg(q))

# divide two polynomials
# div() is used for coefficient division, if the leading coefficient of q is not 1
# (you probably want to use a rational implementation such as fractions.Fraction)
def poly_divmod(p, q, div=None):
  fn = (identity if q[-1] == 1 else (lambda x: div(x, q[-1])))
  (d, r) = (poly_zero, p)
  while r != poly_zero:
    k = len(r) - len(q)
    if k < 0: break
    m = poly_from_pairs([(k, fn(r[-1]))])
    d = poly_add(d, m)
    r = poly_sub(r, poly_mul(m, q))
  return (d, r)

# print a polynomial in a more friendly form
def poly_print(p):
  r = list()
  for (e, c) in enumerate(p):
    if c == 0: continue
    s = str(c)
    if not(c < 0): s = '+' + s
    s = '(' + s + ')'
    if e == 0:
      pass
    elif e == 1:
      s = s + 'x'
    else:
      s = s + 'x^' + str(e)
    r.append(s)
  return join(r[::-1], sep=" ") or "(0)"

# evaluate a polynomial
def poly_value(p, x):
  v = 0
  for n in reversed(p):
    v *= x
    v += n
  return v


# wrap the whole lot up in a class

class Polynomial(list):

  def __repr__(self):
    return self.__class__.__name__ + "[" + poly_print(self) + "]"

  def __hash__(self):
    return hash(tuple(self))

  def __add__(self, other):
    return self.__class__(poly_add(self, other))

  def __mul__(self, other):
    return self.__class__(poly_mul(self, other))

  def __neg__(self):
    return self.__class__(poly_neg(self))

  def __pow__(self, n):
    return self.__class__(poly_pow(self, n))

  def __sub__(self, other):
    return self.__class__(poly_sub(self, other))

  __call__ = poly_value

  def to_pairs(self):
    for p in enumerate(self):
      if p[1] != 0:
        yield p

  @classmethod
  def from_pairs(self, ps):
    return self(poly_from_pairs(ps))

  @classmethod
  def unit(self):
    return self(poly_unit)

  @classmethod
  def zero(self):
    return self(poly_zero)

###############################################################################

# Prime Sieves

_primes_array = bytearray
_primes_size = 1024
_primes_chunk = lambda n: 2 * n


class _PrimeSieveE6(object):

  """
  A prime sieve.

  The 'array' parameter can be used to specify a list like class to implement
  the sieve. Possible values for this are:

  list - use standard Python list
  bytearray - faster and uses less space (default)
  bitarray - (if you have it) less space that bytearray, but more time than list

  >>> _PrimeSieveE6(50).list()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  >>> primes = _PrimeSieveE6(1000000)
  >>> primes.is_prime(10001)
  False
  >>> 10007 in primes
  True
  >>> sum(primes) == 37550402023
  True
  >>> list(primes.range(2, 47))
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

  NOTE: if you make a large sieve it will use up lots of memory.
  """

  # a width 6 prime sieve
  #
  # the sieve itself represents numbers with a residue of 1 and 5 modulo 6
  #
  # i:  0   1   | 2   3  |  4   5  |  6   7  |  8   9  | 10  11 | ...
  # p:  1   5   | 7  11  | 13  17  | 19  23  | 25  29  | 31  35 | ...
  #
  # i->p = (3 * i) + (i & 1) + 1
  # p->i = p // 3 (or in general: p->i = (p + 1) // 3 - (p % 6 == 5))
  #
  # to check numbers up to (but not including) n we need a sieve of size: (n // 3) + (n % 6 == 2)

  def __init__(self, n, array=_primes_array):
    """
    make a sieve of primes up to n
    """
    # initial sieve
    self.sieve = array([0])
    self.max = 1
    # singleton arrays for True and False values
    self.T = array([1])
    self.F = array([0])
    # now extend the sieve to the required size
    self.extend(n)

  def __repr__(self):
    return self.__class__.__name__ + '(max=' + repr(self.max) + ')'

  def extend(self, n):
    """
    extend the sieve up to (at least) n
    """
    if not(n > self.max): return
    #printf("_PrimeSieveE6: expanding to {n}")

    # extend the sieve to the right size
    s = self.sieve
    l = len(s) + 1
    h = ((n + 1) // 3) + (n % 6 == 1)
    s.extend(self.T * (h - l + 1))

    # remove multiples of primes p from indices l to h
    for i in irange(1, isqrt(n) // 3):
      if s[i]:
        odd = (i & 1)
        p = (i * 3) + odd + 1
        k = 2 * p
        # remove multiples of p starting from p^2
        j = (p * p) // 3
        if j < l: j += k * ((l - j) // k)
        s[j::k] = self.F * ((h - j - 1) // k + 1)
        #printf("eliminating {p} {ns}", ns=tuple((z * 3) + (z & 1) + 1 for z in irange(j, h-1, step=k)))
        # remove multiples with the other residue
        q = p + (2 if odd else 4)
        j = (p * q) // 3
        if j < l: j += k * ((l - j) // k)
        s[j::k] = self.F * ((h - j - 1) // k + 1)
        #printf("eliminating {p} {ns}", ns=tuple((z * 3) + (z & 1) + 1 for z in irange(j, h-1, step=k)))

    self.max = n
    #printf("[_PrimeSieveE6: extended to {n}: {b} bytes used]", b=s.__sizeof__())

  # return a list of primes (more space)
  def list(self):
    """
    return a list of primes in the sieve (in numerical order).

    (this will require more memory than generate()).
    """
    return list(_PrimeSieveE6.generate(self))

  # return a generator (less space)
  def generate(self, start=0, end=None):
    """
    generate all primes in the sieve (in numerical order).

    the range of primes can be restricted to starting at <start>
    and ending at <end> (primes less than <end> will be returned)

    (this will require less memory than list())
    """
    if end is None: end = self.max
    if start < 3 and end > 2: yield 2
    if start < 4 and end > 3: yield 3
    s = self.sieve
    # generate primes from <start> up to (but not including) <end>
    for i in range((start + 1) // 3 - (start % 6 == 5), (end + 1) // 3 - (end % 6 == 5)):
      if s[i]: yield (i * 3) + (i & 1) + 1

  # make this an iterable object
  __iter__ = generate

  # range(a, b) - generate primes in the range [a, b) - is the same as generate() now
  range = generate

  # irange = inclusive range
  def irange(self, a, b):
    return self.range(a, b + 1)

  # prime test (may throw IndexError if n is too large)
  def is_prime(self, n):
    """
    check to see if the number is a prime.

    (may throw IndexError for numbers larger than the sieve).
    """
    if n < 2: return False # 0, 1 -> F
    if n < 4: return True # 2, 3 -> T
    (i, r) = divmod(n, 6)
    if r != 1 and r != 5: return False # (n % 6) != (1, 5) -> F
    return bool(self.sieve[n // 3])

  prime = is_prime

  # allows use of "in"
  __contains__ = is_prime

  # generate prime factors of <n>
  # (try setting mr=100 if checking large numbers)
  def prime_factor(self, n, mr=0):
    """
    generate (<prime>, <exponent>) pairs in the prime factorisation of
    positive integer <n>.

    if <mr> is set the program will use a Miller-Rabin probabilistic
    test after <mr> primes have failed to divide the residue.

    Note: This will only divide primes up to the limit of the sieve,
    so this is a complete factorisation for <n> up to the square of the
    limit of the sieve. When <mr> is set it can cope with numbers that
    have one large prime factor.
    """
    # maybe should be: n < 1
    #if n < 0: raise ValueError("can only factorise positive integers")
    if n > 1:
      t = 0
      i = self.generate()
      while n > 1:

        if n < self.max:
          if self.is_prime(n):
            yield (n, 1)
            return

        elif mr and t == mr and is_prime_mr(n):
          yield (n, 1)
          return

        p = next(i)
        e = 0
        while True:
          (d, r) = divmod(n, p)
          if r != 0: break
          e += 1
          n = d
        if e > 0:
          yield (p, e)
          t = 0
        else:
          t += 1

  # functions that can use self.prime_factor() instead of simple prime_factor()

  # return a list of the factors of n
  def factor(self, n):
    """
    return a list of the prime factors of positive integer <n>.

    Note: This will only consider primes up to the limit of the sieve,
    this is a complete factorisation for <n> up to the square of the
    limit of the sieve.
    """
    return factor(n, fn=self.prime_factor)

  def divisors(self, n):
    return divisors(n, fn=self.prime_factor)

  def divisors_pairs(self, n):
    return divisors_pairs(n, fn=self.prime_factor)

  def tau(self, n):
    return tau(n, fn=self.prime_factor)

  def is_square_free(self, n):
    return is_square_free(n, fn=self.prime_factor)

# an expandable version of the sieve

class _PrimeSieveE6X(_PrimeSieveE6):
  """
  Make an expanding sieve of primes with an initial maximum of <n>.

  When the sieve is expanded the function <fn> is used to calculate
  the new maximum, based on the previous maximum.

  The default function doubles the maximum at each expansion.

  To find the 1000th prime,
  (actually a list of length 1 starting with the 1000th prime):
  >>> primes = _PrimeSieveE6X(1000)
  >>> first(primes, 1, 999)
  [7919]

  We can then find the one millionth prime and the generator will
  expand as necessary:
  >>> first(primes, 1, 999999)
  [15485863]

  We can see what the current maximum number considered is:
  >>> primes.max
  16384000

  And can test for primality up to this value:
  >>> 1000003 in primes
  True

  The sieve will automatically expand as it is used:
  >>> primes.is_prime(17000023)
  True
  >>> primes.max
  17000023
  """
  def __init__(self, n, array=_primes_array, fn=_primes_chunk):
    """
    make a sieve of primes with an initial maximum of <n>.

    when the sieve is expanded the function <fn> is used to calculate
    the new maximum, based on the previous maximum.

    the default function doubles the maximum at each expansion.
    """
    self.chunk = fn
    _PrimeSieveE6.__init__(self, n, array=array)

  # expand the sieve up to n, or by the next chunk
  def extend(self, n=None):
    """
    extend the sieve to include primes up to (at least) n.

    if n is not specified that sieve will be expanded according to the
    function specified in __init__().
    """
    if n is None: n = self.chunk(self.max)
    _PrimeSieveE6.extend(self, n)

  # for backwards compatibility
  expand = extend

  # generate all primes, a chunk at a time
  def generate(self, start=0):
    """
    generate primes without limit, expanding the sieve as necessary.

    eventually the sieve will consume all available memory.
    """
    while True:
      # generate all primes currently in the sieve
      for p in _PrimeSieveE6.generate(self, start): yield p
      start = self.max + 1
      # then expand the sieve
      self.expand()

  # make this an iterable object
  __iter__ = generate

  # expand the sieve as necessary
  def is_prime(self, n):
    """
    primality test - the sieve is expanded as necessary before testing.
    """
    self.extend(n)
    return _PrimeSieveE6.is_prime(self, n)

  # allows use of "in"
  __contains__ = is_prime

  # expand the sieve as necessary
  def range(self, a=0, b=None):
    """
    generate primes in the (inclusive) range [a, b].

    the sieve is expanded as necessary beforehand.
    """
    # have we asked for unlimited generation?
    if not b: return self.generate(a)
    # otherwise, upper limit is provided
    self.extend(b)
    return _PrimeSieveE6.range(self, a, b)

# create a suitable prime sieve
def Primes(n=None, expandable=0, array=_primes_array, fn=_primes_chunk):
  """
  Return a suitable prime sieve object.

  n - initial limit of the sieve (the sieve contains primes less than n)
  expandable - should the sieve expand as necessary
  array - list implementation to use
  fn - function used to increase the limit on expanding sieves

  If we are interested in a limited collection of primes, we can do
  this:

  >>> primes = Primes(50)
  >>> primes.list()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  >>> sum(primes)
  328
  >>> 39 in primes
  False

  The collection can be extended manually to a new upper limit:

  >>> primes.extend(100)
  >>> sum(primes)
  1060
  >>> 97 in primes
  True

  but it doesn't automatically expand.

  If we want an automatically expanding version, we can set the
  'expandable' flag to True.

  >>> primes = Primes(50, expandable=1)

  We can find out the current size and contents of the sieve:
  >>> primes.max
  50
  >>> primes.list()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

  But if we use it as a generator it will expand indefinitely, so we
  can only sum a restricted range:
  >>> sum(primes.range(0, 100))
  1060

  If you don't know how many primes you'll need you can just use
  Primes() and get an expandable sieve with primes up to 1024, and the
  limit will double each time the sieve is expanded.

  So, to sum the first 1000 primes:
  >>> sum(first(Primes(), 1000))
  3682913
  """
  # if n is None then make it expandable by default
  if n is None: (n, expandable) = (_primes_size, True)
  # return an appropriate object
  if expandable:
    return _PrimeSieveE6X(n, array=array, fn=fn)
  else:
    return _PrimeSieveE6(n, array=array)

# backwards compatibility
def PrimesGenerator(n=None, array=_primes_array, fn=_primes_chunk):
  return Primes(n, expandable=1, array=array, fn=fn)

# default expandable sieve
primes = Primes(1, expandable=1, array=_primes_array, fn=(lambda n: _primes_size if n < _primes_size else 2 * n))

###############################################################################

# Magic Square Solver:

# this is probably a bit of overkill but it works and I already had the code written

class Impossible(Exception): pass
class Solved(Exception): pass

class MagicSquare(object):

  """
  A magic square solver.

  e.g. to create a 3x3 magic square with 1 at the top centre:

  >>> p = MagicSquare(3)
  >>> p.set(1, 1)
  >>> p.solve()
  True
  >>> p.output()
  [6] [1] [8] 
  [7] [5] [3] 
  [2] [9] [4] 

  When referencing the individual squares through set() and get()
  the squares are linearly indexed from 0 through to n^2-1.

  If you haven't filled out initial values for any squares then
  solve() is likely to take a while, especially for larger magic squares.

  NOTE: This class will currently find _a_ solution for the square
  (if it is solvable), but the solution is not necessarily unique.
  A future version of this code may include a function that generates
  all possible solutions.
  """

  # an n x n magic square
  def __init__(self, n, numbers=None, lines=None):
    """
    create an empty n x n magic square puzzle.

    The numbers to fill out the magic square can be specified.
    (If they are not specified numbers from 1 to n^2 are used).

    The magic lines can be specified as tuples of indices.
    (If they are not specified then it is assumed that all the n
    rows, n columns and 2 diagonals should sum to the magic value).
    """
    n2 = n * n
    if numbers is None: numbers = irange(1, n2)
    numbers = set(numbers)
    s = sum(numbers) // n

    # make the magic lines
    if lines is None:
      lines = []
      l = tuple(range(n))
      for i in range(n):
        # row
        lines.append(tuple(i * n + j for j in l))
        # column
        lines.append(tuple(i + j * n for j in l))
      # diagonals
      lines.append(tuple(j * (n + 1) for j in l))
      lines.append(tuple((j + 1) * (n - 1) for j in l))

    self.n = n
    self.s = s
    self.square = [0] * n2
    self.numbers = numbers
    self.lines = lines

  def set(self, i, v):
    """set the value of a square (linearly indexed from 0)."""
    self.square[i] = v
    self.numbers.remove(v)

  def get(self, i):
    """get the value of a square (linearly indexed from 0)."""
    return self.square[i]

  def output(self):
    """print the magic square."""
    m = max(self.square)
    n = (int(math.log10(m)) + 1 if m > 0 else 1)
    fmt = "[{:>" + str(n) + "s}]"
    for y in range(self.n):
      for x in range(self.n):
        v = self.square[y * self.n + x]
        print(fmt.format(str(v) if v > 0 else ''), end=' ')
      print('')

  # complete: complete missing squares

  def complete(self):
    """strategy to complete missing squares where there is only one possibility"""
    if not self.numbers: raise Solved()
    for line in self.lines:
      ns = tuple(self.square[i] for i in line)
      z = ns.count(0)
      if z == 0 and sum(ns) != self.s: raise Impossible()
      if z != 1: continue
      v = self.s - sum(ns)
      if v not in self.numbers: raise Impossible()
      i = ns.index(0)
      self.set(line[i], v)
      return True
    return False

  # hypothetical: make a guess at a square

  def clone(self):
    """return a copy of this object"""
    return copy.deepcopy(self)

  def become(self, other):
    """set the attributes of this object from the other object"""
    for x in vars(self).keys():
      setattr(self, x, getattr(other, x))

  def hypothetical(self):
    """strategy that guesses a square and sees if the puzzle can be completed"""
    if not self.numbers: raise Solved()
    i = self.square.index(0)
    for v in self.numbers:
      new = self.clone()
      new.set(i, v)
      if new.solve():
        self.become(new)
        return True
    raise Impossible()

  # solve the square
  def solve(self):
    """solve the puzzle, returns True if a solution is found"""
    try:
      while True:
        while self.complete(): pass
        self.hypothetical()
    except (Impossible, Solved):
      pass
    if len(self.numbers) > 0: return False
    for line in self.lines:
      if sum(self.square[i] for i in line) != self.s: return False
    return True

###############################################################################

# Substituted Sum Solver

# originally written for Enigma 63, but applicable to lots of Enigma puzzles

# a substituted sum solver
# terms - list of summands of the sum (each the same length as result)
# result - the result of the sum (sum of the terms)
# digits - set of unallocated digits
# l2d - map from letters to allocated digits
# d2i - map from digits to letters that cannot be allocated to that digit
# n - column we are working on (string index in result)
# carry - carry from the column to the right
# base - base we are working in
# solutions are returned as assignments of letters to digits (the l2d dict)
def _substituted_sum(terms, result, digits, l2d, d2i, n, carry=0, base=10):
  # are we done?
  if n == 0:
    if carry == 0:
      l2d.pop('_', None)
      yield l2d
    return
  # move on to the next column
  n -= 1
  # find unallocated letters in this column
  u = list(uniq(t[n] for t in terms if t[n] not in l2d))
  # and allocate them from the remaining digits
  for ds in itertools.permutations(digits, len(u)):
    _l2d = update(l2d, u, ds)
    # sum the column
    (c, r) = divmod(sum(_l2d[t[n]] for t in terms) + carry, base)
    # is the result what we expect?
    if result[n] in _l2d:
      # the digit of the result is already allocated, check it
      if _l2d[result[n]] != r: continue
      allocated = ds
    else:
      # the digit in the result is one we haven't come across before
      if r not in digits or r in ds: continue
      _l2d[result[n]] = r
      allocated = ds + (r,)
    # check there are no invalid allocations
    if any(any(_l2d[x] == d for x in ls if x in _l2d) for (d, ls) in d2i.items()): continue
    # try the next column
    for r in _substituted_sum(terms, result, digits.difference(allocated), _l2d, d2i, n, c, base):
      yield r

def substitute(s2d, text, digits=None):
  """
  given a symbol-to-digit mapping <s2d> and some text <text>, return
  the text with the digits (as defined by the sequence <digits>)
  substituted for the symbols.

  characters in the text that don't occur in the mapping are unaltered.

  >>> substitute(dict(zip('DEMNORSY', (7, 5, 1, 6, 0, 8, 9, 2))), "SEND + MORE = MONEY")
  '9567 + 1085 = 10652'
  """
  if text is None: return None
  if digits is None: digits = base_digits()
  return join((digits[s2d[x]] if x in s2d else x) for x in text)

# friendly interface to the substituted sum solver
def substituted_sum(terms, result, digits=None, l2d=None, d2i=None, base=10):
  """
  a substituted addition sum solver - encapsulated by the SubstitutedSum class.

  terms - list of summands in the sum
  result - result of the sum (sum of the terms)
  digits - digits to be allocated (default: 0 - base-1, less any allocated digits)
  l2d - initial allocation of digits (default: all digits unallocated)
  d2i - invalid allocations (default: leading digits cannot be 0)
  base - base we're working in (default: 10)
  """
  # fill out the parameters
  if l2d is None:
    l2d = dict()
  if digits is None:
    digits = range(base)
  digits = set(digits).difference(l2d.values())
  if d2i is None:
    d2i = dict()
    d2i[0] = set(x[0] for x in itertools.chain(terms, [result]) if len(x) > 1)
  # number of columns in sum
  n = len(result)
  # make sure the terms are the same length as the result
  ts = list('_' * (n - len(t)) + t for t in terms)
  assert all(len(t) == n for t in ts), "result is shorter than terms"
  l2d['_'] = 0
  # call the solver
  for r in _substituted_sum(ts, result, digits, l2d, d2i, n, 0, base):
    yield r

# object interface to the substituted sum solver
class SubstitutedSum(object):
  """
  A solver for addition sums with letters substituted for digits.

  A substituted sum object has the following useful attributes:
    terms - the summands of the sum
    result - the result of the sum
    text - a textual form of the sum (e.g. "<term1> + <term2> = <result>")

  e.g. Enigma 21: "Solve: BPPDQPC + PRPDQBD = XVDWTRC"
  <https://enigmaticcode.wordpress.com/2012/12/23/enigma-21-addition-letters-for-digits/>
  >>> SubstitutedSum(['BPPDQPC', 'PRPDQBD'], 'XVDWTRC').go()
  BPPDQPC + PRPDQBD = XVDWTRC
  3550657 + 5850630 = 9401287 / B=3 C=7 D=0 P=5 Q=6 R=8 T=2 V=4 W=1 X=9

  e.g. Enigma 63: "Solve: LBRLQQR + LBBBESL + LBRERQR + LBBBEVR = BBEKVMGL"
  <https://enigmaticcode.wordpress.com/2013/01/08/enigma-63-addition-letters-for-digits/>
  >>> SubstitutedSum(['LBRLQQR', 'LBBBESL', 'LBRERQR', 'LBBBEVR'], 'BBEKVMGL').go()
  LBRLQQR + LBBBESL + LBRERQR + LBBBEVR = BBEKVMGL
  8308440 + 8333218 + 8302040 + 8333260 = 33276958 / B=3 E=2 G=5 K=7 L=8 M=9 Q=4 R=0 S=1 V=6
  """

  def __init__(self, terms, result, base=10, digits=None, l2d=None, d2i=None):
    """
    create a substituted addition sum puzzle.

    terms - a list of the summands of the sum.
    result - the result of the sum (i.e. the sum of the terms).

    The following parameters are optional:
    base - the number base the sum is in (default: 10)
    digits - permissible digits to be substituted in (default: determined from base)
    l2d - initial map of letters to digits (default: all letters unassigned)
    d2i - map of digits to invalid letter assignments (default: leading digits cannot be 0)

    If you want to allow leading digits to be 0 pass an empty dictionary for d2i.
    """

    text = join(terms, sep=' + ') + ' = ' + result

    self.terms = terms
    self.base = base
    self.result = result
    self.digits = digits
    self.l2d = l2d
    self.d2i = d2i

    self.text = text

  def solve(self, check=None, verbose=0):
    """
    generate solutions to the substituted addition sum puzzle.

    solutions are returned as a dictionary assigning letters to digits.
    """
    if verbose > 0:
      printf("{self.text}")

    for s in substituted_sum(self.terms, self.result, base=self.base, digits=self.digits, l2d=self.l2d, d2i=self.d2i):
      if check and not(check(s)): continue
      if verbose > 0:
        self.output_solution(s)
      yield s

  def substitute(self, s, text, digits=None):
    """
    given a solution to the substituted sum and some text return the text with
    letters substituted for digits.
    """
    return substitute(s, text, digits=digits)

  def output_solution(self, s, digits=None):
    """
    given a solution to the substituted sum output the assignment of letters
    to digits and the sum with digits substituted for letters.
    """
    printf("{t} / {s}",
      # print the sum with digits substituted in
      t=substitute(s, self.text, digits=digits),
      # output the assignments in letter order
      s=join((k + '=' + str(s[k]) for k in sorted(s.keys())), sep=' ')
    )

  solution = output_solution

  def go(self, check=None, first=0):
    """
    find all solutions (matching the filter <fn>) and output them.
    """
    for s in self.solve(check=check, verbose=1):
      if first: break

  # class method to chain multiple sums together
  @classmethod
  def chain(cls, sums, base=10, digits=None, l2d=None, d2i=None):
    """
    solve a sequence of substituted sum problems.

    sums are specified as a sequence of: (<term>, <term>, ..., <result>)
    """
    # are we done?
    if not sums:
      yield l2d
    else:
      # solve the first sum
      s = sums[0]
      for r in cls(s[:-1], s[-1], base=base, digits=digits, l2d=l2d, d2i=d2i).solve():
        # and recursively solve the rest
        for x in cls.chain(sums[1:], base=base, digits=digits, l2d=r, d2i=d2i): yield x

  @classmethod
  def chain_go(cls, sums, base=10, digits=None, l2d=None, d2i=None):
    template = join(('(' + join(s[:-1], sep=' + ') + ' = ' + s[-1] + ')' for s in sums), sep=' ')
    printf("{template}")
    for s in cls.chain(sums, base=base, digits=digits, l2d=l2d, d2i=d2i):
      printf("{t} / {s}",
        t=substitute(s, template),
        s=join((k + '=' + str(s[k]) for k in sorted(s.keys())), sep=' ')
      )

  # class method to call from the command line
  @classmethod
  def command_line(cls, args):
    """
    run the SubstitutedSum solver with the specified command line arguments.

    e.g. Enigma 327 <https://enigmaticcode.wordpress.com/2016/01/08/enigma-327-it-all-adds-up/>
    % python enigma.py SubstitutedSum "KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE"
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5


    Optional parameters:

    --base=<n> (or -b<n>)

    Specifies the numerical base of the sum, solutions are printed in the
    specified base (so that the letters and digits match up, but don't get
    confused by digits that are represented by letters in bases >10).

    e.g. Enigma 1663 <https://enigmaticcode.wordpress.com/2011/12/04/enigma-1663-flintoffs-farewell/>
    % python enigma.py SubstitutedSum --base=11 "FAREWELL + FREDALO = FLINTOFF"
    (FAREWELL + FREDALO = FLINTOFF)
    (61573788 + 657A189 = 68042966) / A=1 D=10 E=7 F=6 I=0 L=8 N=4 O=9 R=5 T=2 W=3
    (6157A788 + 6573189 = 68042966) / A=1 D=3 E=7 F=6 I=0 L=8 N=4 O=9 R=5 T=2 W=10


    --assign=<letter>,<digit> (or -a<l>,<d>)

    Assign a digit to a letter.

    There can be multiple --assign options.

    e.g. Enigma 1361 <https://enigmaticcode.wordpress.com/2013/02/20/enigma-1361-enigma-variation/>
    % python enigma.py SubstitutedSum --assign="O,0" "ELGAR + ENIGMA = NIMROD"
    (ELGAR + ENIGMA = NIMROD)
    (71439 + 785463 = 856902) / A=3 D=2 E=7 G=4 I=5 L=1 M=6 N=8 O=0 R=9


    --digits=<digit>,<digit>,... (or -d<d>,<d>,...)

    Specify the digits that can be assigned to unassigned letters.
    (Note that the values of the digits are specified in base 10, even if you
    have specified a --base=<n> option)

    e.g. Enigma 1272 <https://enigmaticcode.wordpress.com/2014/12/09/enigma-1272-jonny-wilkinson/>
    % python enigma.py SubstitutedSum --digits="0-8" "WILKI + NSON = JONNY"
    (WILKI + NSON = JONNY)
    (48608 + 3723 = 52331) / I=8 J=5 K=0 L=6 N=3 O=2 S=7 W=4 Y=1
    (48708 + 3623 = 52331) / I=8 J=5 K=0 L=7 N=3 O=2 S=6 W=4 Y=1


    --invalid=<digits>,<letters> (or -i<ds>,<ls>)

    Specify letters that cannot be assigned to a digit.

    There can be multiple --invalid options, but they should be for different
    <digit>s.

    If there are no --invalid options the default will be that leading zeros are
    not allowed. If you want to allow them you can give a "--invalid=0," option.

    Enigma 171 <https://enigmaticcode.wordpress.com/2014/02/23/enigma-171-addition-digits-all-wrong/>
    % python enigma.py SubstitutedSum -i0,016 -i1,1 -i3,3 -i5,5 -i6,6 -i7,7 -i8,8 -i9,9 "1939 + 1079 = 6856"
    (1939 + 1079 = 6856)
    (2767 + 2137 = 4904) / 0=1 1=2 3=6 5=0 6=4 7=3 8=9 9=7


    --help (or -h)

    Prints a usage summary.
    """

    usage = join((
      sprintf("usage: {cls.__name__} [<opts>] \"<term> + <term> + ... = <result>\" ..."),
      "options:",
      "  --base=<n> (or -b<n>) = set base to <n>",
      "  --assign=<letter>,<digit> (or -a<l>,<d>) = assign digit to letter",
      "  --digits=<digit>,... or <digit>-<digit> (or -d...) = available digits",
      "  --invalid=<digits>,<letters> (or -i<ds>,<ls>) = invalid digit to letter assignments",
      "  --help (or -h) = show command-line usage",
    ), sep=nl)

    # process options (--<key>[=<value>] or -<k><v>)
    opt = dict(l2d=dict(), d2i=None)
    while args and args[0].startswith('-'):
      arg = args.pop(0)
      try:
        if arg.startswith('--'):
          (k, _, v) = arg.lstrip('-').partition('=')
        else:
          (k, v) = (arg[1], arg[2:])
        if k == 'h' or k == 'help':
          print(usage)
          return -1
        elif k == 'b' or k == 'base':
          # --base=<n> (or -b)
          opt['base'] = int(v)
        elif k == 'a' or k == 'assign':
          # --assign=<letter>,<digit> (or -a)
          (l, d) = v.split(',', 1)
          opt['l2d'][l] = int(d)
        elif k == 'd' or k == 'digits':
          # --digits=<digit>,... or <digit>-<digit> (or -d)
          opt['digits'] = _digits(v)
        elif k == 'i' or k == 'invalid':
          # --invalid=<digits>,<letters> (or -i<ds>,<ls>)
          if opt['d2i'] is None: opt['d2i'] = dict()
          (ds, s) = _split(v, maxsplit=-1)
          for i in _digits(ds):
            opt['d2i'][i] = opt['d2i'].get(i, set()).union(s)
        else:
          raise ValueError()
      except:
        printf("{cls.__name__}: invalid option: {arg}")
        return -1

    # check command line usage
    if not args:
      print(usage)
      return -1

    # extract the sums
    import re
    sums = list(re.split(r'[\s\+\=]+', arg) for arg in args)

    # call the solver
    cls.chain_go(sums, **opt)
    return 0

###############################################################################

# Generic Substituted Expression Solver

# be aware that the generated code can fall foul of restrictions in the Python
# interpreter.
#
# the standard Python interpreter will throw:
#
#   "SystemError: too many statically nested blocks"; or:
#   "SyntaxError: too many statically nested blocks"
#
# if there are more than 20 levels of nested loops.
#
# and it will also throw:
#
#   "IndentationError: too many levels of indentation"
#
# if there are more than 100 levels of indentation.
#
# the PyPy interpreter has neither of these limitations

# TODO: think about negative values
#
# TODO: consider ordering the symbols, so we can calculate words sooner.
#
# TODO: consider allowing a "wildcard" character, for symbols that can
# take on any available digit (but still not allow leading zeros). [E1579]

_SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

import re

# find words in string <s>
def _find_words(s, r=1):
  words = set(re.findall(r'{(\w+?)}', s))
  if r:
    # return the words
    return words
  else:
    # return individual characters
    return set().union(*words)

# replace words in string <s>
def _replace_words(s, fn):
  # new style, with braces
  _fn = lambda m: fn(m.group(1))
  return re.sub(r'{(\w+?)}', _fn, s)

# return an expression that evaluates word <w> in base <base>
def _word(w, base):
  (m, d) = (1, dict())
  for x in w[::-1]:
    d[x] = d.get(x, 0) + m
    m *= base
  return join((concat(('_', k) + (() if v == 1 else ('*', v))) for (k, v) in d.items()), sep=' + ')

# simulate a function static variable
#
# it would be nice to do:
#
# def gensym(x):
#   static i = 0
#   i += 1
#   return concat(x, i)
#
# but this achieves the same ends using function attributes
#

@static(i=0)
def gensym(x):
  gensym.i += 1
  return concat(x, gensym.i)

# file.writelines does NOT include newline characters
def writelines(fh, lines, sep=None, flush=1):
  if sep is None:
    sep = os.linesep
  for line in lines:
    fh.write(line)
    fh.write(sep)
  if flush: fh.flush()

# split string <s> on any of the characters in <sep>
_split_sep = ',|+'

def _split(s, sep=_split_sep, maxsplit=0):
  d = sep[0]
  if len(sep) > 1:
    # map all other separators to d
    s = re.sub('[' + re.escape(sep[1:]) + ']', d, s)
  if maxsplit == 0:
    return s.split(d)
  elif maxsplit > 0:
    return s.split(d, maxsplit)
  else:
    return s.rsplit(d, -maxsplit)

# a sequence of digit values may be specified (in decimal) as:
#   "<d>-<d>" = a range of digits
#   "<d>,...,<d>" "<d>|...|<d>" "<d>+...+<d>"
#   "<d>" = a single digit
# returns a sequence of integers
def _digits(s):
  # "<d>-<d>"
  if '-' in s:
    (a, _, b) = s.partition('-')
    return irange(int(a), int(b))
  # "<d>,...,<d>" "<d>|...|<d>" "<d>+...+<d>"
  # "<d>"
  return tuple(int(d) for d in _split(s, _split_sep))


class SubstitutedExpression(object):
  """
  A solver for Python expressions with symbols substituted for numbers.

  It takes a Python expression and then tries all possible ways of assigning
  symbols (by default the capital letters) in it to digits and returns those
  assignments which result in the expression having a True value.

  While this is slower than the specialised solvers, like SubstitutedSum(),
  it does allow for more general expressions to be evaluated.


  Enigma 1530 <https://enigmaticcode.wordpress.com/2012/07/09/enigma-1530-tom-daley/>
  >>> SubstitutedExpression('TOM * 13 = DALEY').go()
  (TOM * 13 = DALEY)
  (796 * 13 = 10348) / A=0 D=1 E=4 L=3 M=6 O=9 T=7 Y=8
  [1 solution]
  1


  See SubstitutedExpression.command_line() for more examples.
  """

  def __init__(self, exprs, base=10, symbols=None, digits=None, s2d=None, l2d=None, d2i=None, answer=None, template=None, solution=None, header=None, distinct=1, check=None, env=None, code=None, process=1, reorder=1, first=0, verbose=1):
    """
    create a substituted expression solver.

    exprs - the expression(s)

    exprs can be a single expression, or a sequence of expressions.

    A single expression is of the form:

      "<expr>" or "<expr> = <value>" or (<expr>, <value>)

    where value is a valid "word" (sequence of symbols), or an integer value.

    The following parameters are optional:
    base - the number base to operate in (default: 10)
    symbols - the symbols to substituted in the expression (default: upper case letters)
    digits - the digits to be substituted in (default: determined from base)
    s2d - initial map of symbols to digits (default: all symbols unassigned)
    d2i - map of digits to invalid symbol assignments (default: leading digits cannot be 0)
    answer - an expression for the answer value
    distinct - symbols which should have distinct values (1 = all, 0 = none) (default: 1)
    check - a boolean function used to accept/reject solutions (default: None)
    env - additional environment for evaluation (default: None)
    code - additional lines of code evaluated before solving (default: None)

    If you want to allow leading digits to be 0 pass an empty dictionary for d2i.
    """

    self.exprs = exprs
    self.base = base
    self.symbols = symbols
    self.digits = digits
    self.s2d = (s2d or l2d) # s2d is preferred
    self.d2i = d2i
    self.answer = answer
    self.template = template
    self.solution = solution
    self.header = header
    self.distinct = distinct
    self.check = check
    self.env = env
    self.code = code

    self.process = process
    self.reorder = reorder
    self.first = first
    self.verbose = verbose

    # set by process
    self._processed = 0

    # set by prepare()
    self._prepared = 0

    if process: self._process()


  # sort out calling methods
  def _process(self):

    exprs = self.exprs
    base = self.base
    symbols = self.symbols
    digits = self.digits
    s2d = self.s2d
    d2i = self.d2i
    answer = self.answer
    template = self.template
    distinct = self.distinct
    process = self.process
    verbose = self.verbose

    # sort out verbose argument
    #   4 = output solutions
    #   8 = output header template
    #  16 = output solution count
    #  32 = output timing
    #  64 = output parameters
    # 128 = output solver info
    # 256 = output code
    # ---
    # 508 = all of the above
    if verbose and verbose < 4:
      # old style verbose flags (1, 2, 3)
      v1 = (4 | 8 | 16) # header + solutions + count
      v2 = (v1 | 128) # + solver info
      v3 = (v2 | 32 | 256) # + timing + code
      verbose = (0, v1, v2, v3)[verbose]

    # the symbols to replace (for implicit expressions)
    if symbols is None: symbols = _SYMBOLS

    # process expr to be a list of (<expr>, <value>) pairs, where:
    # <value> is:
    # None = look for a true value
    # word = look for a value equal to the substituted word
    # integer = look for the specific value
    if process:

      # allow expr to be a single string
      if isinstance(exprs, basestring): exprs = [exprs]

      # function fix up implicit parameters
      def fix(s):
        if s is None: return None
        if '{' in s: return s
        return re.sub('[' + symbols + ']+', (lambda m: '{' + m.group(0) + '}'), s)

      # now process the list
      xs = list()
      for expr in exprs:
        if isinstance(expr, basestring):
          # expression is a single string, turn it into an (<expr>, <value>) pair
          (v, s) = ('', re.split(r'\s+=\s+', expr))
          if len(s) == 2: (expr, v) = s
          if not v: v = None
        else:
          # assume expr is already an (<expr>, <value>) pair
          (expr, v) = expr

        # convert implicit (without braces) into explicit (with braces)
        if symbols:
          if isinstance(v, basestring) and '{' not in v and '{' not in expr and all(x in symbols for x in v): v = '{' + v + '}'
          expr = fix(expr)

        # value is either an alphabetic or a numeric literal
        if isinstance(v, basestring) and '{' not in v:
          v = base2int(v, base=base)

        xs.append((expr, v))

      exprs = xs

      # fix up implicit (old style) parameters
      answer = fix(answer)
      template = fix(template)

    # make the output template (which is kept in input order)
    # and also categorise the expressions to (<expr>, <value>, <cat>), where <cat> is:
    # 0 = answer (<value> = None)
    # 1 = expression with no value, we look for a true (<value> = None)
    # 2 = expression with an integer value, we do a direct comparison (<value> = int)
    # 3 = expression with string value, we look to assign/check symbols (<value> = str)
    ts = list()
    xs = list()
    for (x, v) in exprs:
      if v is None:
        ts.append(x)
        xs.append((x, v, 1))
      elif isinstance(v, basestring):
        ts.append(x + ' = ' + v)
        xs.append((x, v[1:-1], 3))
      else:
        ts.append(x + ' = ' + int2base(v, base=base))
        xs.append((x, v, 2))
    if answer:
      ts.append(answer)
      xs.append((answer, None, 0))
    _template = join(('(' + t + ')' for t in ts), sep=' ')
    exprs = xs

    # initial mapping of symbols to digits
    if s2d is None: s2d = dict()

    # allowable digits (and invalid digits)
    if digits is None: digits = range(base)
    digits = set(digits)
    # TODO: I suspect this needs to work with more values of "distinct"
    if distinct == 1: digits = digits.difference(s2d.values())
    idigits = set(range(base)).difference(digits)

    # find words in all exprs
    words = _find_words(_template)
    # and determine the symbols that are used
    symbols = join(sorted(set().union(*words)))

    # invalid (<symbol>, <digit>) assignments
    invalid = set()
    if d2i is not None:
      # it should provide a sequence of (<digit>, <symbol[s]>) pairs
      for (d, ss) in (d2i.items() if hasattr(d2i, 'items') else d2i):
        if d not in digits: printf("WARNING: SubstitutedExpression: non-valid invalid digit {d} specified", d=repr(d))
        invalid.update((s, d) for s in ss)
    else:
      # disallow leading zeros
      if 0 in digits:
        for w in words:
          if len(w) > 1:
            invalid.add((w[0], 0))

    # but for the rest of the time we are only interested
    # in words in the <exprs> (not the <values>)
    words = set()
    for (x, _, _) in exprs:
      words.update(w for w in _find_words(x) if len(w) > 1)

    # find the symbols in the (<expr>, <value>) pairs
    # xs = symbols in <expr>
    # vs = symbols in <value>
    (xs, vs) = (list(), list())
    for (x, v, k) in exprs:
      xs.append(_find_words(x, r=0))
      vs.append(set(v) if k == 3 else set())

    # determine the symbols in each expression
    syms = list(x.union(v) for (x, v) in zip(xs, vs))

    # sort out distinct=0,1
    if type(distinct) is int: distinct = (symbols if distinct else '')
    # distinct should be a sequence (probably of strings)
    if isinstance(distinct, basestring): distinct = [distinct]

    # add the value of the symbols into the template
    self.template = (_template if template is None else template)
    if self.solution is None: self.solution = symbols
    if self.header is None: self.header = _replace_words(self.template, identity)

    # update the processed values
    self.exprs = exprs
    self.symbols = symbols
    self.digits = digits
    self.s2d = s2d
    self.answer = answer
    self.distinct = distinct
    self.verbose = verbose
    self._words = words
    self._invalid = invalid
    self._idigits = idigits
    self._exprs = (exprs, xs, vs, ts, syms)
    self._processed = 1

  # create and compile the code
  # NOTE: the generated code can have more than 20 nested blocks,
  # which raises a SyntaxError in CPython. A workaround is to use PyPy
  # instead (which doesn't seem to have this limitation)
  def _prepare(self):

    base = self.base
    symbols = self.symbols
    digits = self.digits
    s2d = self.s2d
    d2i = self.d2i
    answer = self.answer
    distinct = self.distinct
    env = self.env
    code = self.code
    reorder = self.reorder
    verbose = self.verbose

    words = self._words
    invalid = self._invalid
    idigits = self._idigits
    (exprs, xs, vs, ts, syms) = self._exprs

    # output run parameters
    if self.verbose & 64:
      print("--[code]--" + nl + join(self.save(quote=1), sep=nl) + nl + "--[/code]--" + nl)

    # valid digits for each symbol
    valid = dict((s, list(digits.difference(d for (x, d) in invalid if x == s))) for s in symbols)
    #for k in sorted(valid.keys()): printf("{k} -> {v}", v=valid[k])

    # reorder the expressions into a more appropriate evaluation order
    if reorder:
      # at each stage chose the expression with the fewest remaining possibilities
      d = set(s2d.keys())
      (s, r) = (list(), list(i for (i, _) in enumerate(syms)))
      # formerly we used:
      #
      #                (  is answer?  )  (# of unassiged symbols)  -(number of new symbols we get)
      #fn = lambda i: (exprs[i][2] == 0, len(xs[i].difference(d)), -len(vs[i].difference(d, xs[i])))
      #
      # now we use:
      #
      #               (  is answer?  )  (    total possibilities for unassigned symbols    )  -(number of new symbols we get)
      fn = lambda i: (exprs[i][2] == 0, multiply(len(valid[x]) for x in xs[i] if x not in d), -len(vs[i].difference(d, xs[i])))
      while r:
        i = min(r, key=fn)
        s.append(i)
        d.update(xs[i], vs[i])
        r.remove(i)
      # update the lists
      exprs = list(exprs[i] for i in s)
      xs = list(xs[i] for i in s)
      vs = list(vs[i] for i in s)
      ts = list(ts[i] for i in s)

    if verbose & 128:
      # output solver information
      printf("[base={base}, digits={digits}, symbols={symbols!r}, distinct={distinct!r}]", distinct=join(distinct, sep=','))
      printf("[s2d={s2d}, d2i={d2i}]")
      # output the solving strategy
      (ss, d) = (list(), set(s2d.keys()))
      for (i, x) in enumerate(xs):
        ss.append(sprintf("({e}) [{n}+{m}]", e=ts[i], n=len(x.difference(d)), m=len(vs[i].difference(d, x))))
        d.update(x, vs[i])
      printf("[strategy: {ss}]", ss=join(ss, sep=' -> '))

    # turn distinct into a dict mapping <symbol> -> <excluded symbols>
    if type(distinct) is not dict:
      d = dict()
      for ss in distinct:
        for s in ss:
          if s not in d: d[s] = set()
          d[s].update(x for x in ss if x != s)
      distinct = d

    # generate the program (line by line)
    (prog, _, indent) = ([], '', '  ')

    # start with any initialisation code
    if code:
      # code should be a sequence (of strings)
      if isinstance(code, basestring): code = [code]
      prog.extend(code)

    # wrap it all up as function solver
    solver = gensym('_substituted_expression_solver')
    prog.append(sprintf("{_}def {solver}():"))
    _ += indent

    # set initial values
    done = set()
    for (s, d) in s2d.items():
      prog.append(sprintf("{_}_{s} = {d}"))
      done.add(s)

    # look for words which can be made
    for w in words:
      if all(x in done for x in w):
        prog.append(sprintf("{_}_{w} = {x}", x=_word(w, base)))

    in_loop = False

    # deal with each <expr>,<value> pair
    for ((expr, val, k), xsyms, vsyms) in zip(exprs, xs, vs):

      # deal with each symbol in <expr>
      # TODO: we could consider these in an order that makes words
      # in <words> as soon as possible
      for s in xsyms:
        if s in done: continue
        # allowable digits for s
        ds = valid[s]
        in_loop = True
        prog.append(sprintf("{_}for _{s} in {ds}:"))
        _ += indent
        if done and s in distinct:
          # TODO: we should exclude initial values (that are excluded from ds) here
          check = join((('_' + s + ' != ' + '_' + x) for x in done if x in distinct[s]), sep=' and ')
          if check:
            prog.append(sprintf("{_}if {check}:"))
            _ += indent
        done.add(s)
        # look for words which can now be made
        for w in words:
          if s in w and all(x in done for x in w):
            prog.append(sprintf("{_}_{w} = {x}", x=_word(w, base)))

      # calculate the expression
      if k != 0: # (but not for the answer expression)
        x = _replace_words(expr, (lambda w: '(' + '_' + w + ')'))
        prog.append(sprintf("{_}try:"))
        prog.append(sprintf("{_}  x = int({x})"))
        prog.append(sprintf("{_}except NameError:")) # catch undefined functions
        prog.append(sprintf("{_}  raise"))
        prog.append(sprintf("{_}except:")) # maybe "except (ArithmeticError, ValueError)"
        prog.append(sprintf("{_}  {skip}", skip=('continue' if in_loop else 'pass')))

      # check the value
      if k == 3:
        # this is a literal word
        for (j, y) in enumerate(val[::-1], start=-len(val)):
          if y in done:
            # this is a symbol with an assigned value
            prog.append(sprintf("{_}y = x % {base}"))
            # check the value
            prog.append(sprintf("{_}if y == _{y}:"))
            _ += indent
            prog.append(sprintf("{_}x //= {base}"))
            # and check x == 0 for the final value
            if j == -1:
              prog.append(sprintf("{_}if x == 0:"))
              _ += indent
          else:
            # this is a new symbol...
            prog.append(sprintf("{_}_{y} = x % {base}"))
            check = list()
            # check it is different from existing symbols
            if y in distinct:
              check.extend((('_' + y + ' != ' + '_' + x) for x in done if x in distinct[y]))
            # check any invalid values for this symbol
            for v in idigits.union(v for (s, v) in invalid if y == s):
              check.append('_' + y + ' != ' + str(v))
            if check:
              check = join(check, sep=' and ')
              prog.append(sprintf("{_}if {check}:"))
              _ += indent
            prog.append(sprintf("{_}x //= {base}"))
            # and check x == 0 for the final value
            if j == -1:
              prog.append(sprintf("{_}if x == 0:"))
              _ += indent
            done.add(y)
            # look for words which can now be made
            for w in words:
              if y in w and all(x in done for x in w):
                prog.append(sprintf("{_}_{w} = {x}", x=_word(w, base)))

      elif k == 1:
        # look for a True value
        prog.append(sprintf("{_}if x:"))
        _ += indent

      elif k == 2:
        # it's a comparable value
        prog.append(sprintf("{_}if x == {val}:"))
        _ += indent

    # yield solutions as dictionaries
    d = join((("'" + s + "': _" + s) for s in sorted(done)), sep=', ')
    if answer:
      # compute the answer
      r = _replace_words(answer, (lambda w: '(' + '_' + w + ')'))
      prog.append(sprintf("{_}r = {r}"))
      prog.append(sprintf("{_}yield ({{ {d} }}, r)"))
    else:
      prog.append(sprintf("{_}yield {{ {d} }}"))

    # turn the program lines into a string
    prog = join(prog, sep=nl)

    if verbose & 256:
      printf("-- [code language=\"python\"] --{nl}{prog}{nl}-- [/code] --")

    # compile the solver
    # a bit of jiggery pokery to make this work in several Python versions
    # older Python barfs on:
    #   ns = dict()
    #   eval(prog, None, ns)
    #   solve = ns[solver]
    if not env: env = dict()
    gs = update(globals(), env)
    code = compile(prog, '<string>', 'exec')
    eval(code, gs)

    self._solver = gs[solver]
    self._prepared = 1


  # execute the code
  def solve(self, check=None, first=None, verbose=None):
    """
    generate solutions to the substituted expression problem.

    solutions are returned as a dictionary assigning symbols to digits.

    check - a boolean function called to reject unwanted solutions
    first - if set to positive <n> only the first <n> solutions are returned
    verbose - if set to >0 solutions are output as they are found, >1 additional information is output.
    """

    if not self._prepared: self._prepare()

    solver = self._solver
    answer = self.answer
    header = self.header
    if check is None: check = self.check
    if first is None: first = self.first
    if verbose is None: verbose = self.verbose

    if verbose & 8 and header: print(header)

    n = 0
    for s in solver():
      if check and not(check(s)): continue
      if verbose & 4: self.output_solution((s[0] if answer else s))
      # return the result
      yield s
      n += 1
      if first and first == n: break

  # output a solution as: "<template> / <solution>"
  # <template> = the given template with digits substituted for symbols
  # <solution> = the assignment of symbols (given in <solution>) to digits (in base 10)
  def output_solution(self, d, template=None, solution=None):
    if template is None: template = self.template
    if solution is None: solution = self.solution
    # output the solution using the template
    ss = list()
    if template:
      ss.append(_replace_words(template, (lambda w: substitute(d, w))))
    if solution:
      ss.append(join(((k + '=' + int2base(d[k], base=10)) for k in solution), sep=' '))
    print(join(ss, sep=' / '))


  def go(self, check=None, first=None, verbose=None):
    """
    find solutions to the substituted expression problem and output them.

    first - if set to True will stop after the first solution is output

    returns the number of solutions found, but if the "answer" parameter
    was set during init() returns a multiset() object counting
    the number of times each answer occurs.
    """
    if verbose is None: verbose = self.verbose

    # collect answers (either total number or collected by "answer")
    answer = self.answer
    r = (multiset() if answer else 0)

    # measure internal time
    if verbose & 32:
      t = Timer()
      t.start()

    # solve the problem, counting the answers
    for s in self.solve(check=check, first=first, verbose=verbose):
      if answer:
        r.add(s[1])
      else:
        r += 1

    if verbose & 32: t.stop()

    # output solutions
    if verbose & 16:
      if answer:
        answer = _replace_words(answer, identity)
        # report the answer counts
        for (k, v) in r.most_common():
          printf("{answer} = {k} [{v} solution{s}]", s=('' if v == 1 else 's'))
      else:
        printf("[{r} solution{s}]", s=('' if r == 1 else 's'))

    if verbose & 32: t.report()

    return r


  def substitute(self, s, text, digits=None):
    """
    given a solution to the substituted expression sum and some text,
    return the text with the letters substituted for digits.
    """
    return substitute(s, text, digits=digits)


  # generate appropriate command line arguments to reconstruct this instance
  def to_args(self, quote=1):

    if quote == 0:
      q = ''
    elif quote == 1:
      q = '"'
    else:
      q = quote

    args = []

    if self.base:
      args.append(sprintf("--base={self.base}"))

    if self.symbols:
      args.append(sprintf("--symbols={q}{self.symbols}{q}"))

    if self.distinct is not None:
      distinct = self.distinct
      if not isinstance(distinct, int):
        if not isinstance(self.distinct, basestring):
          distinct = join(distinct, sep=",")
        distinct = q + distinct + q
      args.append(sprintf("--distinct={distinct}"))

    if self.digits:
      args.append(sprintf("--digits={q}{digits}{q}", digits=join(self.digits, sep=",")))

    if self.s2d:
      for (k, v) in sorted(self.s2d.items(), key=lambda t: t[1]):
        args.append(sprintf("--assign={q}{k},{v}{q}"))

    # we should probably make this from self.invalid
    if self.d2i:
      for (k, v) in sorted(self.d2i.items(), key=lambda t: t[0]):
        args.append(sprintf("--invalid={q}{k},{v}{q}", v=join(sorted(v))))

    if self.answer:
      args.append(sprintf("--answer={q}{self.answer}{q}"))

    if self.template:
      args.append(sprintf("--template={q}{self.template}{q}"))

    if self.solution:
      args.append(sprintf("--solution={q}{self.solution}{q}"))

    if self.header:
      args.append(sprintf("--header={q}{self.header}{q}"))

    if self.env is not None:
      raise ValueError("can't generate arg for \"env\" parameter (maybe use \"code\" instead)")

    code = self.code
    if code:
      if isinstance(code, basestring):
        code = [code]
      for x in code:
        if q: x = x.replace(q, "\\" + q) # TODO: check quoting works
        args.append(sprintf("--code={q}{x}{q}"))

    if self.reorder is not None:
      args.append(sprintf("--reorder={self.reorder}"))

    if self.first is not None:
      args.append(sprintf("--first={self.first}"))

    if self.verbose is not None:
      args.append(sprintf("--verbose={self.verbose}"))

    # and the expressions
    for (x, v, k) in self.exprs:
      if k == 1:
        args.append(q + x + q)
      elif k == 2:
        args.append(q + x + " = " + int2base(v, self.base) + q)
      elif k == 3:
        args.append(q + x + " = {" + v + "}" + q)

    return args

  # generate appropriate command line arguments to reconstruct this instance
  def save(self, file=None, quote=1):

    args = self.to_args(quote=quote)
    if not args: raise ValueError()

    args.insert(0, "SubstitutedExpression") # self.__class__.__name__

    if file is None:
      # just return the args
      pass
    elif isinstance(file, basestring):
      # treat the string as a filename
      with open(file, 'wt') as f:
        writelines(f, args)
    else:
      # assume a file handle has been passed
      writelines(file, args)

    return args

  # usage strings
  @classmethod
  def _usage(cls):

    return (
      "usage: SubstitutedExpression [<opts>] <expression> [<expression> ...]",
      "options:",
      "  --symbols=<string> (or -s<string>) = symbols to replace with digits",
      "  --base=<n> (or -b<n>) = set base to <n>",
      "  --assign=<symbol>,<decimal> (or -a<s>,<d>) = assign decimal value to symbol",
      "  --digits=<digit>,... or --digits=<digit>-<digit> (or -d...) = available digits",
      "  --invalid=<digits>,<symbols> (or -i<ds>,<ss>) = invalid digit to symbol assignments",
      "  --answer=<expr> (or -A<expr>) = count answers according to <expr>",
      "  --template=<string> (or -T<s>) = solution template",
      "  --solution=<string> (or -S<s>) = solution symbols",
      "  --header=<string> (or -H<s>) = solution header",
      "  --distinct=<string> (or -D<s>) = symbols that stand for different digits (0 = off, 1 = on)",
      "  --code=<string> (or -C<s>) = initialisation code (can be used multiple times)",
      "  --first (or -1) = stop after the first solution",
      "  --reorder=<n> (or -r<n>) = allow reordering of expressions (0 = off, 1 = on)",
      "  --verbose[=<n>] (or -v[<n>]) = verbosity (0 = off, 1 = solutions, 2+ = more)",
      "  --help (or -h) = show command-line usage",
      "",
      "verbosity levels:",
      "    4 = output solutions (1,2,3)",
      "    8 = output header (1,2,3)",
      "   16 = output solution count (1,2,3)",
      "   32 = output timing info (3)",
      "   64 = output parameters",
      "  128 = output solver info (2,3)",
      "  256 = output Python code (3)",
      "",
    )

  # process option <k> = <v> into <opt>, returns:
  #   None = help
  #   True = option processed
  #   Exception = error
  @classmethod
  def _getopt(cls, k, v, opt):

    if k == 'h' or k == 'help':
      # --help (or -h)
      return None
    elif k == 's' or k == 'symbols':
      # --symbols=<string> (or -s<string>)
      opt['symbols'] = v
    elif k == 'T' or k == 'template':
      opt['template'] = v
    elif k == 'S' or k == 'solution':
      opt['solution'] = v
    elif k == 'H' or k == 'header':
      opt['header'] = v
    elif k == 'b' or k == 'base':
      # --base=<n> (or -b)
      opt['base'] = int(v)
    elif k == 'a' or k == 'assign':
      # --assign=<letter>,<digit> (or -a<letter>,<digit>)
      # NOTE: <digit> is specified in decimal (not --base)
      (l, d) = v.split(',', 1)
      opt['s2d'][l] = int(d)
    elif k == 'd' or k == 'digits':
      # --digits=<digit>,... or <digit>-<digit> (or -d)
      # NOTE: <digits> are specified in decimal (not --base)
      opt['digits'] = _digits(v)
    elif k == 'i' or k == 'invalid':
      # --invalid=<digits>,<letters> (or -i<ds>,<ls>)
      # NOTE: <digits> are specified in decimal (not --base)
      if opt['d2i'] is None: opt['d2i'] = dict()
      if v == '': return True # empty value will allow leading zeros
      (ds, s) = _split(v, maxsplit=-1)
      for i in _digits(ds):
        opt['d2i'][i] = opt['d2i'].get(i, set()).union(s)
    elif k == 'D' or k == 'distinct':
      if v == '0' or v == '1':
        v = int(v)
      else:
        v = _split(v)
      opt['distinct'] = v
    elif k == 'C' or k == 'code':
      if 'code' not in opt: opt['code'] = []
      opt['code'].append(v)
    elif k == 'A' or k == 'answer':
      opt['answer'] = v
    elif k == '1' or k == 'first':
      opt['first'] = (int(v) if v else 1)
    elif k == 'v' or k == 'verbose':
      opt['verbose'] = (sum(_digits(v)) if v else 1)
    elif k == 'r' or k == 'reorder':
      opt['reorder'] = (int(v) if v else 0)

    else:
      # unrecognised option
      raise ValueError()

    return True


  # class method to make an object from arguments
  @classmethod
  def from_args(cls, args):

    #if not args: return

    # process options
    opt = dict(_argv=list(), s2d=dict(), d2i=None, verbose=1, first=0, reorder=1)
    for arg in args:
      # deal with option args
      try:
        if arg.startswith('--'):
          (k, _, v) = arg.lstrip('-').partition('=')
        elif arg.startswith('-'):
          (k, v) = (arg[1], arg[2:])
        else:
          # push non-option args onto _argv
          opt['_argv'].append(arg)
          continue

        if not cls._getopt(k, v, opt):
          return None

      except:
        raise ValueError(sprintf("[{cls.__name__}] invalid option: {arg}"))
    return opt

  # class method to make an object from a file
  @classmethod
  def from_file(cls, file):
    (cmd, args) = parsefile(file)
    assert cmd == cls.__name__
    return cls(args)

  # class method to call from the command line
  @classmethod
  def command_line(cls, args):
    """
    run the SubstitutedExpression solver with the specified command
    line arguments.

    we can solve substituted sum problems (although using
    SubstitutedSum would be faster)


    Enigma 327 <https://enigmaticcode.wordpress.com/2016/01/08/enigma-327-it-all-adds-up/>
    % python enigma.py SubstitutedExpression "KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE"
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5
    [1 solution]

    but we can also use SubstitutedExpression to solve problems that
    don't have a specialsed solver.

    e.g. Sunday Times Teaser 2803
    % python enigma.py SubstitutedExpression --answer="ABCDEFGHIJ" "AB * CDE = FGHIJ" "AB + CD + EF + GH + IJ = CCC"
    (AB * CDE = FGHIJ) (AB + CD + EF + GH + IJ = CCC)
    (52 * 367 = 19084) (52 + 36 + 71 + 90 + 84 = 333) / A=5 B=2 C=3 D=6 E=7 F=1 G=9 H=0 I=8 J=4 / 5236719084
    ABCDEFGHIJ = 5236719084 [1 solution]

    e.g. Sunday Times Teaser 2796
    % python enigma.py SubstitutedExpression --answer="DRAGON" "SAINT + GEORGE = DRAGON" "E % 2 = 0"
    (SAINT + GEORGE = DRAGON) (E % 2 = 0)
    (72415 + 860386 = 932801) (6 % 2 = 0) / A=2 D=9 E=6 G=8 I=4 N=1 O=0 R=3 S=7 T=5 / 932801
    DRAGON = 932801 [1 solution]

    we also have access to any of the routines defined in enigma.py:

    e.g. Enigma 1180 <https://enigmaticcode.wordpress.com/2016/02/15/enigma-1180-anomalies/>
    % python enigma.py SubstitutedExpression --answer="(FOUR, TEN)" "SEVEN - THREE = FOUR" "is_prime(SEVEN)" "is_prime(FOUR)" "is_prime(RUOF)" "is_square(TEN)"
    (SEVEN - THREE = FOUR) (is_prime(SEVEN)) (is_prime(FOUR)) (is_prime(RUOF)) (is_square(TEN))
    (62129 - 58722 = 3407) (is_prime(62129)) (is_prime(3407)) (is_prime(7043)) (is_square(529)) / E=2 F=3 H=8 N=9 O=4 R=7 S=6 T=5 U=0 V=1 / (3407, 529)
    (FOUR, TEN) = (3407, 529) [1 solution]
    """

    if args:
      opt = cls.from_args(args)
      if opt:
        # create the object
        argv = opt.pop('_argv')
        self = cls(argv, **opt)
        if self is not None:
          # call the solver
          self.go()
          return 0

    # failure, output usage message
    print(join(cls._usage(), sep=nl))
    return -1

  # class method to provide a read/eval/print loop
  @classmethod
  def repl(cls, args=(), timed=1):
    """
    Provide a read/eval/print loop for evaluating alphametics.

    Use the following command to invoke it:

      % python enigma.py Alphametic.repl

    timed=1 will time the evaluation.

    """

    try:
      import readline
    except ImportError:
      pass

    v = (4 | 8 | 16)
    if timed: v |= 32

    while True:

      # collect expressions
      exprs = []
      while True:
        try:
          expr = raw_input(sprintf("expr[{n}] (or enter) >>> ", n=len(exprs)))
        except EOFError:
          print("\n[done]")
          return
        expr = expr.strip()
        if expr == "" or expr == ".": break
        exprs.append(expr)

      if not any(x.startswith("--verbose") or x.startswith("-v") for x in exprs):
        exprs.insert(0, sprintf("--verbose={v}"))

      # solve alphametic expressions
      if exprs:
        try:
          cls.command_line(exprs)
        except Exception as e:
          print(e)
          print("[ERROR: try again]")
        print()




def substituted_expression(*args, **kw):
  if 'verbose' not in kw: kw['verbose'] = 0
  for r in SubstitutedExpression(*args, **kw).solve():
    yield r

###############################################################################

# Substituted Division Solver

# new solver that uses the SubstitutedExpression alphametic solver...

# first we need a class that manages a set of "slots"

# (UN, j) = slot j has been unified with this slot
# (EQ, d) = slot has value of digit d
# (NE, d) = slot is not digit d
# (IS, x) = slot has input symbol x
(_UN, _EQ, _NE, _IS) = ('UN', 'EQ', 'NE', 'IS')

_SYMBOLS_UL = _SYMBOLS + _SYMBOLS.lower()

class Slots(object):

  def __init__(self, wildcard='?', symbols=_SYMBOLS_UL):

    # wildcard character in input strings
    self.wildcard = wildcard

    # pool of valid symbols
    self.symbols = symbols

    # slot ids
    self._id = 0

    # slot properties
    self._s2p = collections.defaultdict(set) # <slot> -> <props>
    self._p2s = collections.defaultdict(lambda: collections.defaultdict(set)) # <type> -> <value> -> <slots>

  # allocate a new slot (with (k, v) properties)
  def slot_new(self, *props):
    self._id += 1
    i = self._id
    self.slot_setprops(i, *props)
    return i

  def slot_setprops(self, i, *props):
    ps = self._s2p[i] # properties for slot i
    for (k, v) in props:

      if k == _EQ:
        # incompatible with (NE, v)
        if (_NE, v) in ps: raise ValueError("property mismatch")
        # incompatible with (EQ, u) where u != v
        if any(k1 == _EQ and v1 != v for (k1, v1) in ps): raise ValueError("property mismatch")

      elif k == _NE:
        # incompatible with (EQ, v)
        if (_EQ, v) in ps: raise ValueError("property mismatch")

      # add the properties
      ps.add((k, v))
      self._p2s[k][v].add(i)

  # find (or create) a slot with this property
  def slot_find(self, k, v, create=1):
    # return the lowest numbered slot we find
    try:
      return min(self._p2s[k][v])
    except ValueError:
      pass
    # otherwise create a slot with this property
    if create:
      return self.slot_new((k, v))
    # otherwise there is no slot
    return None

  # allocate a slot for the input symbol <s>
  def _allocate(self, s):

    wildcard = self.wildcard
    symbols = self.symbols

    if s == wildcard:
      # wildcard character, allocate a new slot
      return self.slot_new()

    if s in '0123456789':
      # integer literal, use the same slot for the same literal
      return self.slot_find(_EQ, int(s))

    if s in symbols:
      # a symbol, use the same slot for the same input symbol
      return self.slot_find(_IS, s)

    # unrecognised input symbol
    raise ValueError(sprintf("_allocate: invalid input symbol <{s}>"))

  # allocate a collection of slots for the input terms <ts>
  def allocate(self, ts):
    if ts is None: return None
    return list((None if ss is None else list(self._allocate(s) for s in ss)) for ss in ts)

  # find the leader for this slot
  def _slot(self, i):
    return self.slot_find(_UN, i, create=0) or i

  # unify two slots <i> and <j>
  def _unify(self, i, j):
    i = self._slot(i)
    j = self._slot(j)
    if i == j: return
    (i, j) = (min(i, j), max(i, j))
    # copy any properties from slot j to slot i
    self.slot_setprops(i, *(self._s2p[j]))
    # mark slot j as being unified with i
    self.slot_setprops(i, (_UN, j))

  # unify two sequence of slots <s> and <t>
  def unify(self, s, t):
    assert len(s) == len(t)
    for (i, j) in zip(s, t):
      self._unify(i, j)

  # return the symbol for a slot
  def symbol(self, i):
    # use the (lowest) symbol from an IS property
    vs = sorted(v for (k, v) in self._s2p[self._slot(i)] if k == _IS)
    if vs:
      return vs[0]
    # use the next unallocated symbol
    for v in self.symbols:
      if not self.slot_find(_IS, v, create=0):
        self.slot_setprops(i, (_IS, v))
        return v
    raise ValueError("symbol pool exhausted")

  # return labels for a sequence of slots
  def label(self, ss):
    if ss is None: return None
    return list((None if s is None else join(self.symbol(x) for x in s)) for s in ss)

  # return properties as <value, slots>
  def prop_items(self, k):
    for (v, ss) in self._p2s[k].items():
      if ss:
        yield (v, ss)

  # return a string of the symbols currently assigned
  def symbols_used(self):
    return join(sorted(v for (v, ss) in self.prop_items(_IS)))

# a named tuple for the results (now includes "subs" field)
# (s is the solution from SubstituteExpression, with eliminated symbols reinstated)
SubstitutedDivisionSolution = collections.namedtuple('SubstitutedDivisionSolution', 'a b c r subs d s')

# the new solver

class SubstitutedDivision(SubstitutedExpression):
  """
  A solver for long division sums with letters substituted for digits.

  e.g. Enigma 206

            - - -
      ___________
  - - ) p k m k h
        p m d
        -----
          x p k
            - -
          -----
            k h h
            m b g
            -----
                k
            =====

  In this example there are the following intermediate (subtraction) sums:

    pkm - pmd = xp, xpk - ?? = kh, khh - mbg = k

  When the result contains a 0 digit there is no corresponding
  intermediate sum, in this case the intermediate sum is specified as None.


  Enigma 206 <https://enigmaticcode.wordpress.com/2014/07/13/enigma-206-division-some-letters-for-digits-some-digits-missing/>

  >>> SubstitutedDivision('pkmkh / ?? = ???', ['pkm - pmd = xp', 'xpk - ?? = kh', 'khh - mbg = k']).go()
  pkmkh / ?? = ??? (rem k) [pkm - pmd = xp, xpk - ?? = kh, khh - mbg = k]
  47670 / 77 = 619 (rem 7) [476 - 462 = 14, 147 - 77 = 70, 700 - 693 = 7] / b=9 d=2 g=3 h=0 k=7 m=6 p=4 x=1
  [1 solution]
  1


  See SubstitutedDivision.command_line() for more examples.
  """

  def __init__(self, *args, **kw):
    """
    create a substituted long division solver.

    args - the long division sum to solve.

    a long division sum is considered to have the following components:

      a / b = c remainder r

      the dividend = a
      the divisor = b
      the result = c

      along with a set of intermediate subtractions of the form:

      x - y = z

      one sum for each digit in the result (a 0 digit in the result
      corresponds to an empty intermediate subtraction sum, which is
      specified using None).

      the sum is specified in one of the following ways:

        1. each component is separated out
        (this is how the old SubstitutedDivision() solver was called)

        args = (<a>, <b>, <c>, [(<x>, <y>, <z>), ... ])

        2. the sums are specified as strings:

        args = ("<a> / <b> = <c>", ["<x> - <y> = <z>", ...])

        or:

        args = ("<a> / <b> = <c>", "<x> - <y> = <z>", ...)


    the following keyword arguments from SubstitutedExpression() can be used:

      digits - specify digits to be substituted
      distinct - specify sets of symbols whose values should be distinct
      d2i - invalid digit to symbol map
      s2d - initial symbol to digit map
      answer - collect solutions by answer
      verbose - control output of solutions and tourist information
    """

    # sort out various argument formats

    split = None

    if len(args) == 4:
      # (preferred)
      # arguments are already broken down
      # args = (a, b, c, [(x, y, z), ... ])
      (a, b, c, subs) = args
      subs = list((None if x is None else list(x)) for x in subs)

    elif len(args) == 2:
      # arguments are passed as strings
      # args = ("{a} / {b} = {c}", ["{x} - {y} = {z}", ... ])
      split = args

    elif len(args) == 1:
      # args are passed as a list of strings (probably from the command line)
      # args = (["{a} / {b} = {c}", "{x} - {y} = {z}", ...],)
      args = args[0]
      split = (args[0], args[1:])

    else:
      raise ValueError("invalid arguments")

    # split compound arguments into component parts
    if split:
      (div, subs) = split
      debrace = (lambda x: re.sub(r'[\{\}]', '', x))
      (div, subs) = (debrace(div), list((None if x is None else debrace(x)) for x in subs))
      (a, b, c) = re.split(r'[\s\/\=]+', div)
      subs = list(((re.split(r'[\s\-\=]+', x) if isinstance(x, basestring) else x) if x else None) for x in subs)

    def fmt(v, brace=0, none="0"):
      return (none if v is None else ('{' + v + '}' if brace else v))

    def fmt_subs(subs, brace=0, sep=", "):
      s = list()
      for t in subs:
        if t is None:
          s.append("None")
        else:
          (x, y, z) = (fmt(v, brace=brace) for v in t)
          s.append(x + ' - ' + y + ' = ' + z)
      return join(s, sep=sep)

    # we use None instead of 0 if the result comes out exactly
    # and extract the remainder
    rem = []
    for i in irange(-1, -len(subs), step=-1):
      if subs[i]:
        if (not(subs[i][-1]) or subs[i][-1] == '0'): subs[i][-1] = None
        if subs[i][-1] is not None: rem.insert(0, subs[i][-1])
        break
      else:
        rem.insert(0, a[i])
    rem = (None if not rem else join(rem))

    # create the solution header (from the input parameters)
    header = sprintf("{a} / {b} = {c} (rem {r}) [{subs}]", r=fmt(rem), subs=fmt_subs(subs))

    # create a slots object
    slots = Slots()

    # allocate slots for the input data
    (a, b, c) = slots.allocate((a, b, c))
    subs = list(slots.allocate(x) for x in subs)
    assert len(c) == len(subs), "result/intermediate mismatch"

    # no leading zeros (or singleton zeros, except for remainder)
    for s in flatten([(a, b, c)] + subs):
      if s is None: continue
      slots.slot_setprops(s[0], (_NE, 0))

    # an empty intermediate implies zero in the result
    # (and non-empty intermediates implies non-zero in the result)
    for (s, r) in zip(subs, c):
      if not s:
        slots._unify(r, slots.slot_find(_EQ, 0))
      else:
        slots.slot_setprops(r, (_NE, 0))

    # unify slots in the intermediate sums
    (i, j, prev) = (0, len(a) + 1 - len(subs), None)
    for (k, v) in enumerate(subs):
      if v:
        slots.unify(([] if k == 0 else prev[2]) + a[i:j], v[0])
        i = j
        prev = v
      j += 1

    # if the sum comes out exactly there is no remainder
    if rem is None:
      # we can unify the two terms in the final subtraction sum
      slots.unify(subs[-1][0], subs[-1][1])

    # record the symbols used in the input strings
    input_symbols = slots.symbols_used()

    # assign symbols for the slots
    (a, b, c) = slots.label((a, b, c))
    subs = list(slots.label(s) for s in subs)
    if rem is not None: rem = subs[-1][-1]

    # output the slot information
    if 0:
      for (k, vs) in slots._s2p.items():
        u = slots._p2s[_UN][k]
        if u:
          printf("slot {k} -> slot {u}")
        else:
          printf("slot {k} = {vs}")

    # record the arguments required for a solution
    self.args = (a, b, c, subs, rem)
    self.input_symbols = dict((k, slots.symbol(slots.slot_find(_IS, k, create=0))) for k in input_symbols)

    # assemble a SubstitutedExpression object
    expr = list()

    # the main division sum
    if rem is None:
      expr.append(sprintf("{b} * {c} = {a}"))
    else:
      expr.append(sprintf("{b} * {c} + {rem} = {a}"))

    # the multiples
    for (s, r) in zip(subs, c):
      if s is None: continue
      (x, y, z) = s
      expr.append(sprintf("{b} * {r} = {y}"))

    # the subtraction sums
    for s in subs:
      if s is None: continue
      (x, y, z) = s
      if z is None: continue
      expr.append(sprintf("{x} - {y} = {z}"))

    # add in any additional expressions
    if 'extra' in kw: expr.extend(kw['extra'])

    # remove duplicate expressions
    expr = list(uniq(expr))

    # solver parameters
    input_syms = join(sorted(set(self.input_symbols.keys())))
    opt = dict()
    opt['symbols'] = slots.symbols_used()
    opt['distinct'] = kw.get('distinct', input_syms)
    opt['template'] = sprintf("{{{a}}} / {{{b}}} = {{{c}}} (rem {r}) [{subs}]", r=fmt(rem, brace=1), subs=fmt_subs(subs, brace=1))
    opt['solution'] = kw.get('solution', input_syms)
    opt['header'] = kw.get('header', header)

    # initial values
    s2d = kw.get('s2d', dict())
    for (v, ss) in slots.prop_items(_EQ):
      for s in ss:
        s2d[slots.symbol(s)] = v
    opt['s2d'] = s2d

    # invalid digits
    d2i = collections.defaultdict(set)
    if kw.get('d2i', None):
      for (k, v) in kw['d2i'].items():
        d2i[k].update(self.input_symbols[s] for s in v)
    for (v, ss) in slots.prop_items(_NE):
      if 'digits' in kw and v not in kw['digits']: continue
      for s in ss:
        d2i[v].add(slots.symbol(s))
    opt['d2i'] = d2i

    # verbatim options
    for v in ('digits', 'answer', 'code', 'verbose'):
      if v in kw:
        opt[v] = kw[v]

    # initialise the substituted expression
    SubstitutedExpression.__init__(self, expr, **opt)

  def substitute_all(self, d, ss):
    if ss is None: return None
    return tuple(int(self.substitute(d, s)) for s in ss)

  def solve(self, check=None, first=None, verbose=None):
    """
    generate solutions for the substituted long division problem.

    solutions are returned as a SubstitutedDivisionSolution() object

    check - a boolean function called to reject unwanted solutions
    first - if set to True only the first solution is returned
    verbose - an integer controlling the output of solutions and additional information
    """
    if verbose is None: verbose = self.verbose
    answer = self.answer
    # solution templates
    (ta, tb, tc, tsubs, tr) = self.args
    if tr is None: tr = '0'
    for s in reversed(tsubs):
      if s:
        if s[-1] is None: s[-1] = '0'
        break
    # find solutions (but disable solution output)
    for s in SubstitutedExpression.solve(self, verbose=(verbose & ~4)):
      if answer: (s, ans) = s
      # substitute the solution values
      (a, b, c, r) = self.substitute_all(s, (ta, tb, tc, tr))
      subs = tuple(self.substitute_all(s, x) for x in tsubs)
      # find the values of the input symbols
      d = dict((k, s[v]) for (k, v) in self.input_symbols.items())
      # copy any input symbols that were eliminated
      for (k, v) in self.input_symbols.items():
        if k not in s: s[k] = s[v]
      # made a solution object
      ss = SubstitutedDivisionSolution(a, b, c, r, subs, d, s)
      if check and not(check(ss)): continue
      # output the solution
      if verbose & 4: self.output_solution(ss)
      # return the result
      yield ((ss, ans) if answer else ss)
      if first: break

  def output_solution(self, s):
    # copy any input symbols that were eliminated
    SubstitutedExpression.output_solution(self, s.s)

  def solution_intermediates(self, s):
    # the intermediate subtraction sums are now part of the solution
    return s.subs

  # deal with additional SubstitutedDivision() options

  @classmethod
  def _usage(cls):
    return (
      "usage: SubstitutedDivision [<opts>] \"<a> / <b> = <c>\" \"<x> - <y> = <z>\" ...",
      "options:",
      "  --extra=<expr> (or -E<expr>) = extra alphametic expression (option may be repeated)",
    ) + SubstitutedExpression._usage()[2:]


  @classmethod
  def _getopt(cls, k, v, opt):
    if k == 'E' or k == 'extra':
      if not opt.get('extra', None): opt['extra'] = []
      opt['extra'].append(v)

    else:
      return SubstitutedExpression._getopt(k, v, opt)

    return True


  @classmethod
  def command_line(cls, args):
    """
    run the SubstitutedDivision solver with the specified command line
    arguments.

    the division sum is specified on the command line as:

      "<a> / <b> = <c>" "<x> - <y> = <z>" ...

    there should be as many intermediate subtraction sums as there are
    digits in the result <c>. when there is an empty intermediate sum
    (which corresponds to a 0 in the result) an empty argument should
    be passed. if there is no remainder the final intermediate
    subtraction will look like "<x> - <y> = 0".

    literal digits in the arguments stand for themselves, a ?
    character stands for any digit, and a letter stands for a digit
    whose value is not known.

    solver parameters can be specified on the command line in the same
    way as for the SubstitutedExpression solver, along with the
    additional "--extra / -E" parameter.

    Some exapmles:


    [Enigma 206] <https://enigmaticcode.wordpress.com/2014/07/13/enigma-206-division-some-letters-for-digits-some-digits-missing/>

    % python enigma.py SubstitutedDivision "pkmkh / ?? = ???" "pkm - pmd = xp" "xpk - ?? = kh" "khh - mbg = k"
    7????? / ?? = ????? (rem 0) [7? - ?? = ??, ??? - ?? = ?, None, ??? - ?? = ??, ??? - ??7 = 0]
    760287 / 33 = 23039 (rem 0) [76 - 66 = 10, 100 - 99 = 1, None, 128 - 99 = 29, 297 - 297 = 0]
    [1 solution]


    [Enigma 250] <https://enigmaticcode.wordpress.com/2015/01/13/enigma-250-a-couple-of-sevens/>

    The third intermediate subtraction sum is empty.

    % python enigma.py SubstitutedDivision "7????? / ?? = ?????" "7? - ?? = ??" "??? - ?? = ?" "" "??? - ?? = ??" "??? - ??7 = 0"
    7????? / ?? = ????? (rem 0) [7? - ?? = ??, ??? - ?? = ?, None, ??? - ?? = ??, ??? - ??7 = 0]
    760287 / 33 = 23039 (rem 0) [76 - 66 = 10, 100 - 99 = 1, None, 128 - 99 = 29, 297 - 297 = 0]
    [1 solution]


    [Enigma 226] <https://enigmaticcode.wordpress.com/2014/10/02/enigma-226-cold-comfort-five/>

    A solver parameter is used to stop X from taking on the value of 5.

    % python enigma.py SubstitutedDivision --invalid="5,X" "???0000 / ?? = ?????" "??? - ?? = ?" "?? - ?? = ??" "??? - ?X? = ??" "??? - ??? = ??" "??? - ??? = 0"
    ???0000 / ?? = ????? (rem 0) [??? - ?? = ?, ?? - ?? = ??, ??? - ?X? = ??, ??? - ??? = ??, ??? - ??? = 0]
    1050000 / 48 = 21875 (rem 0) [105 - 96 = 9, 90 - 48 = 42, 420 - 384 = 36, 360 - 336 = 24, 240 - 240 = 0] / X=8
    [1 solution]


    [Enigma 16] <https://enigmaticcode.wordpress.com/2012/12/12/enigma-16-four-five-six-seven/>

    The --extra parameter is used to provide an additional condition.

    % python enigma.py SubstitutedDivision --distinct="" "C??? / A? = ???" "C? - ?D = ?" "" "??? - B?? = 0" --extra="sum([A != 4, B != 5, C != 6, D != 7]) = 1"
    C??? / A? = ??? (rem 0) [C? - ?D = ?, None, ??? - B?? = 0]
    6213 / 57 = 109 (rem 0) [62 - 57 = 5, None, 513 - 513 = 0] / A=5 B=5 C=6 D=7
    [1 solution]
    """
    # this function is only here so we can set the docstring, so just call the parent class
    return super(SubstitutedDivision, cls).command_line(args)

###############################################################################

# Cross Figure Solver

class CrossFigure(object):
  """
  A solver for simple cross-figure puzzles.

  As an example this is a simplified solution of Enigma 1755:
  <http://enigmaticcode.wordpress.com/2013/06/26/enigma-1755-sudoprime-ii/#comment-2515>

  Consider the grid:

  A # # # B
  C D # E F
  # G H J #
  K L # M N
  P # # # Q

  The are 15 solution cells they are indexed 0 to 14, but we label them
  as above to make things easier:

  >>> (A, B, C, D, E, F, G, H, J, K, L, M, N, P, Q) = irange(0, 14)

  Create the puzzle, the numbers A and G are already filled out:

  >>> p = CrossFigure('7?????3????????')

  The 2-digit answers are primes (for readability I've reduced the list of primes):

  >>> ans2 = [(A, C), (K, P), (B, F), (N, Q), (C, D), (E, F), (K, L), (M, N)]
  >>> primes2 = [19, 31, 37, 41, 43, 47, 61, 67, 71, 73, 79]
  >>> p.set_answer(ans2, primes2)

  The 3-digit answers are also primes (again this is a reduced list):

  >>> ans3 = [(D, G, L), (G, H, J), (E, J, M)]
  >>> primes3 = [137, 139, 163, 167, 173, 307, 317, 367, 379, 397, 631, 673, 691]
  >>> p.set_answer(ans3, primes3)

  No digit is repeated in a row, column or diagonal:

  >>> rows = [(A, B), (C, D, E, F), (G, H, J), (K, L, M, N), (P, Q)]
  >>> cols = [(A, C, K, P), (D, G, L), (E, J, M), (B, F, N, Q)]
  >>> diags = [(A, D, H, M, Q), (B, E, H, L, P)]
  >>> p.set_distinct(rows + cols + diags)

  And the final check is that no even digit is repeated:

  >>> p.set_check(lambda grid: not any(grid.count(d) > 1 for d in '02468'))

  Now run the solver, which iterates over the solutions:

  >>> list(p.solve())
  [['7', '3', '3', '1', '6', '7', '3', '0', '7', '4', '7', '3', '1', '1', '9']]

  Note that the solution in the grid are returned as strings.
  """

  def __init__(self, grid):
    self.grid = list(grid)
    self.answers = list()
    self.groups = list()
    self.fn = (lambda grid: True)

  # set answers and their candidate solutions
  def set_answer(self, answers, candidates):
    candidates = list(str(x) for x in candidates)
    for a in answers: self.answers.append((a, candidates))

  # set groups of distinct digits
  def set_distinct(self, groups):
    self.groups.extend(groups)

  # set final check for a solution
  def set_check(self, fn):
    self.fn = fn

  # check the distinct groups
  def _check_distinct(self, grid):
    for d in self.groups:
      s = tuple(grid[i] for i in d if grid[i] != '?')
      if len(s) > 0 and len(set(s)) < len(s): return False
    return True

  # check the answers match their candidates
  def _check_answers(self, grid):
    # check all answers match the candidates
    for (ans, cs) in self.answers:
      t = join(grid[i] for i in ans)
      if t not in cs: return False
    # and run any final check
    return self.fn(grid)

  def match(self, t, ns):
    for n in ns:
      if all(a in ('?', b) for (a, b) in zip(t, n)):
        yield n

  # the solver
  def solve(self, grid=None, seen=None):
    if grid is None: grid = self.grid
    if seen is None: seen = set()
    # is the grid complete?
    if '?' not in grid:
      # skip duplicated solutions
      s = join(grid)
      if s not in seen:
        seen.add(s)
        if self._check_answers(grid):
          yield grid
    else:
      # find a partially filled out answer
      ts = list()
      for (ans, cs) in self.answers:
        t = tuple(grid[i] for i in ans)
        n = t.count('?')
        if n > 0: ts.append((n, t, ans, cs))
      # with the fewest missing letters
      (n, t, ans, cs) = min(ts)
      # and fill it out
      for n in self.match(t, cs):
        grid2 = update(grid, ans, n)
        if self._check_distinct(grid2):
          for x in self.solve(grid2, seen):
            yield x

  # return all answers in the grid (as strings)
  def get_answers(self, grid):
    for (ans, cs) in self.answers:
      yield join(grid[i] for i in ans)

###############################################################################

# Football League Table Utility

class Football(object):
  """
  Utility routines for solving Football League Table puzzles.

  For usage examples see:
  Enigma 7 <http://enigmaticcode.wordpress.com/2012/11/24/enigma-7-football-substitutes/#comment-2911>
  """

  # initialise the scoring system
  def __init__(self, games=None, points=None, swap=None):
    """
    initialise the Football object.

    each match is considered to be between team 0 vs. team 1.

    <games> is a sequence of possible match outcomes, this defaults to
    the following: 'w' win for team 0 (loss for team 1), 'd' draw, 'l'
    loss for team 0 (win for team 1), 'x' match not played.

    <points> is a dictionary giving the points awarded for a match
    outcome (from team 0's viewpoint). It defaults to 2 points for a
    win, 1 point for a draw.

    <swap> is a dictionary used to change from team 0 to team 1's
    perspective. By default it swaps 'w' to 'l' and vice versa.

    You might want to set <games> to 'wld' if all matches are played,
    or 'wl' if there are no draws. And you can use <points> to
    accommodate different scoring regimes.
    """
    # set the defaults
    if games is None:
      games = 'wldx'
    if points is None:
      points = dict(w=2, d=1)
    if swap is None:
      swap = { 'w': 'l', 'l': 'w' }

    self._games = tuple(games)
    self._points = points
    self._swap = swap
    self._table = collections.namedtuple('Table', ('played',) + self._games + ('points',))

  def swap(self, m):
    return self._swap.get(m, m)

  # generate games
  def games(self, *gs, **kw):
    """
    A generator for possible game outcomes.

    Usually used like this:

      for (ab, bc, bd) in football.games(repeat=3):
        print(ab, bc, bd)

    This will generate possible match outcomes for <ab>, <bc> and
    <bd>. Each outcome will be chosen from the <games> parameter
    specified in the creation of the Football object, so be default
    will be: 'w', 'd', 'l', 'x'.

    Or you can specify specific outcomes:

      for (ab, bc, bd) in football.games('wd', 'dl', 'wl'):
        print(ab, bc, bd)

    If no arguments are specified then a single outcome is assumed:

    for ab in football.games():
      print(ab)
    """
    if not gs: gs = [self._games]
    if 'repeat' in kw: gs = gs * kw['repeat']
    if len(gs) == 1:
      # [Python 3]: yield from ...
      for r in gs[0]: yield r
    else:
      # [Python 3]: yield from ...
      for r in itertools.product(*gs): yield r

  # compute the table
  def table(self, gs, ts):
    """
    Compute the table given a sequence of match outcomes and team assignments.

    <gs> is a sequence of game outcomes (from team 0's point of view)
    <ts> is a sequence identifying the team (0 for team 0, 1 for team 1)

    For example, to compute a table for team B:

      B = football.table([ab, bc, bd], [1, 0, 0])
      print(B.played, B.w, B.d, B.l, B.points)

    The returned table object has attributes named by the possible
    match outcomes (by default, .w, .d, .l, .x) and also .played (the
    total number of games played) and .points (calculated points).
    """
    r = dict((x, 0) for x in self._games)
    played = points = 0
    for (g, t) in zip(gs, ts):
      if t: g = self._swap.get(g, g)
      r[g] += 1
      if g != 'x':
        played += 1
        points += self._points.get(g, 0)
    return self._table(*((played,) + tuple(r[x] for x in self._games) + (points,)))

  # generate possible score lines
  def scores(self, gs, ts, f, a, pss=None, pts=None, s=[]):
    """
    Generate possible score lines for a sequence of match outcomes <gs>,
    team assignments <ts>, and total goals for <f> and against <a>.

    A sequence of scores for matches already played <pss> and
    corresponding team assignments <pts> can be specified, in which case
    the goals scored in these matches will be subtracted from <f> and
    <a> before the score lines are calculated.

    A sequence of scores matching the input templates will be
    produced. Each score is a tuple (<team 0>, <team 1>) for a played
    match, or None for an unplayed match.

    For example if team B has 9 goals for and 6 goal against:

      for (AB, BC, BD) in football.scores([ab, bc, bd], [1, 0, 0], 9, 6):
        print(AB, BC, BD)
    """
    if pss:
      (pf, pa) = self.goals(pss, pts)
      (f, a) = (f - pf, a - pa)
    return self._scores(gs, ts, f, a, [])

  def _scores(self, gs, ts, f, a, s):
    # are we done?
    if not gs:
      if f == a == 0:
        yield s
    else:
      # check the first game
      # [Python 3]: (x, *xs) = xs
      (g, gs) = (gs[0], gs[1:])
      (t, ts) = (ts[0], ts[1:])
      if t: g = self._swap.get(g, g)
      # is it unplayed?
      if g == 'x':
        # [Python 3]: yield from ...
        for r in self._scores(gs, ts, f, a, s + [None]): yield r
      # is it a draw?
      elif g == 'd':
        for i in irange(0, min(f, a)):
          # [Python 3]: yield from ...
          for r in self._scores(gs, ts, f - i, a - i, s + [(i, i)]): yield r
      # is it a win?
      elif g == 'w':
        for j in irange(0, a):
          for i in irange(j + 1, f):
            s0 = ((j, i) if t else (i, j))
            # [Python 3]: yield from ...
            for r in self._scores(gs, ts, f - i, a - j, s + [s0]): yield r
      # is it a loss?
      elif g == 'l':
        for i in irange(0, f):
          for j in irange(i + 1, a):
            s0 = ((j, i) if t else (i, j))
            # [Python 3]: yield from
            for r in self._scores(gs, ts, f - i, a - j, s + [s0]): yield r

  # compute goals for, against
  def goals(self, ss, ts):
    """
    Return goals for and against given a sequence of scores <ss>, team
    assignments <ts> and the total goals for <f> and against <a>.
    """
    (f, a) = (0, 0)
    for (s, t) in zip(ss, ts):
      if s is None: continue
      f += s[t]
      a += s[t ^ 1]
    return (f, a)

  # compute outcomes based on scores
  def outcomes(self, ss, ts=None):
    """
    return a sequence of outcomes ('x', 'w', 'd', 'l') for a sequence of scores.
    """
    if ts is None: ts = [0] * len(ss)
    return tuple(('x' if s is None else 'ldw'[compare(s[0 ^ t], s[1 ^ t]) + 1]) for (s, t) in zip(ss, ts))

  # extract values from a dictionary relevant to team t
  # return (<vs>, <ts>)
  def extract(self, d, t):
    """
    Extract values from dictionary <d> that are relevant to team <t>.

    A pair (<vs>, <ts>) is returned.
    <vs> is the list of relevant values.
    <ts> is the index of the team in the corresponding key

    Given a dictionary of matches outcomes <matches> the table row for
    team A can be constructed with:

      (gs, ts) = football.extract(matches, A)
      tableA = football.table(gs, ts)

    Similarly, with a dictionary of scores <scores>, goals for /
    against can be calculated this:

      (ss, ts) = football.extract(scores, A)
      (forA, againstA) = football.goals(ss, ts)
    """
    (vs, ts) = (list(), list())
    for (k, v) in d.items():
      i = find(k, t)
      if i != -1:
        vs.append(v)
        ts.append(i)
    return (vs, ts)

  # shortcuts to extract table row and goals for/against
  extract_table = lambda self, ms, t: self.table(*self.extract(ms, t))
  extract_goals = lambda self, ss, t: self.goals(*self.extract(ss, t))


  # solver for a table with substituted values

  def _substituted_table(self, table, n, teams, matches, d, vs):
    # are we done?
    if not teams:
      yield (matches, d)
      return

    # check a row of the table (used by the Football.substituted_table() solver)
    # a (possibly) updated value for d is returned
    # t - team to check (int index)
    # d - assignments of letters to numbers (dict)
    # r - table row to check (as a dict)
    # table - columns in the substituted table (dict)
    # vs - allowable values
    def check_row(t, d, r, table, vs):
      cow = True # copy on write flag for d
      for (k, v) in r.items():
        # extract the corresponding letter
        x = table.get(k, None)
        if x is None: continue
        x = x[t]
        # and match it
        if x == '?': continue
        if x in d:
          # is this letter assigned to a different number?
          if d[x] != v: return None
        else:
          # is this a valid value?
          if v not in vs: return None
          # is this number already assigned to a different letter?
          if v in d.values(): return None
          # assign the new value
          if cow:
            d = dict(d)
            cow = False
          d[x] = v
      # return the (possibly updated) mapping
      return d


    # deal with team t
    t = teams[0]

    # matches for team t
    ms = list((x, t) for x in range(0, t)) + list((t, x) for x in range(t + 1, n))
    # and the matches remaining
    rs = diff(ms, matches)

    if not rs:
      # there are no remaining matches to choose
      # compute the row in the table for team t
      r = self.table((matches[m] for m in ms), (m.index(t) for m in ms))
      # check the output of the table
      d1 = check_row(t, d, r._asdict(), table, vs)
      if d1:
        # there were no mismatches, solve for the remaining teams
        for z in self._substituted_table(table, n, teams[1:], matches, d1, vs): yield z
      return

    # choose outcomes for each remaining match
    for s in self.games(repeat=len(rs)):
      matches1 = update(matches, rs, s)
      # compute the row in the table for team t
      r = self.table((matches1[m] for m in ms), (m.index(t) for m in ms))
      # check the output of the table
      d1 = check_row(t, d, r._asdict(), table, vs)
      if d1:
        # there were no mismatches, solve for the remaining teams
        for z in self._substituted_table(table, n, teams[1:], matches1, d1, vs): yield z

  # solve a substituted table problem
  def substituted_table(self, table, teams=None, matches=None, d=None, vs=None):
    """
    solve a substituted table football problem where numbers in the
    table have been substituted for letters.

    generates pairs (<matches>, <d>) where <matches> is a dict() of
    match outcomes indexed by team indices, so the value at (<i>, <j>)
    is the outcome for the match between the teams with indices <i>
    and <j> in the table. <d> is dict() mapping letters used in the
    table to their corresponding integer values.

    table - a dict() mapping the column names to the substituted
    values in the columns of the table. '?' represents an empty cell
    in the table. columns need not be specified if they have no
    non-empty values.

    teams - a sequence of indices specifying the order the teams will
    be processed in. if no order is specified a heuristic order will
    be chosen.

    matches - a dictionary of known match outcomes. usually this is
    the empty dictionary.

    d - a dictionary mapping known letters to numbers. usually this is
    empty.

    vs - allowable values in the table. if not specified single digits
    will be used.
    """
    n = max(len(x) for x in table.values())
    if teams is None:
      # choose an order to process the teams in
      rows = tuple(zip(*(table.values())))
      teams = sorted(range(0, n), key=lambda i: (rows[i].count('?'), len(set(rows[i]))))
    if matches is None: matches = dict()
    if d is None: d = dict()
    if vs is None: vs = list(irange(0, 9))
    for z in self._substituted_table(table, n, teams, matches, d, vs): yield z


  def _substituted_table_goals(self, gf, ga, matches, d, teams, scores):
    # are we done?
    if not teams:
      yield scores
      return

    # deal with team t
    t = teams[0]

    # matches for team t
    ms = list(m for m in matches.keys() if t in m)
    # matches remaining to be scored
    rs = diff(ms, scores)
    if not rs:
      # check the values
      (f, a) = self.goals(list(scores[m] for m in ms), list(m.index(t) for m in ms))
      if f == d[gf[t]] and a == d[ga[t]]:
        for z in self._substituted_table_goals(gf, ga, matches, d, teams[1:], scores): yield z
      return

    # matches we already have scores for
    sms = list(m for m in ms if m in scores)
    # find possible scores for each remaining match
    for s in self.scores(list(matches[m] for m in rs), list(m.index(t) for m in rs), d[gf[t]], d[ga[t]], list(scores[m] for m in sms), list(m.index(t) for m in sms)):
      scores2 = update(scores, rs, s)
      for z in self._substituted_table_goals(gf, ga, matches, d, teams[1:], scores2): yield z

  # gf, ga - goals for, goals against
  # matches - match outcomes
  # teams - order teams are processed in
  # scores - score lines
  def substituted_table_goals(self, gf, ga, matches, d=None, teams=None, scores=None):
    """
    determine the scores in matches from a substituted table football problem.

    generates dicts <scores>, which give possible score lines for the
    matches in <matches> (if a match is specified as 'x' (unplayed) a
    score of None is returned).

    gf, ga - goals for, goals against columns in the table. specified
    as symbols that index into the dict() <d> to give the actual
    values.

    matches - the match outcomes. usually this will be the result of a
    call to substituted_table().

    teams - the order the teams are processed in.

    scores - known scores. usually this is empty.
    """
    if d is None: d = dict((str(i), i) for i in irange(0, 9))
    if teams is None: teams = list(range(0, len(gf)))
    if scores is None: scores = dict()
    # fill out unplayed matches
    for (k, v) in matches.items():
      if v == 'x': scores[k] = None
    for z in self._substituted_table_goals(gf, ga, matches, d, teams, scores): yield z


  def output_matches(self, matches, scores=None, teams=None, d=None, start=None, end=''):
    """
    output a collection of matches.

    matches - dict() of match outcomes. usually the result of a call
    to substituted_table().

    scores - dict() of scores in the matches. usually the result of a
    call to substituted_table_goals().

    teams - labels to use for the teams (rather than the row indices).

    d - dict() of symbol to value assignments to output.

    start, end - delimiters to use before and after the matches are
    output.
    """
    if start is not None:
      printf("{start}")
    for k in sorted(matches.keys()):
      m = matches[k]
      if scores:
        if scores.get(k, None):
          s = join(scores[k], sep='-')
        else:
          if m == 'x':
            s = '---'
          else:
            s = '?-?'
      else:
        s = ''
      if teams:
        k = tuple(teams[t] for t in k)
      printf("{k} = ({m}) {s}", k=join(k, sep=' vs '))
    if d is not None:
      printf("{d}", d=join((join((k, d[k]), sep='=') for k in sorted(d.keys())), sep=' '))
    if end is not None:
      printf("{end}")

###############################################################################

# Domino Grid solver (see Enigma 179, Enigma 303, Enigma 342, ...)

class DominoGrid(object):

  def __init__(self, N, M, grid):
    # checks
    n = len(grid)
    assert n == N * M
    (D, r) = divmod(n - grid.count(None), 2)
    assert r == 0
    self.grid = grid
    self.N = N # columns
    self.M = M # rows
    self.D = D # number of dominoes

  # solve the grid
  # fixed can contiain pairs of indices of fixed dominoes  
  def solve(self, fixed=None):
    (N, M, D, grid) = (self.N, self.M, self.D, self.grid)

    # g = grid
    # n = label of next domino (1 to D)
    # ds = dominoes already placed
    def _solve(g, n, ds):
      # are we done?
      if n > D:
        # output the pairings
        yield g
      else:
        # find the next unassigned square
        for (i, d) in enumerate(g):
          if d < 0: continue
          (y, x) = divmod(i, N)
          # find placements for the domino
          js = list()
          # horizontally
          if x < N - 1 and not(g[i + 1] < 0): js.append(i + 1)
          # vertically
          if y < M - 1 and not(g[i + N] < 0): js.append(i + N)
          # try possible placements
          for j in js:
            d = ordered(g[i], g[j])
            if d not in ds:
              for s in _solve(update(g, [i, j], [-n, -n]), n + 1, ds.union([d])): yield s
          break

    # fixed can contain initial placements of dominoes
    (n, ds) = (1, set())
    if fixed:
      for (i, j) in fixed:
        assert abs(i - j) in (1, N)
        d = ordered(grid[i], grid[j])
        assert d not in ds
        grid = update(grid, [i, j], [-n, -n])
        ds.add(d)
        n += 1

    # solve for the remaining dominoes
    return _solve(grid, n, ds)

  # output a solution grid
  def output_solution(self, s, prefix=''):
    (N, M, grid) = (self.N, self.M, self.grid)

    s1 = s2 = ''
    for (i, a) in enumerate(grid):
      (r, c) = divmod(i, N)
      if r + 1 < M:
        s2 += ('| ' if a is not None and s[i] == s[i + N] else '  ')
      s1 += ('.' if a is None else str(a))
      if c + 1 < N:
        s1 += ('-' if a is not None and s[i] == s[i + 1] else ' ')
      else:
        print(prefix + s1)
        print(prefix + s2)
        s1 = s2 = ''

  # solve a grid and output solutions
  def go(self, fixed=None, sep='', prefix=''):
    for s in self.solve(fixed):
      self.output_solution(s, prefix=prefix)
      print(sep)

###############################################################################

# Timing

import atexit
import time

if hasattr(time, 'process_time'):
  _timer = time.process_time
if hasattr(time, 'perf_counter'):
  _timer = time.perf_counter
elif sys.platform == "win32":
  _timer = time.clock
else:
  _timer = time.time

class Timer(object):

  """
  This module provides elapsed timing measures.

  There is a default timing object called "timer" created. So you can
  determine the elapsed runtime of code fragments using:

    from enigma import timer

    timer.start()

    some_code()

    timer.stop()

  and when the program exits it will report the elapsed time thus:

    [timing] elapsed time: 0.0008729s (872.85us)

  By default the elapsed time reported when the Python interpreter
  exits. But if you want more control you can do the following:

    from enigma import Timer

    ...

    # create the timer
    timer = Timer('name')

    # start the timer
    timer.start()

    # the code you want to time would go here
    some_code()

    # stop the timer
    timer.stop()

    # print the report
    timer.report()


  If you don't call start() then the timer will be started when it is
  created.

  If you don't call report() then the elapsed time report will be
  printed when the Python interpreter exits.

  If you don't call stop() then the timer will be stopped when the
  Python interpreter exits, just before the report is printed.


  You can create multiple timers. It might help to give them different
  names to distinguish their reports. (The default name is "timing").


  You can also wrap code to be timed like this:

    with Timer('name'):
      # the code you want to time goes here
      some_code()


  You can create a function that will report the timing each time it
  is called by decorating it with the timed decorator:

    from enigma import timed

    @timed
    def whatever():
      some_code()

"""


  def __init__(self, name='timing', timer=_timer, file=sys.stderr, exit_report=1, auto_start=1):
    """
    Create (and start) a timer.

    name = the name used in the report
    timer = function used to measure time (should return a number of seconds)
    file = where the report is sent
    exit_report = should the report be generated at exit
    auto_start = should the timer be automatically started
    """
    self._t0 = None
    self._t1 = None
    self._name = name
    # timer can be the name of a function from time, e.g. 'time' or 'clock'
    if type(timer) is str: timer = getattr(time, timer)
    self._timer = timer
    self._file = file
    self._exit_report = exit_report
    self._report = None
    if auto_start: self.start()

  def start(self):
    """
    Set the start time of a timer.
    """
    if self._exit_report:
      atexit.register(self.report, force=0)
      self._exit_report = False
    self._t1 = None
    self._t0 = self._timer()

  def stop(self):
    """
    Set the stop time of a timer.
    """
    self._t1 = self._timer()

  def elapsed(self, disable_report=1):
    """
    Return the elapsed time of a stopped timer

    disable_report = should the report be disabled
    """
    if disable_report: self._report = '<none>'
    return (self._t1 or self._timer()) - self._t0

  def format(self, t, fmt='{:.2f}'):
    """
    Format a time for the report.
    """
    u = 's'
    if t < 1.0: (t, u) = (1000 * t, 'ms')
    if t < 1.0: (t, u) = (1000 * t, 'us')
    return (fmt + u).format(t)

  def report(self, force=1):
    """
    Stop the timer and generate the report (if required).

    The report will only be generated once (if it's not been disabled).
    """
    if self._report and not(force): return self._report
    if self._t1 is None: self.stop()
    e = self.elapsed()
    self._report = sprintf("[{n}] elapsed time: {e:.7f}s ({f})", n=self._name, f=self.format(e))
    print(self._report, file=self._file)

  def printf(self, fmt='', **kw):
    e = self.elapsed()
    s = _sprintf(fmt, kw, sys._getframe(1))
    printf("[{n} {e}] {s}", n=self._name, e=self.format(e))

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.report()

# function wrapper
def timed(f):
  """
  Return a timed version of function <f>.
  """
  @functools.wraps(f)
  def _timed(*args, **kw):
    n = f.__name__
    t = Timer(n)
    r = f(*args, **kw)
    #printf("[{n} {args} {kw} => {r}]")
    t.report()
    return r
  return _timed

# create a default timer
timer = Timer(auto_start=0)

###############################################################################

# namespace

# an even simpler form of the 'Record' (or 'types.SimpleNamespace') class
# to make sub-namespaces within the module
#
# (I don't think this is very Pythonic, but it works)

class namespace(object):

  def __init__(self, name, vs):
    self.__name = name
    self.__dict__.update(**vs)

  def __repr__(self):
    return '<namespace ' + repr(self.__name) + '>'

###############################################################################

# NOTE: template_system is in testing, interface may change

# template system problems

def __template_system():

  # find values that match a system of template sequences
  def solve(templates, values=None):
    """
    Solve a template system. [See Enigma 307, Enigma 357]

    The values returned are sequence of strings, that match the given
    template sequence.

    The template sequence is a sequence of templates, and each item in
    the sequence is itself a sequence of strings. Every string in the
    returned values can be matched to one of the elements of each the
    templates, with no values needing to share a template string. But
    note that the matching of values to templates does not have occur
    in the order that the individual template strings are specified.

    When values are matched templates the '?' character represents a
    wildcard that can be matched to any character.

    So the pair of values:
      ('AB', 'AC')

    will match the following templates (the ordering of the individual
    templates doesn't matter):
      ('A?', '?C') matches as ('AB', 'AC')
      ('A?', '?B') matches as ('AC', 'AB')

    So we can solve this pair of templates as follows:
    >>> list(sorted(x) for x in template_system.solve([('A?', '?C'), ('A?', '?B')]))
    [['AB', 'AC']]

    Note that if values are not fully determined by the template system
    then wildcard characters may be present in the returned values.
    """

    # first some useful functions:

    # combine strings <s1> and <s2> (of equal length) using wildcard <w>
    def combine(s1, s2, w='?'):
      s = ''
      # consider each pair of characters
      # ? ? -> ?
      # x ? -> x
      # ? x -> x
      # x x -> x
      # x y -> FAIL
      for (a, b) in zip(s1, s2):
        if a == b or b == w:
          s += a
        elif a == w:
          s += b
        else:
          return None
      return s


    # match a sequence of values <vs> to the sequence of templates <ts> (in some order)
    def match(vs, ts, r=[]):
      # are we done?
      if not ts:
        yield (r if not vs else r + list(vs))
      else:
        # match the first template to a value
        t = ts[0]
        for (i, vi) in enumerate(vs):
          v = combine(vi, t)
          if v is None: continue
          # and try to match the remaining values to the remaining templates
          for x in match(vs[:i] + vs[i + 1:], ts[1:], r + [v]): yield x


    # generate values matching <vs> that also match all template sequences <ts>
    def generate(vs, ts):
      # are we done?
      if not ts:
        yield vs
      else:
        # find values that match the first template
        for s in match(vs, ts[0]):
          # and then try to match the remaining templates
          for x in generate(s, ts[1:]): yield x


    # if no values are provided use the longest template
    if values is None:
      (i, values) = max(enumerate(templates), key=lambda v: len(v[1]))
      templates = templates[:i] + templates[i + 1:]

    # now we can solve the system of templates
    # (in Python 3 we can use "yield from"
    for z in generate(values, templates): yield z

  # return the namespace
  return locals()


template_system = namespace('template_system', __template_system())

###############################################################################

# NOTE: grouping is in testing, interface may change

# grouping problems

def __grouping():

  # group the lists of elements in <vs> into groups (one element from each list)
  # such that the values in the groups satisfy the selection function <fn>
  def groups(vs, fn=None, s=[]):
    # are we done?
    if not vs[0]:
      yield tuple(s)
    else:
      # otherwise choose the next group to go with category 0
      for v in itertools.product(*(enumerate(x) for x in vs[1:])):
        # find indices and elements of the other categories
        (js, t) = zip(*v)
        # the full group is
        group = (vs[0][0],) + t
        # check the group
        if fn(*group):
          # solve for the remaining elements (can use "yield from" in Python 3)
          for z in groups([vs[0][1:]] + [x[:j] + x[j + 1:] for (x, j) in zip(vs[1:], js)], fn, s + [group]): yield z

  # output a grouping
  def output_groups(gs, sep=", ", end=""):
    for g in gs:
      print(sep.join(g))
    print(end)

  # output all groupings
  def solve(vs, fn, sep=", ", end=""):
    for gs in groups(vs, fn):
      output_groups(gs, sep, end)


  # a k-gang has a leader x, and k followers chosen from a sequence ys
  # pairwise they satisfy the selection function <fn>
  # return a set of followers for leader x
  def gang(k, x, ys, fn):
    # select possible followers
    for vs in subsets((y for y in ys if fn(x, y)), size=k):
      if fn(*vs):
        yield vs

  # find multiple k-gangs for leaders in xs, followers in ys
  def gangs(k, xs, ys, fn, gs=[]):
    # are we done?
    if not xs:
      yield gs
    else:
      # find a gang for the first leader
      for g in gang(k, xs[0], ys, fn):
        # and solve for the rest
        for s in gangs(k, xs[1:], diff(ys, g), fn, gs + [g]): yield s

  def output_gangs(xs, ys, sep=", ", end=""):
    for (x, y) in zip(xs, ys):
      print(x + ": " + sep.join(y))
    print(end)


  # useful selection functions

  # return the set of letters in a string
  def letters(s):
    return set(x for x in s.lower() if x.isalpha())

  # return a check function that checks each pair of values shares exactly <k> letters
  def share_letters(k):
    fn = ((lambda x: x == k) if type(k) is int else k)
    # check each pair of values shares exactly <k> different letters
    def check(*vs):
      return all(fn(len(letters(a).intersection(letters(b)))) for (a, b) in itertools.combinations(vs, 2))
    return check

  # return the namespace
  return locals()


grouping = namespace('grouping', __grouping())

###############################################################################

# matrix routines (see Enigma 287)

def __matrix():

  # given two matrices A and B, returns (det(A), X) st A * X = B
  # A must be square, and the elements must support __truediv__
  def _gauss(A, B):

    n = len(A)
    p = len(B[0])
    det = 1

    for i in range(0, n - 1):

      k = i
      for j in range(i + 1, n):
        if abs(A[j][i]) > abs(A[k][i]):
          k = j

      if k != i:
        (A[i], A[k]) = (A[k], A[i])
        (B[i], B[k]) = (B[k], B[i])
        det = -det

      for j in range(i + 1, n):
        t = A[j][i] / A[i][i] # note use of /
        for k in range(i + 1, n):
          A[j][k] -= t * A[i][k]
        for k in range(p):
          B[j][k] -= t * B[i][k]

    for i in range(n - 1, -1, -1):
      for j in range(i + 1, n):
        t = A[i][j]
        for k in range(p):
          B[i][k] -= t * B[j][k]

      t = 1 / A[i][i] # note use of /
      det *= A[i][i]
      for j in range(p):
        B[i][j] *= t

    return (det, B)


  import fractions

  # map <fn> over all elements of (2d) matrix <M>
  def map2d(fn, M):
    if fn is None: fn = identity
    return list(list(fn(x) for x in r) for r in M)

  # default B is the identity matrix corresponding to A in this case X is the inverse of A
  def gauss(A, B=None, F=fractions.Fraction):

    # check A is square
    n = len(A)
    assert all(len(r) == n for r in A)

    # if B is None, use the identity matrix
    if B is None:
      B = list([0] * n for _ in range(0, n))
      for i in range(0, n):
        B[i][i] = 1

    # convert A and B (so that the elements supports __truediv__)
    A = map2d(F, A)
    B = map2d(F, B)

    # solve it
    try:
      return _gauss(A, B)
    except ZeroDivisionError:
      return (0, None)


  # solve a system of linear equations
  def linear(A, B=0, F=fractions.Fraction):
    """
    solve a system of linear equations.

      A x = B

    A is the matrix of coefficients of the variables (n equations in m variables)
    B is the the vector of constants (a sequence of a single value that will be replicated)
    F is the field to operate over (which must support __truediv__)

    If the system is underspecified an "incomplete" error is raised.
    If the system is inconsistent an "inconsistent" error is raised.

    Otherwise a sequence of the solution values x is returned (which will be in the field F)
    """

    # verify A
    n = len(A)
    m = len(A[0])
    assert all(len(x) == m for x in A)

    # make B into a viable matrix
    # if B is a scalar value, replicate it n times
    if not isinstance(B, Sequence): B = [B] * n

    A = map2d(F, A)
    B = list(map(F, B))

    # for each column i
    for i in itertools.count(0):
      if not(i < m): break
      if n < m: raise ValueError("incomplete")

      # choose the row with the largest value in the column i
      j = max(range(i, n), key=(lambda j: abs(A[j][i])))

      # if necessary bring it to row i
      if j != i: (A[i], A[j], B[i], B[j]) = (A[j], A[i], B[j], B[i])

      # scale equation i so the co-efficient in column i is 1
      v = A[i][i]
      if v == 0: raise ValueError("incomplete")
      if v != 1:
        for k in range(i, m):
          A[i][k] /= v
        B[i] /= v

      # eliminate co-efficients in row i
      rs = list()
      for j in range(0, n):
        if j != i:
          t = A[j][i]
          if t != 0:
            for k in range(i, m):
              A[j][k] -= t * A[i][k]
            B[j] -= t * B[i]
            # if all coefficients are 0
            if all(A[j][k] == 0 for k in range(0, m)):
              if B[j] == 0:
                rs.insert(0, j)
              else:
                # the system is inconsistent
                raise ValueError("inconsistent")
      # delete any dependent rows
      if rs:
        for j in rs:
          del A[j]
          del B[j]
        n -= len(rs)

    assert len(B) == m
    return B


  # return the namespace
  return locals()

matrix = namespace('matrix', __matrix())

###############################################################################

# some handy development routines


# this looks for a "STOP" file, and if present removes it and returns True
def stop(files=None, files_extra=None, use_exit=0, verbose=1):
  if files is None:
    # list of files, default is: STOP.<pid>, STOP.<prog>, STOP
    files = [ ("STOP", os.getpid()) ]
    f = sys.argv[0]
    if f:
      files.append(("STOP", os.path.splitext(os.path.basename(f))[0]))
    files.append("STOP")
  if files_extra is not None:
    files = files_extra + files
  for f in files:
    if not isinstance(f, basestring):
      f = join(f, sep='.')
    if os.path.isfile(f):
      # found one
      if verbose: printf("found stop file \"{f}\"")
      os.unlink(f)
      if use_exit: sys.exit(0)
      return True
  # not found
  return False


# this allows you to get an interactive shell on a running Python process
# by sending it SIGUSR1 (30), but is only enabled if "i" appears in
# the environment variable $PY_ENIGMA
#
# similarly it also enables you to set a function using status() that will
# be called if SIGUSR2 (31) is sent
#
# if "v" appears in $PY_ENIGMA a message will be printed giving the PID
# of the process (so you can do: "kill -SIGUSR1 <PID>")

@static(fn=None)
def status(fn, atexit=0):
  status.fn = fn
  if atexit: atexit.register(fn)

_PY_ENIGMA = os.getenv("PY_ENIGMA") or ''

if 'i' in _PY_ENIGMA:

  if 'v' in _PY_ENIGMA: printf("[PY_ENIGMA: pid={pid}]", pid=os.getpid())

  # start an interactive Python shell using the environment of the specified frame
  def shell(frame=None, env=None):
    vs = dict()
    if frame:
      vs = update(frame.f_globals, frame.f_locals)
      printf("[file {frame.f_code.co_filename}, line {frame.f_lineno}, function {frame.f_code.co_name}]")
      vs['_frame'] = frame
    if env:
      vs = update(vs, env)

    import code
    import readline
    import rlcompleter
    readline.parse_and_bind('tab: complete')
    code.interact(local=vs)


  import signal

  # SIGUSR1 -> start an interactive shell

  def _signal_handler_shell(signum, frame):
    printf("[interrupt ... (Ctrl-D to resume) ...]")
    shell(frame=frame)
    printf("[continuing ...]")

  if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, _signal_handler_shell)
  else:
    print("[PY_ENIGMA: failed to install handler for SIGUSR1]")

  # SIGUSR2 -> report status

  def _signal_handler_status(signum, frame):
    if status.fn: status.fn()

  if hasattr(signal, 'SIGUSR2'):
    signal.signal(signal.SIGUSR2, _signal_handler_status)
  else:
    print("[PY_ENIGMA: failed to install handler for SIGUSR2]")

# shortcuts to send signals
sigusr1 = lambda pid: os.kill(pid, signal.SIGUSR1)
sigusr2 = lambda pid: os.kill(pid, signal.SIGUSR2)

###############################################################################

# parse a run file (which uses a shell-like syntax)

def parsefile(path, *args):
  import shlex

  with open(path, 'r') as f:
    # parse the file removing whitespace, comments, quotes
    lexer = shlex.shlex(f, posix=1)
    lexer.whitespace_split = True
    words = list(lexer)

  cmd = words.pop(0)

  def divide(s, fn=lambda s: s.startswith('-')):
    for (i, x) in enumerate(s):
      if not fn(x):
        return (s[:i], s[i:])
    return (s, ())

  # insert any extra args
  if args:
    ((s1, s3), (s2, s4)) = (divide(words), divide(args))
    words = flatten((s1, s2, s3, s4))

  return (cmd, words)


_run_exit = None

# run command line arguments
# always returns None, but sets _run_exit
def run(cmd, *args, **kw):
  """
  run with command line arguments

  <cmd> can be a class in enigma.py that accepts a command line,
  or it can be a run file, Python program or other scripts

  <args> are the command line arguments to be provided

  additional options are:

    timed - if set, time the execution of <cmd>

    flags - 'p' = enable prompts, 'v' = enable verbose
  """

  global _run_exit, _PY_ENIGMA
  _run_exit = None

  timed = kw.get('timed')
  flags = kw.get('flags', '')
  #interact = kw.get('interact')

  # enabling 'prompt' disables timing
  if 'p' in _PY_ENIGMA or 'p' in flags: timed = 0
  saved = None

  # an alternative way to run a solver is to use "-r / --run <file> <additional-args>"
  if cmd == '-r' or cmd == '--run':
    (cmd, args) = (args[0], args[1:])
  elif cmd.startswith('-'):
    return

  # if cmd names a file
  if os.path.isfile(cmd):
    if timed and not isinstance(timed, basestring): timed = os.path.basename(cmd)
    if cmd.endswith(".run"):
      # *.run => treat it as a run file
      (cmd, args) = parsefile(cmd, *args)
    else:
      if cmd.endswith(".py"):
        # use runpy for *.py
        import runpy
        get_argv(force=1, args=args)
        sys.argv = [cmd] + list(args)
        if flags:
          saved = [_PY_ENIGMA]
          _PY_ENIGMA = join(sorted(uniq(_PY_ENIGMA + flags)))
        try:
          if timed: timed = Timer(name=timed)
          r = runpy.run_path(cmd)
          if timed: timed.report()
        finally:
          if saved:
            [_PY_ENIGMA] = saved
      else:
        # attempt to use a shebang line (see: run.py)
        path = os.path.abspath(cmd)
        with open(path, 'r') as fh:
          import shlex
          import subprocess
          s = next(fh)
          # find the shebang
          shebang = "#!"
          i = s.find(shebang)
          assert i != -1, "interpreter not found"
          cmd = s[i + len(shebang):].strip()
          cmd = shlex.split(cmd)
          cmd.append(path)
          cmd.extend(args)
        if flags:
          saved = [_PY_ENIGMA]
          _PY_ENIGMA = join(sorted(uniq(_PY_ENIGMA + flags)))
        try:
          if timed: timed = Timer(name=timed)
          subprocess.call(cmd)
          if timed: timed.report()
          r = 1
        finally:
          if saved:
            [PY_ENIGMA] = saved
        _run_exit = (0 if r else -1)
      return

  # if cmd names a class[.method]
  alias = { 'Alphametic': 'SubstitutedExpression' }
  (obj, _, fn_name) = cmd.partition('.')
  if not fn_name: fn_name = 'command_line'
  fn = globals().get(alias.get(obj, obj))
  if fn:
    fn = getattr(fn, fn_name, None)
    if fn:
      if timed and not isinstance(timed, basestring): timed = 'timing'
      if flags:
        saved = [_PY_ENIGMA]
        _PY_ENIGMA = join(sorted(uniq(_PY_ENIGMA + flags)))
      try:
        if timed: timed = Timer(name=timed)
        _run_exit = (fn(list(args)) or 0)
        if timed: timed.report()
      finally:
        if saved:
          [_PY_ENIGMA] = saved
      return
    else:
      printf("enigma.py: {obj}.{fn_name}() not implemented")

  # if we get this far we can't find the solver
  printf("enigma.py: unable to run \"{cmd}\"")
  _run_exit = -1
  return

def timed_run(*args):
  run(*args, timed=1)

###############################################################################

# implementation of command line options

# help (-h)
def _enigma_help():
  print('command line arguments:')
  print('  <class> <args> = run command_line(<args>) method on class')
  print('  [-r | --run] <file> [<additional-args>] = run the solver and args specified in <file>')
  print('  -t[v] = run tests [v = verbose]')
  print('  -u[cdr] = check for updates [c = only check, d = always download, r = rename after download]')
  print('  -h = this help')


# run doctests (-t)
def _enigma_test(verbose=0):
  import doctest
  return doctest.testmod(enigma, verbose=verbose)


# check for updates to enigma.py (-u)
# check = only check the current version
# download = always download the latest version
def __enigma_update(url, check=1, download=0, rename=0):

  if _python == 2:
    # Python 2.x
    from urllib2 import urlopen
  else:
    # Python 3.x
    from urllib.request import urlopen

  u = urlopen(url + 'py-enigma-version.txt')
  v = u.readline(16).decode().strip()
  printf("latest version is {v}")

  if (__version__ < v and not check) or download:
    name = v + '-enigma.py'
    printf("downloading latest version to \"{name}\"")
    with open(name, 'wb') as f:
      u = urlopen(url + 'enigma.py')
      while True:
        print('.', end='')
        data = u.read(8192)
        if not data: break
        f.write(data)
    printf("{nl}download complete")
    if rename:
      printf("renaming \"{name}\" to \"enigma.py\"")
      os.rename(name, "enigma.py")
  elif __version__ < v:
    print("enigma.py is NOT up to date")
  else:
    print("enigma.py is up to date")


@static(url='http://www.magwag.plus.com/jim/')
def _enigma_update(url=None, check=1, download=0, rename=0):
  """
  check enigma.py version, and download the latest version if
  necessary.

  this function is called by the -u command line option.

    % python enigma.py -u
    [enigma.py version 2019-07-06 (Python 3.7.4)]
    checking for updates...
    latest version is 2019-07-06
    enigma.py is up to date  

  check - set to check current version against latest
  download - set to always download latest version
  rename - set to rename downloaded file to enigma.py
  """
  print('checking for updates...')

  if url is None: url = _enigma_update.url

  try:
    __enigma_update(url, check=check, download=download, rename=rename)
  except IOError as e:
    print(e)
    printf("ERROR: failed to download update from {_enigma_update.url}")


__doc__ += """

COMMAND LINE USAGE:

enigma.py has the following command-line usage:

  python enigma.py

    The reports the current version of the enigma.py module, and the
    current python version:

      % python enigma.py
      [enigma.py version {version} (Python {python})]

      % python3 enigma.py
      [enigma.py version {version} (Python {python3})]


  python enigma.py -t[v]

    This will use the doctest module to run the example code given in
    the documentation strings.

    If -t is specified there should be no further output from the
    tests, unless one of them fails.

    If there are test failures on your platform, please let me know
    (along with information on the platform you are using and the
    versions of Python and enigma.py), and I'll try to fix the code
    (or the test) to work on your platform.

    If -tv is specified then more verbose information about the status
    of the tests will be provided.


  python enigma.py -u[cdr]

    The enigma.py module can be used to check for updates. Running
    with the -u flag will check if there is a new version of the
    module available (requires a function internet connection), and if
    there is it will download it.

    If the module can be updated you will see something like this:

      % python enigma.py -ur
      [enigma.py version 2013-09-10 (Python {python})]
      checking for updates...
      latest version is {version}
      downloading latest version to "{version}-enigma.py"
      ........
      download complete
      renaming "{version}-enigma.py" to "enigma.py"

    Note that the updated version is downloaded to a file named
    "<version>-enigma.py" in the current directory. You can then
    upgrade by renaming this file to "enigma.py" (this will happen
    automatically if the 'r' flag is passed).

    If you are running the latest version you will see something like
    this:

      % python enigma.py -u
      [enigma.py version {version} (Python {python})]
      checking for updates...
      latest version is {version}
      enigma.py is up to date

    If -uc is specified then the module will only check if an update
    is available, it won't download it.

    If -ud is specified then the latest version will always be
    downloaded.


  python enigma.py -h

    Provides a quick summary of the command line usage:

      % python enigma.py -h
      [enigma.py version {version} (Python {python})]
      command line arguments:
        <class> <args> = run command_line(<args>) method on class
        [-r | --run] <file> [<additional-args>] = run the solver and args specified in <file>
        -t[v] = run tests [v = verbose]
        -u[cdr] = check for updates [c = only check, d = always download, r = rename]
        -h = this help

  Solvers that support the command_line() class method can be invoked
  directly from the command line like this:

  python enigma.py <class> <args> ...

    Supported solvers are:
      SubstitutedSum
      SubstitutedDivision
      SubstitutedExpression

    For example, Enigma 327 can be solved using:

    % python enigma.py SubstitutedSum "KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE"
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5


    Enigma 440 can be solved using:

    % python enigma.py SubstitutedDivision "????? / ?x = ??x" "??? - ?? = ?" "" "??? - ??x = 0"
    ????? / ?x = ??x (rem 0) [??? - ?? = ?, None, ??? - ??x = 0]
    10176 / 96 = 106 (rem 0) [101 - 96 = 5, None, 576 - 576 = 0] / x=6
    [1 solution]


    Enigma 1530 can be solved using:

    % python enigma.py SubstitutedExpression "TOM * 13 = DALEY"
    (TOM * 13 == DALEY)
    (796 * 13 == 10348) / A=0 D=1 E=4 L=3 M=6 O=9 T=7 Y=8
    [1 solution]


    Alternatively the arguments to enigma.py can be placed in a text file
    and then executed with the --run / -r command, for example:

    % python enigma.py --run enigma327.run
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5

""".format(version=__version__, python='2.7.16', python3='3.7.4')

if __name__ == "__main__":

  # allow solvers to run from the command line:
  #   % python enigma.py <class> <args> ...
  # or put all the arguments into a file and use:
  #   % python enigma.py -r <file> <additional-args>
  #   % python enigma.py --run <file> <additional-args>
  #   % python enigma.py <file> <additional-args>
  args = argv()
  if args:
    run(*args)
    if _run_exit is not None:
      sys.exit(_run_exit)

  # identify the version number
  #print('[python version ' + sys.version.replace("\n", " ") + ']')
  printf('[enigma.py version {__version__} (Python {v})]', v=sys.version.split(None, 1)[0])

  # parse arguments into a dict
  args = dict((arg[1], arg[2:]) for arg in args if len(arg) > 1 and arg[0] == '-')

  # -h => help
  if 'h' in args:
    _enigma_help()

  # -t => run tests
  # -tv => in verbose mode
  if 't' in args:
    _enigma_test(verbose=('v' in args['t']))

  # -u => check for updates, and download newer version
  # -uc => just check for updates (don't download)
  # -ud => always download latest version
  # -u[d]r => rename downloaded file to "enigma.py"
  if 'u' in args:
    _enigma_update(check=('c' in args['u']), download=('d' in args['u']), rename=('r' in args['u']))
