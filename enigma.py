#!/usr/bin/env python -t
###############################################################################
#
# File:         enigma.py
# RCS:          $Header: $
# Description:  Useful routines for solving Enigma Puzzles
# Author:       Jim Randell
# Created:      Mon Jul 27 14:15:02 2009
# Modified:     Mon Apr  4 10:36:27 2016 (Jim Randell) jim.randell@gmail.com
# Language:     Python
# Package:      N/A
# Status:       Free for non-commercial use
#
# (C) Copyright 2009-2016, Jim Randell, all rights reserved.
#
###############################################################################
# -*- mode: Python; py-indent-offset: 2; -*-

"""
A collection of useful code for solving New Scientist Enigma (and similar) puzzles.

The latest version is available at <http://www.magwag.plus.com/jim/enigma.html>.

Currently this module provides the following functions and classes:

base2int              - convert a string in the specified base to an integer
C                     - combinatorial function (nCk)
cached                - decorator for caching functions
cbrt                  - the (real) cube root of a number
chunk                 - go through an iterable in chunks
compare               - comparator function
concat                - concatenate a list of values into a string
coprime_pairs         - generate coprime pairs
cslice                - cumulative slices of an array
csum                  - cumulative sum
diff                  - sequence difference
digrt                 - the digital root of a number
divc                  - ceiling division
divf                  - floor division
divisor_pairs         - generate pairs of divisors of a number
divisor               - generate the divisors of a number
divisors              - the divisors of a number
egcd                  - extended gcd
factor                - the prime factorisation of a number
factorial             - factorial function
farey                 - generate Farey sequences of coprime pairs
filter2               - partition an iterator into values that satisfy a predicate, and those that do not
filter_unique         - partition an iterator into values that are unique, and those that are not
find                  - find the index of an object in an iterable
find_max              - find the maximum value of a function
find_min              - find the minimum value of a function
find_value            - find where a function has a specified value
find_zero             - find where a function is zero
first                 - return items from the start of an iterator
flatten               - flatten a list of lists
flattened             - fully flatten a nested structure
gcd                   - greatest common divisor
grid_adjacency        - adjacency matrix for an n x m grid
hypot                 - calculate hypotenuse
icount                - count the number of elements of an iterator that satisfy a predicate
int2base              - convert an integer to a string in the specified base
int2roman             - convert an integer to a Roman Numeral
int2words             - convert an integer to equivalent English words
intc                  - ceiling conversion of float to int
intf                  - floor conversion of float to int
invmod                - multiplicative inverse of n modulo m
ipartitions           - partition a sequence with repeated values by index
irange                - inclusive range iterator
is_cube               - check a number is a perfect cube
is_distinct           - check a value is distinct from other values
is_duplicate          - check to see if value (as a string) contains duplicate characters
is_pairwise_distinct  - check all arguments are distinct
is_power              - check a number is a perfect power
is_prime              - simple prime test
is_prime_mr           - Miller-Rabin fast prime test
is_roman              - check a Roman Numeral is valid
is_square             - check a number is a perfect square
is_triangular         - check a number is a triangular number
isqrt                 - intf(sqrt(x))
join                  - concatenate strings
lcm                   - lowest common multiple
mgcd                  - multiple gcd
multiply              - the product of numbers in a sequence
nconcat               - concatenate single digits into an integer
number                - create an integer from a string ignoring non-digits
P                     - permutations function (nPk)
partitions            - partition a sequence of distinct values into tuples
pi                    - float approximation to pi
powerset              - the powerset of an iterator
prime_factor          - generate terms in the prime factorisation of a number
printf                - print with interpolated variables
recurring             - decimal representation of fractions
repdigit              - number consisting of repeated digits
roman2int             - convert a Roman Numeral to an integer
split                 - split a value into characters
sprintf               - interpolate variables into a string
sqrt                  - the (positive) square root of a number
subseqs               - sub-sequences of an iterable
substituted_sum       - a solver for substituted sums
T                     - T(n) is the nth triangular number
tau                   - tau(n) is the number of divisors of n
timed                 - decorator for timing functions
timer                 - a Timer object
trirt                 - the (positive) triangular root of a number
tuples                - generate overlapping tuples from a sequence
uniq                  - unique elements of an iterator

Accumulator           - a class for accumulating values
CrossFigure           - a class for solving cross figure puzzles
Football              - a class for solving football league table puzzles
MagicSquare           - a class for solving magic squares
Primes                - a class for creating prime sieves
SubstitutedDivision   - a class for solving substituted long division sums
SubstitutedSum        - a class for solving substituted addition sums
Timer                 - a class for measuring elapsed timings
"""

from __future__ import print_function

__author__ = "Jim Randell <jim.randell@gmail.com>"
__version__ = "2016-04-04"

__credits__ = """Brian Gladman, contributor"""

import sys

import operator
import math
import functools
import itertools
import collections

# maybe use the "six" module for some of this stuff
if sys.version_info[0] == 2:
  # Python 2.x
  range = xrange
  reduce = reduce
  basestring = basestring
  raw_input = raw_input
elif sys.version_info[0] > 2:
  # Python 3.x
  range = range
  reduce = functools.reduce
  basestring = str
  raw_input = input

# useful routines that can be re-exported
pi = math.pi
sqrt = math.sqrt

# useful routines for solving Enigma puzzles

# like cmp() in Python 2, but results are always -1, 0, +1.
def compare(a, b):
  """
  return -1 if a < b, 0 if a == b and +1 if b < a.

  >>> compare(42, 0)
  1
  >>> compare(0, 42)
  -1
  >>> compare(42, 42)
  0
  >>> compare('evil', 'EVIL')
  1
  """
  return (b < a) - (a < b)


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
  # it's probably faster to use a builtin...
  return len(set(args)) == len(args)
  ## even through the following may do fewer tests:
  #for i in range(len(args) - 1):
  #  if args[i] in args[i + 1:]: return False
  #return True

pairwise_distinct = is_pairwise_distinct

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
  return str(sep).join(str(x) for x in s)

def concat(*args, **kw):
  """
  return a string consisting of the concatenation of the elements of the sequence <args>.
  the elements will be converted to strings (using str(x)) before concatenation.

  you can use it instead of str.join() to join non-string lists by specifying a 'sep' argument.

  >>> concat('h', 'e', 'l', 'l', 'o')
  'hello'
  >>> concat(1, 2, 3, 4, 5)
  '12345'
  >>> concat(1, 2, 3, 4, 5, sep=',')
  '1,2,3,4,5'
  """
  sep = kw.get('sep', '')
  return join((str(x) for x in args), sep)


def nconcat(*digits, **kw):
  """
  return an integer consisting of the concatenation of the list <digits> of digits

  >>> nconcat(1, 2, 3, 4, 5)
  12345
  >>> nconcat(13, 14, 10, 13, base=16)
  57005
  >>> nconcat(123,456,789, base=1000)
  123456789
  >>> nconcat(127,0,0,1, base=256)
  2130706433
  """
  # in Python3 [[ def nconcat(*digits, base=10): ]] is allowed instead
  base = kw.get('base', 10)
  return reduce(lambda a, b: a * base + b, digits, 0)
  # or: (slower, and only works with digits < 10)
  #return int(concat(*digits), base=base)


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
  return base2int(s, base=base, strip=True)


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
def chunk(s, n=2):
  """
  iterate through iterable <s> in chunks of size <n>.

  (for overlapping tuples see tuples())

  >>> list(chunk(irange(1, 8)))
  [(1, 2), (3, 4), (5, 6), (7, 8)]
  >>> list(chunk(irange(1, 8), 3))
  [(1, 2, 3), (4, 5, 6), (7, 8)]
  """
  i = iter(s)
  n = range(n)
  while True:
    s = tuple(next(i) for x in n)
    if not len(s): break
    yield s


def diff(a, b, fn=None):
  """
  return the subsequence of <a> that excludes elements in <b>.

  >>> diff((1, 2, 3, 4, 5), (3, 5, 2))
  (1, 4)
  >>> join(diff('newhampshire', 'wham'))
  'nepsire'
  """
  return tuple(x for x in a if x not in set(b))


# recipe itertools documentation
def powerset(i, min_size=0, max_size=None):
  """
  generate the powerset (i.e. all subsets) of an iterator.

  >>> list(powerset((1, 2, 3)))
  [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
  """
  s = list(i)
  if max_size is None: max_size = len(s)
  return itertools.chain.from_iterable(itertools.combinations(s, n) for n in irange(min_size, max_size))

subsets = powerset


# see also partition() recipe from itertools documentation
# (but note that partition() returns (false, true) lists)
def filter2(p, i):
  """
  use a predicate to partition an iterable it those elements that
  satisfy the predicate, and those that do not.

  >>> filter2(lambda n: n % 2 == 0, irange(1, 10))
  ([2, 4, 6, 8, 10], [1, 3, 5, 7, 9])
  """
  t = list((x, p(x)) for x in i)
  return (list(x for (x, v) in t if v), list(x for (x, v) in t if not v))


def identity(x):
  """
  the identity function
  """
  return x

def filter_unique(s, f=identity, g=identity):
  """
  for every object, x, in sequence, s, consider the map f(x) -> g(x)
  and return a partition of s into those objects where f(x) implies a
  unique value for g(x), and those objects where f(x) implies multiple
  values for g(x).
  
  returns the partition of the original sequence as (<unique values>, <non-unique values>)

  See: Enigma 265 <https://enigmaticcode.wordpress.com/2015/03/14/enigma-265-the-parable-of-the-wise-fool/#comment-4167>

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


# count the number of occurrences of a predicate in an iterator
# TODO: rename this so it doesn't clash with itertools.count

def icount(i, p, t=None):
  """
  count the number of elements in iterator <i> that satisfy predicate <p>,
  the termination limit <t> controls how much of the iterator we visit,
  so we don't have to count all occurrences.

  So, to find if exactly <n> elements of <i> satisfy <p> use:

  icount(i, p, n + 1) == n

  This will examine all elements of <i> to verify there are exactly 4 primes
  less than 10:
  >>> icount(irange(1, 10), is_prime, 5) == 4
  True

  But this will stop after testing 73 (the 21st prime):
  >>> icount(irange(1, 100), is_prime, 21) == 20
  False

  To find if at least <n> elements of <i> satisfy <p> use:

  icount(i, p, n) == n

  The following will stop testing at 71 (the 20th prime):
  >>> icount(irange(1, 100), is_prime, 20) == 20
  True

  To find if at most <n> elements of <i> satisfy <p> use:

  icount(i, p, n + 1) < n + 1

  The following will stop testing at 73 (the 21st prime):
  >>> icount(irange(1, 100), is_prime, 21) < 21
  False

  """
  n = 0
  for x in i:
    if p(x):
      n += 1
      if n == t: break
  return n


# find, like index(), but return -1 instead of throwing an error

def find(s, v):
  """
  find the first index of a value in a sequence, return -1 if not found.

  for a string to works like str.find() (for single characters)
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
    for (i, x) in enumerate(s):
      if x == v: return i
    else:
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


def partitions(s, n, pad=False, value=None, distinct=None):
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
  d, r = divmod(len(s), n)
  if r > 0:
    if not pad: raise ValueError("invalid sequence length {l} for {n}-tuples".format(l=len(s), n=n))
    s = tuple(s) + (value,) * (n - r)
  if d == 0 or (d == 1 and r == 0):
    yield (s,)
  else:
    if distinct is None: distinct = is_pairwise_distinct(*s)
    fn = (_partitions if distinct else ipartitions)
    # or in Python 3: [[ yield from fn(s, n) ]]
    for p in fn(s, n): yield p


def first(i, count=1, skip=0):
  """
  return the first <count> items in iterator <i> (skipping the initial
  <skip> items) as a list.

  if you import itertools this would be a way to find the first 10 primes:
  >>> first((n for n in itertools.count(1) if is_prime(n)), count=10)
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  """
  return list(itertools.islice(i, skip, skip + count))


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
  generate unique values from iterator <i> (maintaining order),
  where values are compared using <fn>.

  this function assumes that common elements are generated in <i>
  together, so it only needs to track the last value.

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

F13 = 1.0 / 3.0

def cbrt(x):
  """
  Return the cube root of a number (as a float).

  >>> cbrt(27.0)
  3.0
  >>> cbrt(-27.0)
  -3.0
  """
  return (-math.pow(-x, F13) if x < 0 else math.pow(x, F13))

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
    ds = (1, 2)
    j = 0
    while i * i <= n:
      e = 0
      while True:
        (d, m) = divmod(n, i)
        if m > 0: break
        e += 1
        n = d
      if e > 0: yield (i, e)
      i += ds[j % 2]
      if j == 1: ds = (2, 4)
      j += 1
    # anything left is prime
    if n > 1: yield (n, 1)

# maybe should be called factors() or factorise()
def factor(n):
  """
  return a list of the prime factors of positive integer <n>.

  for integers less than 1, None is returned.

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
  for (p, e) in prime_factor(n):
    factors.extend([p] * e)
  return factors


def divisor_pairs(n):
  """
  generate divisors (a, b) of positive integer <n>, such that <a> * <b> = <n>.

  the pairs are generated such that <a> <= <b>, in order of increasing <a>.

  >>> list(divisor_pairs(36))
  [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6)]
  >>> list(divisor_pairs(101))
  [(1, 101)]
  """
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
    for i in irange(1, n):
      t.extend(x * p for x in s)
      p *= m
    s.extend(t)
  s.sort()
  return s


def divisors(n):
  """
  return the divisors of positive integer <n> as a sorted list.

  >>> divisors(36)
  [1, 2, 3, 4, 6, 9, 12, 18, 36]
  >>> divisors(101)
  [1, 101]
  """
  return multiples(prime_factor(n))


def is_prime(n):
  """
  return True if the positive integer <n> is prime.

  >>> is_prime(101)
  True
  >>> is_prime(1001)
  False

  """
  return n > 1 and len(factor(n)) == 1

prime = is_prime


# Miller-Rabin primality test - contributed by Brian Gladman
def is_prime_mr(n, r=10):
  """
  Miller-Rabin primality test for <n>.
  <r> is the number of rounds 

  if False is returned the number is definitely composite.
  if True is returned the number is probably prime.

  >>> is_prime_mr(341550071728361)
  True
  >>> is_prime_mr(341550071728321)
  False
  """
  if n < 10: return n in (2, 3, 5, 7)
  import random
  t = n - 1
  s = 0
  while not(t & 1):
    t >>= 1
    s += 1
  for i in range(r):
    a = random.randrange(2, n - 1)
    x = pow(a, t, n)
    if x != 1 and x != n - 1:
      for j in range(s - 1):
        x = (x * x) % n
        if x == 1:
          return False
        if x == n - 1:
          break
      else:
        return False
  return True


def tau(n):
  """
  count the number of divisors of a positive integer <n>.
  
  tau(n) = len(divisors(n)) (but faster)

  >>> tau(factorial(12))
  792
  """
  return multiply(e + 1 for (_, e) in prime_factor(n))


def farey(n):
  """
  generate the Farey sequence F(n) - the sequence of coprime
  pairs (a, b) where 0 < a < b <= n. pairs are generated
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

def coprime_pairs(n=None, order=False):
  """
  generate coprime pairs (a, b) with 0 < a < b <= n.

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
  for p in filter(fn, ((2, 1), (3, 1))):
    _push(ps, p)
  while ps:
    (b, a) = _pop(ps)
    yield (a, b)
    for p in filter(fn, ((2 * b - a, b), (2 * a + b, a), (2 * b + a, b))):
      _push(ps, p)


# if we don't overflow floats (happens around 2**53) this works...
#   def is_power(n, m):
#     i = int(n ** (1.0 / float(m)) + 0.5)
#     return (i if i ** m == n else None)
# but here we use Newton's method, which should work on arbitrary large integers
# and we special case m = 2 (is_square)
#
# NOTE: that this will return 0 if n = 0 and None if n is not a perfect m-th power,
# so [[ power(n, m) ]] will evaluate to True only for positive n
# if you want to allow n to be 0 you should check: [[ power(n, m) is not None ]]
def is_power(n, m):
  """
  check positive integer <n> is a perfect power of positive integer <m>.

  if <n> is a perfect <m>th power, returns the integer <m>th root.
  if <n> is not a perfect <m>th power, returns None.

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
  if n == 0: return 0
  # initial guess
  try:
    # make a sneaky close guess for non-huge numbers
    a = int(n ** (1.0 / float(m)))
  except OverflowError:
    # default initial guess
    a = 1
  # calculate successive approximations via Newton's method
  m1 = m - 1
  while True:
    d = (n // (a ** m1) - a) // m
    a += d
    if -2 < d < 2: break
  return (a if a ** m == n else None)


# it would be more Pythonic to encapsulate is_square in a class with the initialisation
# in __init__, and the actual call in __call__, and then instantiate an object to be
# the is_square() function (i.e. [[ is_square = _is_square_class(80) ]]), but it is
# more efficient (and perhaps more readable) to just use normal variables, although
# if you're using PyPy the class based version is just as fast (if not slightly faster)

# experimentally 80, 48, 72, 32 are good values (24, 16 also work OK)
_is_square_mod = 80
_is_square_residues = set((i * i) % _is_square_mod for i in range(_is_square_mod))
_is_square_reject = list(i not in _is_square_residues for i in range(_is_square_mod))

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
  if n == 0: return 0
  if n < 0: return None
  # early rejection: check <square> mod <some value> against a precomputed cache
  # e.g. <square> mod 80 = 0, 1, 4, 9, 16, 20, 25, 36, 41, 49, 64, 65 (rejects 88% of numbers)
  if _is_square_reject[n % _is_square_mod]: return None
  # make an initial guess
  try:
    # make a sneaky close guess for non-huge numbers
    a = int(n ** 0.5)
  except OverflowError:
    # default initial guess
    a = 1
  # calculate successive approximations via Newton's method
  while True:
    d = (n // a - a) // 2
    a += d
    if -2 < d < 2: break
  return (a if a * a == n else None)


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

# keep the old names as aliases
power = is_power
cube = is_cube
square = is_square

# calculate intf(sqrt(n)) using Newton's method
def isqrt(n):
  """
  calculate intf(sqrt(n)) using Newton's method.

  >>> isqrt(9)
  3
  >>> isqrt(15)
  3
  >>> isqrt(16)
  4
  >>> isqrt(17)
  4
  """
  if n == 0: return 0
  try:
    a = int(n ** 0.5 + 1e-6)
  except OverflowError:
    a = n
  while True:
    b = (a + n // a) // 2
    if a <= b: break
    a = b
  return a


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
  return (i if i * (i + 1) == n * 2 else None)


def digrt(n):
  """
  return the digital root of positive integer <n>.

  >>> digrt(123456789)
  9
  >>> digrt(sum((1, 2, 3, 4, 5, 6, 7, 8, 9)))
  9
  >>> digrt(factorial(100))
  9
  """
  return (0 if n == 0 else int(1 + (n - 1) % 9))


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
  >>> hex(repdigit(6, 7, base=16))
  '0x777777'
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


def is_duplicate(s):
  """
  check to see if <s> (as a string) contains duplicate characters.

  >>> is_duplicate("hello")
  True
  >>> is_duplicate("world")
  False
  >>> is_duplicate(99 ** 2)
  False
  """
  s = str(s)
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
  return (a * b) // gcd(a, b)


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

  >>> C(10, 3)
  120
  """
  if k > n:
    return 0
  else:
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)



def recurring(a, b, recur=False, base=10):
  """
  find recurring decimal representation of the fraction <a> / <b>
  return strings (<integer-part>, <non-recurring-decimal-part>, <recurring-decimal-part>)
  if you want rationals that normally terminate represented as non-terminating set <recur>

  >>> recurring(1, 7)
  ('0', '', '142857')
  >>> recurring(3, 2)
  ('1', '5', '')
  >>> recurring(3, 2, recur=True)
  ('1', '4', '9')
  >>> recurring(5, 17, base=16)
  ('0', '', '4B')
  """
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
      return (int2base(i, base), s[:j], s[j:])
    except KeyError:
      # no, we haven't
      r[a] = n
      n += 1
      (d, a) = divmod(base * a, b)
      if recur and a == 0: (d, a) = (d - 1, b)
      if not(d == a == 0):
        # add to the digit string
        s += int2base(d, base)


def _sprintf(fmt, vs, kw):
  if kw:
    vs = vs.copy()
    for (k, v) in kw.items():
      vs[k] = v
  return fmt.format(**vs)


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
  return _sprintf(fmt, sys._getframe(1).f_locals, kw)

# print with variables interpolated into the format string
def printf(fmt='', **kw):
  """
  print format string <fmt> with interpolated variables and keyword arguments.

  >>> (a, b, c) = (1, 2, 3)
  >>> printf("a={a} b={b} c={c}")
  a=1 b=2 c=3
  >>> printf("a={a} b={b} c={c}", c=42)
  a=1 b=2 c=42
  """
  print(_sprintf(fmt, sys._getframe(1).f_locals, kw))


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
      return c[k]
    except KeyError:
      r = c[k] = f(*k)
      return r
  return _cached


# inclusive range iterator
def irange(a, b=None, step=1):
  """
  a range iterator that includes both endpoints.

  if only one endpoint is specified then this is taken as the highest value,
  and a lowest value of 1 is used (so irange(n) produces n integers from 1 to n).

  >>> list(irange(1, 9))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(irange(9, 1, step=-1))
  [9, 8, 7, 6, 5, 4, 3, 2, 1]
  >>> list(irange(0, 10, step=3))
  [0, 3, 6, 9]
  >>> list(irange(9))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  """
  if b is None: (a, b) = (1, a)
  return range(a, b + (1 if step > 0 else -1), step)


# flatten a list of lists
def flatten(l, fn=list):
  """
  flatten a list of lists (actually an iterator of iterators).

  >>> flatten([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> flatten(((1, 2), (3, 4, 5), (6, 7, 8, 9)), fn=tuple)
  (1, 2, 3, 4, 5, 6, 7, 8, 9)
  >>> flatten([['abc'], ['def', 'ghi']])
  ['abc', 'def', 'ghi']
  """
  return fn(j for i in l for j in i)

# an iterator that fully flattens a nested structure
def flattened(s, n=None):
  """
  fully flatten a nested structure <s> (to depth <n>, default is to fully flatten).

  >>> list(flattened([[1, [2, [3, 4, [5], [[]], [[6, 7], 8], [[9]]]], []]]))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(flattened([[1, [2, [3, 4, [5], [[]], [[6, 7], 8], [[9]]]], []]], 3))
  [1, 2, 3, 4, [5], [[]], [[6, 7], 8], [[9]]]
  >>> list(flattened([['abc'], ['def', 'ghi']]))
  ['abc', 'def', 'ghi']
  """
  n1 = (None if n is None else n - 1)
  for i in s:
    if (isinstance(i, collections.Sequence) and not isinstance(i, basestring)) and (n is None or n > 0):
      for j in flattened(i, n1):
        yield j
    else:
      yield i


# adjacency matrix for an n (columns) x m (rows) grid
# entries are returned as lists in case you want to modify them before use
def grid_adjacency(n, m, deltas=None, include_adjacent=True, include_diagonal=False):
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

  if 'deltas' is not specified the 'include_adjacent' and 'include_diagonal'
  flags are used to specify which squares are adjacent to the target square.
  'include_adjacent' includes the N, S, E, W squares
  'include_diagonal' includes the NW, NE, SW, SE squares

  >>> grid_adjacency(2, 2)
  [[1, 2], [0, 3], [0, 3], [1, 2]]
  >>> sorted(grid_adjacency(3, 3)[4])
  [1, 3, 5, 7]
  >>> sorted(grid_adjacency(3, 3, include_diagonal=True)[4])
  [0, 1, 2, 3, 5, 6, 7, 8]
  """
  # if deltas aren't provided use standard deltas
  if deltas is None:
    deltas = list()
    if include_adjacent: deltas.extend([(0, -1), (-1, 0), (1, 0), (0, 1)])
    if include_diagonal: deltas.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])
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
    for _ in irange(1, n):
      t.append(next(i))
    while True:
      # return the tuple
      yield tuple(t)
      # move the next value in to the tuple
      t.pop(0)
      t.append(next(i))
  except StopIteration:
    pass


# subseqs: generate the subsequences of an iterator
def subseqs(iterable, min_size=0, max_size=None):
  """
  generate the sub-sequences of an iterable.
  min_size and max_size can be used to limit the length of the sub-sequences.

  >>> list(subseqs((1, 2, 3)))
  [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
  >>> list(subseqs((1, 2, 3), min_size=1, max_size=2))
  [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
  """
  l = list(iterable)
  n = len(l)
  r_min = min(min_size, n)
  r_max = (n if max_size is None else min(max_size, n))
  for r in irange(r_min, r_max):
    for s in itertools.combinations(l, r):
      yield s


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
      yield k, d[k]

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
  (f1, f2) = fn(x1), fn(x2)
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
  >>> r = find_zero(lambda x: x ** 2 + 4, 0.0, 10.0)
  Traceback (most recent call last):
    ...
  AssertionError: Value not found
  """
  r = find_min(f, a, b, t, m=abs)
  assert not(ft < abs(r.fv)), "Value not found"
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
  s = ''
  for (n, i, m) in _romans:
    (d, r) = divmod(x, i)
    if d < 1: continue
    s += n * d
    x = r
  return s


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




# digits for use in converting bases
_digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def int2base(i, base=10, digits=_digits):
  """
  convert an integer <i> to a string representation in the specified base <base>.

  By default this routine only handles single digits up 36 in any given base,
  but the digits parameter can be specified to give the symbols for larger bases.

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
  """
  assert base > 1
  if i == 0: return digits[0]
  elif i < 0: return '-' + int2base(-i, base=base, digits=digits)
  r = ''
  while i > 0:
    (i, n) = divmod(i, base)
    r = digits[n] + r
  return r

def base2int(s, base=10, strip=False, digits=_digits):
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

###############################################################################

# specialised classes:

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
  """

  def __init__(self, fn=operator.add, value=None, data=None):
    """
    create an Accumulator.

    The accumulation function and initial value can be specified.
    """
    self.fn = fn
    self.value = value
    self.data = data

  def __repr__(self):
    return 'Accumulator(value=' + repr(self.value) + ', data=' + repr(self.data) + ')'

  def accumulate(self, v=1):
    """
    Accumulate a value.

    If the current value is None then this value replaces the current value.
    Otherwise it is combined with the current value using the accumulation
    function which is called as fn(<current-value>, v).
    """
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


###############################################################################

# Prime Sieves

_primes_array = bytearray
_primes_chunk = lambda n: 2 * n
_primes_size = 1024


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
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

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
  # p->i = p // 3
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


  def extend(self, n):
    """
    extend the sieve up to (at least) n
    """
    if not(n > self.max): return

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

    (this will require less memory than list())
    """
    if end is None: end = self.max
    if start < 3 and end > 1: yield 2
    if start < 4 and end > 2: yield 3
    s = self.sieve
    for i in range(start // 3, (end + 1) // 3):
      if s[i]: yield (i * 3) + (i & 1) + 1

  # make this an iterable object
  __iter__ = generate

  # range(a, b) - generate primes in the (inclusive) range [a, b] - is the same as generate now
  range = generate

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
  def prime_factor(self, n):
    """
    generate (<prime>, <exponent>) pairs in the prime factorisation of positive integer <n>.

    Note: This will only consider primes up to the limit of the sieve, this is a complete
    factorisation for <n> up to the square of the limit of the sieve.
    """
    # maybe should be: n < 1
    #if n < 0: raise ValueError("can only factorise positive integers")
    if n > 1:
      for p in self.generate():
        if n < p: return
        e = 0
        while True:
          (d, r) = divmod(n, p)
          if r > 0: break
          e += 1
          n = d
        if e > 0: yield (p, e)


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

    when the sieve is expanded the function <fn> is used to calculate the new maximum,
    based on the previous maximum.

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

  # for backwards compatability
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
    primaility test - the sieve is expanded as necessary before testing.
    """
    self.extend(n)
    return _PrimeSieveE6.is_prime(self, n)

  # allows use of "in"
  __contains__ = is_prime

  # expand the sieve as necessary
  def range(self, a, b):
    """
    generate primes in the (inclusive) range [a, b].

    the sieve is expanded as necessary beforehand.
    """
    self.extend(b)
    return _PrimeSieveE6.range(self, a, b)

  # expand the sieve as necessary
  def prime_factor(self, n):
    self.extend(isqrt(n))
    return _PrimeSieveE6.prime_factor(self, n)
  

# create a suitable prime sieve
def Primes(n=None, expandable=False, array=_primes_array, fn=_primes_chunk):
  """
  Return a suitable prime sieve object.

  n - initial limit of the sieve (the sieve contains primes up to n)
  expandable - should the sieve expand as necessary
  array - list implelementation to use
  fn - function used to increase the limit on expanding sieves

  If we are interested in a limited collection of primes, we can do this:

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

  >>> primes = Primes(50, expandable=True)

  We can find out the current size and contents of the sieve:
  >>> primes.max
  50
  >>> primes.list()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

  But if we use it as a generator it will expand indefiniately, so we
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

# backwards compatability
def PrimesGenerator(n=None, array=_primes_array, fn=_primes_chunk):
  return Primes(n, expandable=True, array=array, fn=fn)

###############################################################################

# Magic Square Solver:

# this is probably a bit of overkill but it works and I already had the code written

import copy

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
    if numbers is None:
      numbers = set(range(1, n2 + 1))
    s = sum(numbers) // n

    # make the magic lines
    if lines is None:
      lines = []
      l = tuple(range(n))
      for i in range(n):
        # row
        lines.append(tuple(i*n + j for j in l))
        # column
        lines.append(tuple(i + j*n for j in l))
      # diagonals
      lines.append(tuple(j*(n+1) for j in l))
      lines.append(tuple((j+1)*(n-1) for j in l))

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
    if not self.numbers: raise Solved
    for line in self.lines:
      ns = tuple(self.square[i] for i in line)
      z = ns.count(0)
      if z == 0 and sum(ns) != self.s: raise Impossible
      if z != 1: continue
      v = self.s - sum(ns)
      if not v in self.numbers: raise Impossible
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
    if not self.numbers: raise Solved
    i = self.square.index(0)
    for v in self.numbers:
      new = self.clone()
      new.set(i, v)
      if new.solve():
        self.become(new)
        return True
    raise Impossible
  
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
    _l2d = l2d.copy()
    _l2d.update(zip(u, ds))
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
    d2i[0] = join(uniq(x[0] for x in itertools.chain(terms, [result]) if len(x) > 1))
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

  Enigma 21: "Solve: BPPDQPC + PRPDQBD = XVDWTRC"
  <https://enigmaticcode.wordpress.com/2012/12/23/enigma-21-addition-letters-for-digits/>
  >>> SubstitutedSum(['BPPDQPC', 'PRPDQBD'], 'XVDWTRC').go()
  B=3 C=7 D=0 P=5 Q=6 R=8 T=2 V=4 W=1 X=9 / 3550657 + 5850630 = 9401287

  Enigma 63: "Solve: LBRLQQR + LBBBESL + LBRERQR + LBBBEVR = BBEKVMGL"
  <https://enigmaticcode.wordpress.com/2013/01/08/enigma-63-addition-letters-for-digits/>
  >>> SubstitutedSum(['LBRLQQR', 'LBBBESL', 'LBRERQR', 'LBBBEVR'], 'BBEKVMGL').go()
  B=3 E=2 G=5 K=7 L=8 M=9 Q=4 R=0 S=1 V=6 / 8308440 + 8333218 + 8302040 + 8333260 = 33276958
  """
  
  def __init__(self, terms, result, digits=None, l2d=None, d2i=None, base=10):
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
    self.terms = terms
    self.result = result
    self.digits = digits
    self.l2d = l2d
    self.d2i = d2i
    self.base = base
    self.text = join(terms, sep=' + ') + ' = ' + result

  def solve(self, fn=None):
    """
    generate solutions to the substituted addition sum puzzle.

    solutions are returned as a dict assigning letters to digits. 
    """
    if fn is None: fn = lambda x: True
    for r in substituted_sum(self.terms, self.result, digits=self.digits, l2d=self.l2d, d2i=self.d2i, base=self.base):
      if fn(r):
        yield r

  def substitute(self, l2d, text, digits=_digits):
    """
    given a solution to the substituted sum and some text return the text with
    letters substituted for digits.
    """
    return join((digits[l2d[c]] if c in l2d else c) for c in text)

  def solution(self, l2d):
    """
    given a solution to the substituted sum output the assignment of letters
    to digits and the sum with digits substituted for letters.
    """
    print(
      # output the assignments in letter order
      join((k + '=' + str(l2d[k]) for k in sorted(l2d.keys())), sep=' '), '/',
      # print the sum with digits substituted in
      self.substitute(l2d, self.text)
    )

  def go(self, fn=None):
    """
    find all solutions (matching the filter <fn>) and output them.
    """
    for s in self.solve(fn):
      self.solution(s)

  # class method to call from the command line
  @classmethod
  def command_line(cls, args):
    """
    run the SubstitutedSum solver with the specified command line arguments.

    e.g. Enigma 327 <https://enigmaticcode.wordpress.com/2016/01/08/enigma-327-it-all-adds-up/>
    % python enigma.py SubstitutedSum 'KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE'
    [solving KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE ...]
    A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5 / 1912803 + 2428850 + 4312835 = 8654488


    Optional parameters:

    --base=<n> (or -b<n>)

    Specifies the numerical base of the sum, solutions are printed in the
    specified base (so that the letters and digits match up, but don't get
    confused by digits that are represented by letters in bases >10).

    e.g. Enigma 1663 <https://enigmaticcode.wordpress.com/2011/12/04/enigma-1663-flintoffs-farewell/>
    % python enigma.py SubstitutedSum --base=11 'FAREWELL + FREDALO = FLINTOFF'
    [solving FAREWELL + FREDALO = FLINTOFF ...]
    A=1 D=10 E=7 F=6 I=0 L=8 N=4 O=9 R=5 T=2 W=3 / 61573788 + 657A189 = 68042966
    A=1 D=3 E=7 F=6 I=0 L=8 N=4 O=9 R=5 T=2 W=10 / 6157A788 + 6573189 = 68042966


    --assign=<letter>,<digit> (or -a<l>,<d>)

    Assign a digit to a letter.

    There can be multiple --assign options.

    e.g. Enigma 1361 <https://enigmaticcode.wordpress.com/2013/02/20/enigma-1361-enigma-variation/>
    % python enigma.py SubstitutedSum --assign=O,0 'ELGAR + ENIGMA = NIMROD'
    [solving ELGAR + ENIGMA = NIMROD ...]
    A=3 D=2 E=7 G=4 I=5 L=1 M=6 N=8 O=0 R=9 / 71439 + 785463 = 856902

    
    --digits=<digit>,<digit>,... (or -d<d>,<d>,...)

    Specify the digits that can be assigned to unassigned letters.
    (Note that the values of the digits are specified in base 10, even if you
    have specified a --base=<n> option)

    e.g. Enigma 1272 <https://enigmaticcode.wordpress.com/2014/12/09/enigma-1272-jonny-wilkinson/>
    % python enigma.py SubstitutedSum --digits=0,1,2,3,4,5,6,7,8 'WILKI + NSON = JONNY'
    [solving WILKI + NSON = JONNY ...]
    I=8 J=5 K=0 L=6 N=3 O=2 S=7 W=4 Y=1 / 48608 + 3723 = 52331
    I=8 J=5 K=0 L=7 N=3 O=2 S=6 W=4 Y=1 / 48708 + 3623 = 52331


    --invalid=<digit>,<letters> (or -i<d>,<ls>)

    Specify letters that cannot be assigned to a digit.

    There can be multiple --invalid options, but they should be for different
    <digit>s.

    If there are no --invalid options the default will be that leading zeros are
    not allowed. If you want to allow them you can give a "--invalid=0," option.

    Enigma 171 <https://enigmaticcode.wordpress.com/2014/02/23/enigma-171-addition-digits-all-wrong/>
    % python enigma.py SubstitutedSum -i0,016 -i1,1 -i3,3 -i5,5 -i6,6 -i7,7 -i8,8 -i9,9 '1939 + 1079 = 6856'
    [solving 1939 + 1079 = 6856 ...]
    0=1 1=2 3=6 5=0 6=4 7=3 8=9 9=7 / 2767 + 2137 = 4904


    --help (or -h)

    Prints a usage summary.
    """

    usage = join((
      sprintf("usage: {cls.__name__} [<opts>] <term> <term> ... <result>"),
      "options:",
      "  --base=<n> (or -b<n>) = set base",
      "  --assign=<letter>,<digit> (or -a<l>,<d>) = assign digit to letter",
      "  --digits=<digit>,<digit>,... (or -d<d>,<d>,...) = available digits",
      "  --invalid=<digit>,<letters> (or -i<d>,<ls>) = invalid digit to letter assignments",
      "  --help (or -h) = show command-line usage",
    ), sep="\n")

    # process options (--<key>[=<value>] or -<k><v>)
    opt = { 'base': 10, 'digits': None, 'l2d': dict(), 'd2i': None }
    while args and args[0].startswith('-'):
      arg = args.pop(0)
      try:
        if arg.startswith('--'):
          (k, v) = arg.lstrip('-').split('=', 1)
        else:
          (k, v) = (arg[1], arg[2:])
        if k == 'h' or k == 'help':
          print(usage)
          return -1
        elif k == 'b' or k == 'base':
          # --base=<n> (or -b)
          opt[k] = int(v)
        elif k == 'a' or k == 'assign':
          # --assign=<letter>,<digit> (or -a)
          (l, d) = v.split(',', 1)
          opt['l2d'][l] = int(d)
        elif k == 'd' or k == 'digits':
          # --digits=<digit>,<digit>,<digit>,... (or -d)
          ds = v.split(',')
          opt['digits'] = tuple(int(d) for d in ds)
        elif k == 'i' or k == 'invalid':
          # --invalid=<digit>,<letters> (or -i)
          if opt['d2i'] is None: opt['d2i'] = dict()
          (d, s) = v.split(',', 1)
          opt['d2i'][int(d)] = s
        else:
          raise ValueError
      except:
        printf("{cls.__name__}: invalid option: {arg}")
        return -1

    # check command line usage
    if not args:
      print(usage)
      return -1

    # extract the terms and result
    import re
    args = re.split(r'[\s\+\=]+', join(args, sep=' '))
    result = args.pop()
    printf("[solving {terms} = {result} ...]", terms=join(args, sep=' + '))

    # call the solver
    cls(args, result, **opt).go()
    return 0

###############################################################################

# Substituted Division Solver

# originally written for Enigma 206, but applicable to lots of Enigma puzzles

SubstitutedDivisionSolution = collections.namedtuple('SubstitutedDivisionSolution', 'a b c r d')

class SubstitutedDivision(object):
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
                =

  In this example there are the following intermediate (subtraction) sums:

    pkm - pmd = xp, xpk - ?? = kh, khh - mbg = k

  The first term in each of these sums can be inferred from the
  dividend and result of the previous intermediate sum, so we don't
  need to specify them.

  When the result contains a 0 digit there is no corresponding
  intermediate sum, in this case the intermediate sum is specified as None.

  In problems like Enigma 309 where letters are specified in parts of
  the intermediate sums that are copied down from the dividend, but
  not specified in the dividend itself you need to copy the letter back
  into the dividend manually, or fully specify the intermediate with the
  additional information in it.
  See <https://enigmaticcode.wordpress.com/2015/09/10/enigma-309-missing-letters/>

  Enigma 206 <https://enigmaticcode.wordpress.com/2014/07/13/enigma-206-division-some-letters-for-digits-some-digits-missing/>

  >>> SubstitutedDivision('pkmkh', '??', '???', [('pmd', 'xp'), ('??', 'kh'), ('mbg', 'k')]).go()
  47670 / 77 = 619 rem 7 [b=9 d=2 g=3 h=0 k=7 m=6 p=4 x=1] [476 - 462 = 14, 147 - 77 = 70, 700 - 693 = 7]


  Enigma 250 <https://enigmaticcode.wordpress.com/2015/01/13/enigma-250-a-couple-of-sevens/>
  >>> SubstitutedDivision('7?????', '??', '?????', [('??', '??'), ('??', '?'), None, ('??', '??'), ('??7', '')], { '7': 7 }).go()
  760287 / 33 = 23039 rem 0 [7=7] [76 - 66 = 10, 100 - 99 = 1, 128 - 99 = 29, 297 - 297 = 0]


  Enigma 309 <https://enigmaticcode.wordpress.com/2015/09/10/enigma-309-missing-letters/>
  >>> SubstitutedDivision('h???g?', '??', 'm?gh', [('g??', '??'), ('ky?', '?'), ('m?', 'x'), ('??', '')]).go()
  202616 / 43 = 4712 rem 0 [g=1 h=2 k=3 m=4 x=8 y=0] [202 - 172 = 30, 306 - 301 = 5, 51 - 43 = 8, 86 - 86 = 0]


  Enigma 309 <https://enigmaticcode.wordpress.com/2015/09/10/enigma-309-missing-letters/>
  >>> SubstitutedDivision('h?????', '??', 'm?gh', [('g??', '??'), ('ky?', '?'), ('?g', 'm?', 'x'), ('??', '')]).go()
  202616 / 43 = 4712 rem 0 [g=1 h=2 k=3 m=4 x=8 y=0] [202 - 172 = 30, 306 - 301 = 5, 51 - 43 = 8, 86 - 86 = 0]
  """

  def __init__(self, dividend, divisor, result, intermediates, mapping={}, wildcard='?', digits=None):
    """
    create a substituted long division puzzle.

    a / b = c remainder r

    dividend - the dividend (a)
    divisor - the divisor (b)
    result - the result (c)
    intermediates - a list of pairs representing the intermediate subtraction sums

    The following parameters are optional:
    mapping - initial map of letters to digits (default: all letters are unassigned)
    wildcard - the wildcard character used in the sum (default: '?')
    digits - set of digits to use (default: the digits 0 to 9)
    """
    self.dividend = dividend
    self.divisor = divisor
    self.result = result
    self.intermediates = intermediates
    self.mapping = mapping
    self.wildcard = wildcard
    self.base = 10
    self.digits = (set(irange(0, self.base - 1)) if digits is None else digits)
    # checks: there should be an intermediate for each digit of the result
    assert len(result) == len(intermediates), "result/intermediate length mismatch"

  # solutions are generated as a named tuple (a, b, c, r, d)
  # where a / b = c rem r, and d is mapping of symbols to digits
  def solve(self, fn=None):
    """
    generate solutions to the substituted long division sum puzzle.

    solutions are returned as a SubstitutedDivisionSolution() object with the following attributes:
    a - the dividend
    b - the divisor
    c - the result
    r - the remainder
    d - the map of letters to digits
    """

    # first let's have some internal functions useful for the solver
    wildcard = self.wildcard
    base = self.base
    digits = self.digits

    # update a map consistently with (key, value) pairs
    # an updated mapping will be returned, or None
    def update(d, kvs):
      for (k, v) in kvs:
        # check v is an allowable digit
        if v not in digits: return None
        if k == wildcard:
          # if k is a wildcard then that's OK
          pass
        elif k in d:
          # if k is already in the map it had better map to v
          if d[k] != v: return None
        else:
          # both k and v should be new values in the map
          if v in d.values(): return None
          # update the map
          d = d.copy()
          d[k] = v
      return d

    # match a string <s> to a number <n> using mapping <d>
    # an updated mapping will be returned, or None
    def match_number(d, s, n):
      # the empty string matches 0
      if s == '': return (d if n == 0 else None)
      # split the number into digits
      ns = split(int2base(n, base), int)
      # they should be the same length
      if len(s) != len(ns): return None
      # try to update the map
      return update(d, zip(s, ns))

    # match multiple (<s>, <n>) pairs
    def match_numbers(d, *args):
      for (s, n) in args:
        d = match_number(d, s, n)
        if d is None: break
      return d

    # generate possible numbers matching <s> with dictionary <d>
    def generate_numbers(s, d, slz=True):
      # the empty string matches 0
      if s == '':
        yield (0, d)
      elif len(s) == 1:
        if s == wildcard:
          for x in digits:
            if not(slz and x == 0):
              yield (x, d)
        elif s in d:
          x = d[s]
          if not(slz and x == 0):
            yield (x, d)
        else:
          for x in digits:
            if not(slz and x == 0):
              d2 = update(d, [(s, x)])
              if d2:
                yield (x, d2)
      else:
        # multi-character string, do the final character
        for (y, d1) in generate_numbers(s[-1], d, False):
          # and the rest
          for (x, d2) in generate_numbers(s[:-1], d1, slz):
            yield (base * x + y, d2)

    # match strings <md> against single digit multiples of <n>, according to map <d>
    # return (<list of single digits>, <map>)
    def generate_multiples(ms, n, d):
      if not ms:
        yield ([], d)
      else:      
        # find a match for the first one
        (m, s) = ms[0]
        # special case: s is None matches 0
        if s is None:
          if m == wildcard:
            for (x, d2) in generate_multiples(ms[1:], n, d):
              yield ([0] + x, d2)
          elif m in d:
            if d[m] == 0:
              for (x, d2) in generate_multiples(ms[1:], n, d):
                yield ([0] + x, d2)
          else:
            if 0 not in d.values():
              d2 = d.copy()
              d2[m] = 0
              for (x, d3) in generate_multiples(ms[1:], n, d2):
                yield ([0] + x, d3)
        # if m is already allocated
        elif m in d:
          i = d[m]
          d2 = match_number(d, s, i * n)
          if d2 is not None:
            for (x, d3) in generate_multiples(ms[1:], n, d2):
              yield ([i] + x, d3)
        # generate allowable non-zero digits for wildcard/new assignment
        else:
          for i in digits:
            if i == 0: continue
            if m != wildcard and i in d.values(): continue
            d2 = match_number(d, s, i * n)
            if d2 is None: continue
            # and the rest
            for (x, d3) in generate_multiples(ms[1:], n, d2):
              d4 = (d3 if m == wildcard else update(d3, [(m, i)]))
              if d4 is not None:
                yield ([i] + x, d4)

    # match the intermediates/dividend against the actual values in a/b = c
    # an updated mapping will be returned, or None
    def match_intermediates(a, b, c, zs, dividend, d):
      i = len(int2base(a, base)) - len(zs)
      sx = dividend[:i]
      sxr = list(dividend[i:])
      n = base ** (len(zs) - 1)
      z = 0
      # consider each intermediate sum: x - y = z
      for s in zs:
        sx = sx + sxr.pop(0)        
        (x, a) = divmod(a, n)
        x = base * z + x
        (y, c) = divmod(c, n)
        y *= b
        z = x - y
        # the remainder must be in the correct range
        if not(0 <= z < b): return
        if s is None:
          # there is no intermediate when y == 0
          assert y == 0
        else:
          d = match_numbers(d, (s, z), (sx, x))
          if d is None: return
          sx = s
        n //= base
      yield d

    # now for the actual solver

    dividend = self.dividend
    divisor = self.divisor
    result = self.result
    intermediates = self.intermediates
    if fn is None: fn = lambda x: True

    # in the intermediates (x - y = z) we're interested in the ys and the zs
    xs = list((None if s is None or len(s) < 3 else s[-3]) for s in intermediates)
    ys = list((None if s is None else s[-2]) for s in intermediates)
    zs = list((None if s is None else s[-1]) for s in intermediates)

    # if there are any xs specified we need to check them against the dividend
    # and copy up any specified letters that are not specified in the dividend
    if any(x is not None for x in xs):
      d = list(dividend)
      (a, b) = (0, len(d) - len(ys) + 1)
      for (x, y) in zip(xs, ys):
        # if there's no intermediate sum...
        if y is None:
          # ... an extra element from the dividend is copied next time
          b += 1
          continue
        # if the x part of the sum is specified
        if x is not None:
          # check it against the corresponding part of the dividend
          for (i, (p, q)) in enumerate(zip(d[a:b], x[a - b:]), start=a):
            # if something is specified in the intermediate...
            if q != wildcard:
              # ... but not in the dividend...
              if p == wildcard:
                # then update the dividend
                d[i] = q
              else:
                # otherwise they should be the same
                assert p == q, "dividend/intermediate mismatch"
        # go on to the next intermediate
        a = b
        b += 1
      dividend = join(d)

    # list of (<digit of result>, <multiple of divisor>) pairs
    ms = list(zip(result, ys))

    # initial mapping of letters to digits
    d0 = self.mapping

    # consider the sum as: a / b = c remainder r

    # consider possible divisors (b)
    for (b, d1) in generate_numbers(divisor, d0):

      # find multiples of the divisor that match
      for (cd, d2) in generate_multiples(ms, b, d1):
        # the result (c) is now defined
        c = nconcat(*cd, base=base)

        # find possible remainders (r)
        for (r, d3) in generate_numbers(zs[-1], d2):
          # so the actual sum is a / b = c rem r
          a = b * c + r
          # check that the computed dividend matches the template
          d4 = match_number(d3, dividend, a)
          if d4 is None: continue

          # now check the intermediate results match up
          for d5 in match_intermediates(a, b, c, zs, dividend, d4):
            s = SubstitutedDivisionSolution(a, b, c, r, d5)
            if fn(s):
              yield s
    
  # generate actual intermediates
  # note: "empty" intermediate sums are not returned
  def solution_intermediates(self, s):
    """
    generate the actual intermediate subtraction sums.

    Note: "empty" intermediate sums (specified as None in the input) are not returned.
    """
    (a, x, c, intermediates) = (list(int2base(s.a, self.base)), 0, 0, [])
    while a:
      x = x * self.base + int(a.pop(0))
      (y, r) = divmod(x, s.b)
      if y == 0: continue
      intermediates.append((x, s.b * y, r))
      c = c * self.base + y
      x = r
    return intermediates

  # substituted text using the solution
  def substitute(self, s, text, digits=_digits):
    """
    given a solution to the substituted division sum and some text,
    return the text with the letters substituted for digits.
    """
    return join((digits[s.d[c]] if c in s.d else c) for c in text)
  
  # output the solution
  def solution(self, s):
    """
    output a solution in the form:

    a / b = c rem r [mapping] [intermediates]
    """
    (a, b, c, r, d) = s
    printf("{a} / {b} = {c} rem {r} [{d}] [{intermediates}]",
           d=join((k + '=' + str(d[k]) for k in sorted(d.keys())), sep=' '),
           intermediates=join((sprintf("{x} - {y} = {z}") for (x, y, z) in self.solution_intermediates(s)), sep=', '))

  # find all solutions and output them
  def go(self, fn=None):
    """
    find all solutions (matching filter function <fn>) and output them.
    """
    for s in self.solve(fn):
      self.solution(s)

  # class method to call from the command line
  @classmethod
  def command_line(cls, args):
    """
    run the SubstitutedDivision solver with the specified command line arguments.

    e.g. for Enigma 309: (note use of # to denote the empty string)
    % python enigma.py SubstitutedDivision 'h????? / ?? = m?gh' 'h?? - g?? = ??' '??? - ky? = ?' '?g - m? = x' 'x? - ?? = #'
    [solving h????? / ?? = m?gh, [('h??', 'g??', '??'), ('???', 'ky?', '?'), ('?g', 'm?', 'x'), ('x?', '??', '')] ...]
    202616 / 43 = 4712 rem 0 [g=1 h=2 k=3 m=4 x=8 y=0] [202 - 172 = 30, 306 - 301 = 5, 51 - 43 = 8, 86 - 86 = 0]

    e.g for Enigma 440: (note use of empty argument for missing intermediate)
    % python enigma.py SubstitutedDivision '????? ?x ??x' '?? ?' '' '??x #'
    [solving ????? / ?x = ??x, [('??', '?'), None, ('??x', '')] ...]
    10176 / 96 = 106 rem 0 [x=6] [101 - 96 = 5, 576 - 576 = 0]    
    """

    # check command line usage
    if len(args) < 2:
      printf("usage: {cls.__name__} '<a> / <b> = <c>' '[<x> - <y> = <z>] | [<y> <z>]' ...")
      return -1

    # extract the terms and result
    import re
    (a, b, c) = re.split(r'[\s\/\=]+', args[0])
    # intermediate sums: empty string is denoted by '#', empty intermediate is denoted by ''
    intermediates = list(map((lambda x: (None if x == [''] else x)), (re.split(r'[\s\-\=\#]+', x) for x in args[1:])))
    printf("[solving {a} / {b} = {c}, {intermediates} ...]", intermediates=list(None if x is None else tuple(x) for x in intermediates))

    # call the solver
    cls(a, b, c, intermediates).go()
    return 0


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

  def update(self, g, t, n):
    g = list(g)
    for (i, d) in zip(t, n):
      g[i] = d
    return g

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
        grid2 = self.update(grid, ans, n)
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
  Useful utility routines for solving Football League Table puzzles.

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
      points = { 'w': 2, 'd': 1 }
    if swap is None:
      swap = { 'w': 'l', 'l': 'w' }

    self._games = tuple(games)
    self._points = points
    self._swap = swap
    self._table = collections.namedtuple('Table', ('played',) + self._games + ('points',))

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

    The returned table object has attributes named by the possible
    match outcomes (by default, .w, .d, .l, .x) and also .played (the
    total number of games played) and .points (calculated points).

    B = football.table([ab, bc, bd], [1, 0, 0])
    print(B.played, B.w, B.d, B.l, B.points)
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
  def outcomes(self, ss):
    """
    return a sequence of outcomes ('x', 'w', 'd', 'l') for a sequence of scores.
    """
    return tuple(('x' if s is None else 'ldw'[compare(s[0], s[1]) + 1]) for s in ss)


###############################################################################

# Timing

import atexit
import time

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


  def __init__(self, name='timing', timer=_timer, file=sys.stderr, exit_report=True, auto_start=True):
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
      atexit.register(self.report, force=False)
      self._exit_report = False
    self._t1 = None
    self._t0 = self._timer()

  def stop(self):
    """
    Set the stop time of a timer.
    """
    self._t1 = self._timer()

  def elapsed(self, disable_report=True):
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

  def report(self, force=True):
    """
    Stop the timer and generate the report (if required).

    The report will only be generated once (if it's not been disabled).
    """
    if self._report and not(force): return self._report
    if self._t1 is None: self.stop()
    e = self.elapsed()
    self._report = "[{n}] elapsed time: {e:.7f}s ({f})".format(n=self._name, e=e, f=self.format(e))
    print(self._report, file=self._file)

  def printf(self, fmt='', **kw):
    e = self.elapsed()
    print('[{n} {e}] {s}'.format(n=self._name, e=self.format(e), s=_sprintf(fmt, sys._getframe(1).f_locals, kw)))

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
timer = Timer(auto_start=False)

###############################################################################

# check for updates to enigma.py
# check = only check the current version
# download = always download the latest version
def _enigma_update(url, check=True, download=True):
  print('checking for updates...')

  if sys.version_info[0] == 2:
    # Python 2.x
    from urllib2 import urlopen, URLError
  elif sys.version_info[0] > 2:
    # Python 3.x
    from urllib.request import urlopen, URLError

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
    print("\ndownload complete")
  elif __version__ < v:
    print("enigma.py is NOT up to date")
  else:
    print("enigma.py is up to date")


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


  python enigma.py -u[cd]

    The enigma.py module can be used to check for updates. Running
    with the -u flag will check if there is a new version of the
    module available (requires a function internet connection), and if
    there is it will download it.

    If the module can be updated you will see something like this:

      % python enigma.py -u                          
      [enigma.py version 2013-09-10 (Python {python})]
      checking for updates...
      latest version is {version}
      downloading latest version to "{version}-enigma.py"
      ........
      download complete

    Note that the updated version is downloaded to a file named
    "<version>-enigma.py" in the current directory. You can then
    upgrade by renaming this file to "enigma.py".

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
        -t[v] = run tests [v = verbose]
        -u[cd] = check for updates [c = only check, d = always download]
        -h = this help

  Solvers that support the command_line() class method can be invoked
  directly from the command line like this:

  python enigma.py <class> <args> ...

    Supported solvers are: SubstitutedSum, SubstitutedDivision.

      For example, Enigma 327 can be solved using:

      % python enigma.py SubstitutedSum 'KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE'
      [solving KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE ...]
      A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5 / 1912803 + 2428850 + 4312835 = 8654488

      Enigma 440 can be solved using:
      
      % python enigma.py SubstitutedDivision '????? ?x ??x' '?? ?' '' '??x #'
      [solving ????? / ?x = ??x, [('??', '?'), None, ('??x', '')] ...]
      10176 / 96 = 106 rem 0 [x=6] [101 - 96 = 5, 576 - 576 = 0]

""".format(version=__version__, python='2.7.11', python3='3.5.1')

if __name__ == "__main__":

  # allow solvers to run from the command line: python enigma.py <class> <args> ...
  if len(sys.argv) > 1:
    (cmd, args, r) = (sys.argv[1], sys.argv[2:], -1)
    if cmd[0] != '-':
      try:
        r = vars()[cmd].command_line(args)
      except AttributeError:
        printf("{cmd}: command_line() not implemented")
      sys.exit(r)
    
  # identify the version number
  #print('[python version ' + sys.version.replace("\n", " ") + ']')
  printf('[enigma.py version {__version__} (Python {v[0]}.{v[1]}.{v[2]})]', v=sys.version_info)

  # parse arguments
  args = dict((arg[1], arg[2:]) for arg in sys.argv[1:] if len(arg) > 1 and arg[0] == '-')

  # help
  if 'h' in args:
    print('command line arguments:')
    print('  <class> <args> = run command_line(<args>) method on class')
    print('  -t[v] = run tests [v = verbose]')
    print('  -u[cd] = check for updates [c = only check, d = always download]')
    print('  -h = this help')

  # run tests
  if 't' in args:
    import doctest
    doctest.testmod(verbose=('v' in args['t']))

  # -u => check for updates, and download newer version
  # -uc => just check for updates (don't download)
  # -ud => always download latest version
  if 'u' in args:
    url='http://www.magwag.plus.com/jim/'
    try:
      _enigma_update(url, check=('c' in args['u']), download=('d' in args['u']))
    except IOError as e:
      print(e)
      printf("failed to download update from {url}")
