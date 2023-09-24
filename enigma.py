#! python3
###############################################################################
#
# File:         enigma.py
# RCS:          $Header: $
# Description:  Useful routines for solving Enigma Puzzles
# Author:       Jim Randell
# Created:      Mon Jul 27 14:15:02 2009
# Modified:     Sun Sep 24 08:34:13 2023 (Jim Randell) jim.randell@gmail.com
# Language:     Python (Python 2.7, Python 3.6 - 3.12)
# Package:      N/A
# Status:       Free for non-commercial use
# URI:          http://www.magwag.plus.com/jim/enigma.html
#
# (c) Copyright 2009-2023, Jim Randell, all rights reserved.
#
###############################################################################
# -*- mode: Python; python-indent-offset: 2; coding: ascii -*-

"""
A collection of useful code for solving New Scientist Enigma (and similar) puzzles.

The latest version is available at <http://www.magwag.plus.com/jim/enigma.html>.

Currently this module provides the following functions and classes:

all_different          - check arguments are pairwise distinct
all_same               - check arguments all have the same value
arg                    - extract an argument from the command line
args                   - extract a list of arguments from the command line
as_int                 - check argument is an integer
base2int               - convert a string in the specified base to an integer
base_digits            - get/set digits used in numerical base conversion
bit_from_positions     - construct an integer by setting bits in specified positions
bit_permutations       - generate bit permutations
bit_positions          - return positions of bits set
C, nCr                 - combinatorial function (nCr)
cache, cached          - decorator for caching functions
catch                  - catch errors in a function call
cb                     - the cube of the argument
cbrt                   - the (real) cube root of a number
ceil                   - generalised ceiling function
chain                  - see: flatten()
choose                 - choose a sequence of values satisfying some functions
chunk                  - go through an iterable in chunks
clock                  - clock arithmetic variant on mod()
clump                  - collect contiguous blocks of the same value
collect                - collect items according to accept/reject criteria
compare                - comparator function
concat                 - concatenate a list of values into a string
contains               - check for contiguous subsequence
coprime_pairs          - generate coprime pairs
cproduct               - cartesian product of a sequence of sequences
cslice                 - cumulative slices of an array
csum                   - cumulative sum
decompose              - construct and call a Decompose() function
diff                   - sequence difference
digit_map              - create a map of digits to corresponding integer values
digrt                  - the digital root of a number
divc                   - ceiling division
divf                   - floor division
div                    - exact division (or None)
divisor                - generate the divisors of a number
divisor_pairs          - generate pairs of divisors of a number
divisors               - the divisors of a number
divisors_pairs         - generate pairs of divisors of a number
divisors_tuples        - generate tuples of divisors of a number
dot                    - vector dot product
drop_factors           - reduce a number by removing factors
dsum                   - digit sum of a number
ediv                   - exact division (or raise an error)
egcd                   - extended gcd
exact_cover            - find exact covers from a collection of subsets
express                - express an amount using specific denominations
factor                 - the prime factorisation of a number
factorial              - factorial function
farey                  - generate Farey sequences of coprime pairs
fcompose               - forward functional composition
fdiv                   - float division
fib                    - generate fibonacci sequences
filter2                - partition an iterator into values that satisfy a predicate, and those that do not
filter_unique          - partition an iterator into values that are unique, and those that are not
find, rfind            - find the index of an object in a sequence
find_max               - find the maximum value of a function
find_min               - find the minimum value of a function
find_value             - find where a function has a specified value
find_zero              - find where a function is zero
first                  - return items from the start of an iterator
flatten                - flatten a list of lists
flattened              - fully flatten a nested structure
floor                  - generalised floor function
format_recurring       - output the result from recurring()
fraction, Fraction     - convert numerator / denominator to lowest terms
gcd                    - greatest common divisor
grid_adjacency         - adjacency matrix for an n x m grid
group                  - collect values of a sequences into groups
hypot                  - calculate hypotenuse
icount                 - count the number of elements of an iterator that satisfy a predicate
implies                - logical implication (p -> q)
int2base               - convert an integer to a string in the specified base
int2bcd                - convert an integer to binary coded decimal
int2roman              - convert an integer to a Roman Numeral
int2words              - convert an integer to equivalent English words
intc                   - ceiling conversion of float to int
interleave             - interleave values from a bunch of iterators
intersect              - find the intersection of a collection of containers
intf                   - floor conversion of float to int
intr                   - round a value to the nearest integer
invmod                 - multiplicative inverse of n modulo m
ipartitions            - partition a sequence with repeated values by index
irange                 - inclusive range iterator
irangef                - inclusive range iterator with fractional steps
iroot                  - integer kth root function
is_coprime             - check two numbers are coprime
is_cube, is_cube_z     - check a number is a perfect cube
is_distinct            - check a value is distinct from other values
is_duplicate           - check to see if value (as a string) contains duplicate characters
is_pairwise_distinct   - check all arguments are distinct
is_palindrome          - check a sequence is palindromic
is_power               - check if n = i^k for some integer i
is_power_of            - check if n = k^i for some integer i
is_prime               - simple prime test
is_prime_mr            - Miller-Rabin fast prime test for large numbers
is_roman               - check a Roman Numeral is valid
is_square, is_square_p - check a number is a perfect square
is_square_free         - check a number is square free
is_triangular          - check a number is a triangular number
isqrt                  - intf(sqrt(x))
join, joinf            - concatenate objects into a string
lcm                    - lowest common multiple
line_bisect            - find the perpendicular bisector of a line
line_distance          - minimum distance from a point to a line
line_intersect         - find the intersection of two lines
M                      - multichoose function (nMk)
map2str                - format a map for output
match                  - match a value against a template
mgcd                   - multiple gcd
mlcm                   - multiple lcm
mod                    - return a function to find residues modulo m
multiply               - the product of numbers in a sequence
nconcat                - concatenate single digits into an integer
ndigits                - number of digits used to represent a number in a base
nreverse               - reverse the digits in an integer
nsplit                 - split an integer into single digits
number                 - create an integer from a string ignoring non-digits
ordered                - return arguments as an ordered tuple
P, nPr                 - permutations function (nPr)
partitions             - partition a sequence of distinct values into tuples
peek                   - return an element of a container
pi                     - float approximation to pi
poly_*                 - routines manipulating polynomials, wrapped as Polynomial
powers                 - generate a range of powers
prime_factor           - generate terms in the prime factorisation of a number
prime_factor_rho       - generate prime factors of large numbers
printf                 - print with interpolated variables
pythagorean_triples    - generate Pythagorean triples
quadratic              - find roots of a quadratic equation
ratio, ratio_q         - find lowest terms integer ratio
rational               - represet a rational number
rcompose               - reverse functional composition
reciprocals            - generate reciprocals that sum to a given fraction
recurring              - decimal representation of fractions
recurring2fraction     - find the fraction corrresponding to a decimal expansion
repdigit               - number consisting of repeated digits
repeat                 - repeatedly apply a function to a value
restrict               - the restriction of a container to certain keys
reverse                - reverse a sequence
roman2int              - convert a Roman Numeral to an integer
rotate                 - rotate a sequence
seq_all_different      - check elements of a sequence are pairwise distinct
seq_all_same           - check elements of a sequence are all the same
singleton              - return the value from a single valued container
split                  - split a value into characters
sprintf                - interpolate variables into a string
sq                     - square of x
sqrt, sqrtc, sqrtf     - the (positive) square root of a number
static                 - decorator to simulate static variables
subfactorial           - subfactorial function
subsets, subseqs       - generate subsequences of an iterator
substitute             - substitute symbols for digits in text
substituted_expression - a substituted expression (alphametic/cryptarithm) solver
substituted_sum        - a solver for substituted sums
sum_of_squares         - decompose an integer into a sum of squares
sumsq                  - calculate the sum of the squares of a sequence of values
tau                    - tau(n) is the number of divisors of n
timed                  - decorator for timing functions
timer                  - a Timer object
translate              - substitute values in text
tri, T                 - tri(n) is the nth triangular number
trim                   - remove elements from the start/end of a sequence
trirt                  - the (positive) triangular root of a number
tuples                 - generate overlapping tuples from a sequence
ulambda                - complex parameter unpacking
union                  - construct the union of a bunch of containers
uniq, uniq1            - unique elements of an iterator
unzip                  - inverse of zip
unpack                 - return a function that unpacks its arguments
update                 - create an updated version of a container object
zip_eq                 - check sequences contain the same elements

Accumulator            - a class for accumulating values
CrossFigure            - a class for solving cross figure puzzles
Decompose              - return a decompose() function
Delay                  - a class for the delayed evaluation of a function
Denominations          - express amounts using specified denominations
DominoGrid             - a class for solving domino grid puzzles
Football               - a class for solving football league table puzzles
MagicSquare            - a class for solving magic squares
Matrix                 - a class for manipulation 2d matrices
multiset               - an implementation of multisets (bags)
Polynomial             - a class for manipulating polynomials
Primes                 - a class for creating prime sieves
Rational               - select an implementation for rational numbers
SubstitutedDivision    - a class for solving substituted long division sums
SubstitutedExpression  - a class for solving general substituted expression (alphametic/cryptarithm) problems
SubstitutedSum         - a class for solving substituted addition sums
Timer                  - a class for measuring elapsed timings
"""

# Python 3 style print() and division
from __future__ import (print_function, division)

__author__ = "Jim Randell <jim.randell@gmail.com>"
__version__ = "2023-09-11"

__credits__ = """Brian Gladman, contributor"""

import sys
import os

import operator
import math
import functools
import itertools
import collections
import re

# (see also the "six" module)
_pythonv = sys.version_info[0:2]  # Python version e.g. (2, 7) or (3, 11)
if _pythonv[0] == 2:
  # Python 2.x
  _python = 2
  if _pythonv[1] < 7:
    print("[enigma.py] WARNING: Python {v} is very old. Things may not work.".format(v=sys.version.split(None, 1)[0]))
  xrange = xrange
  reduce = reduce
  basestring = basestring
  raw_input = raw_input
  Sequence = collections.Sequence
  Iterable = collections.Iterable
elif _pythonv[0] > 2:
  # Python 3.x
  _python = 3
  xrange = range
  reduce = functools.reduce
  basestring = str
  raw_input = input
  if _pythonv > (3, 6):
    # Python 3.7 onwards
    # not: [[ Sequence = collections.abc.Sequence ]]
    from collections.abc import (Sequence, Iterable)
  else:
    Sequence = collections.Sequence
    Iterable = collections.Iterable

_builtin_min = min
_builtin_max = max
_builtin_sum = sum

# call(<function>, <sequence of args>, [<dict of keywords>]) is an alternative to apply()
call = lambda fn, args=(), kw=dict(): fn(*args, **kw)

# re-exported functions from standard library (to save on imports)
defaultdict = collections.defaultdict
namedtuple = collections.namedtuple
product = itertools.product  # cartesian product, but see also: cproduct()

# detect if running under PyPy
_pypy = getattr(sys, 'pypy_version_info', None)

# useful constants
enigma = sys.modules[__name__]
nl = "\n"
pi = math.pi
two_pi = pi + pi
inf = float('+inf')
empty = frozenset()  # the empty set

str_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # upper case letters
str_lower = "abcdefghijklmnopqrstuvwxyz"  # lower case letters
str_digit = "0123456789"  # decimal digits
str_nl = nl

_PY_ENIGMA = os.getenv("PY_ENIGMA") or ''

version = lambda: __version__

# import a value from a qualified spec, e.g.:
#   Q = import_fn('fractions.Fraction')
#   Q = import_fn('gmpy2.mpq')
#   Q = import_fn('mpmath.rational.mpq')
#   urlopen = import_fn('urllib2.urlopen')  # Python 2
#   urlopen = import_fn('urllib.request.urlopen')  # Python 3
def import_fn(spec):
  # we could use importlib.import_module() here
  importer = lambda x: __import__(x, fromlist=[''])
  if '.' not in spec: return importer(spec)
  (mod, fn) = spec.rsplit('.', 1)
  return getattr(importer(mod), fn)

# lazy importer
# will import on call _and_ attempt to replace <name> in <space> with the imported function
def lazy_import(spec, **kw):
  name = (kw['name'] if 'name' in kw else str.rsplit(spec, '.', 1)[-1])
  space = (kw['space'] if 'space' in kw else sys._getframe(1).f_globals) # replace in the defining space
  def _f(*args, **kw):
    fn = _f.fn
    if not fn:
      #printf("lazy_import: importing {spec}")
      # cache the function
      fn = _f.fn = import_fn(spec)
      if name and space:
        # attempt to replace the name in appropriate namespace
        try:
          space[name] = fn
        except Exception:
          pass
    # call the function
    return fn(*args, **kw)
  _f.fn = None
  return _f

# add attributes to a function (to use as static variables)
# (but for better performance use global variables, or mutable default parameters)
def static(**kw):
  """
  simulates static variables in a function by adding attributes to it.

  static variable <v> in function <f> is accessed as <f.v>.

  e.g.:

    @static(n=0)
    def gensym(x):
      gensym.n += 1
      return concat(x, gensym.n)

    >> gensym('foo')
    'foo1'
    >> gensym('bar')
    'bar2'
    >> gensym('baz')
    'baz3'

  (for better performance you can use global variables)
  """
  def _inner(fn):
    for (k, v) in kw.items():
      setattr(fn, k, v)
    return fn
  return _inner

# useful as a decorator for caching functions (@cached).
# TODO: add a parameter for adding a 'key' function
def cached(f):
  """
  return a cached version of function <f>.

  the cache can be accessed as attribute 'cache' on function <f>.

  cache() is also available which will use Python's own function
  (functools.cache), if available, otherwise cached().

  see also: functools.lru_cache() (Python 3.2), functools.cache() (Python 3.9).
  """
  f.cache = cache = dict()
  @functools.wraps(f)
  def _inner(*k):
    try:
      #if k in cache: printf("[{f.__name__}: cache hit, {k}")
      return cache[k]
    except KeyError:
      r = cache[k] = f(*k)
      #printf("[{f.__name__}: {k} -> {r}]")
      return r
  return _inner

# or you can use cache, to get functools.cache() (if available) or cached() if not.
cache = getattr(functools, 'cache', cached)

# wrap a function in another function, e.g. @wrap(uniq, verbose=1)
def wrap(_fn, *args, **kw):
  """
  a decorator that allows a function to be wrapped in another function.

  for example:

  >>> @wrap(uniq)
  ... def sqmod(n, m):
  ...   for i in irange(1, n):
  ...     yield sq(i) % m

  will only provide values the first time they are encountered:

  >>> list(sqmod(10, 10))
  [1, 4, 9, 6, 5, 0]
  """
  def _inner(_f):
    @functools.wraps(_f)
    def __inner(*fargs, **fkw): return _fn(_f(*fargs, **fkw), *args, **kw)
    return __inner
  return _inner

# the identity function
def identity(x):
  """the identity function: identity(x) == x"""
  return x

# a function that returns a true value
def true(*args, **kw):
  # type: (...) -> bool
  """a function that ignores any arguments and returns True"""
  return True

# a function that returns its arguments
#def tupl(*args, fn=None): return (args if fn is None else fn(args))

# can we treat x as an integer?
# include = +/-/0, check for +ve, -ve, 0
def as_int(x, include="", **kw):
  # type: (...) -> int
  """
  can argument <x> be treated as an integer?

  <include> can be used to restrict the allowed range, by specifying
  one or more of:
    + = allow positive integers
    - = allow negative integers
    0 = allow zero

  <default> can be specified as a value returned instead of raising an error

  so things like this work:

    as_int(0)  -->  0
    as_int(42)  -->  42
    as_int(42.0)  -->  42
    as_int(Fraction(129, 3))  -->  43
    as_int(sympy.Integer(42))  -->  42
    as_int(sympy.Float(42.0))  -->  42
    as_int(sympy.Rational(129, 3))  -->  43

  and things like this raise an error:

    as_int("42")
    as_int(42.5)
    as_int(Fraction(129, 2))
    as_int(42+0j)
    as_int(42, include="-")
    as_int(0, include="+")
  """
  try:
    n = int(x)
    if x == n:
      if include:
        if n > 0:
          if '+' in include:
            return n
        elif n < 0:
          if '-' in include:
            return n
        else:
          if '0' in include:
            return n
      else:
        return n
    return kw['default']
  except Exception:
    pass
  msg = "invalid integer: " + repr(x)
  if include: msg += ' [include: ' + include + ']'
  raise ValueError(msg)

# division that always produces a float
def fdiv(a, b, fn=float):
  """
  float result of <a> divided by <b>.

  >>> fdiv(3, 2)
  1.5

  >>> fdiv(9, 3)
  3.0
  """
  return fn(a) / fn(b)

# useful routines for solving Enigma puzzles

# less than/greater than (or equal) to a target; useful for filter() etc.
def lt(t): return (lambda x: x < t)
def le(t): return (lambda x: x <= t)
def gt(t): return (lambda x: x > t)
def ge(t): return (lambda x: x >= t)
def between(a, b): return (lambda x: a < x < b)  # exclusive between
def betweene(a, b): return (lambda x: a <= x <= b)  # inclusive between

# membership/non-membership of a collection
def isin(s): return (lambda x: x in s)
def isnotin(s): return (lambda x: x not in s)

# return a function that increments by a fixed amount
def inc(i=1): return (lambda x, i=i: x + i)

def mod(m):
  """
  return a function to compute residues modulo <m>.

  >>> list(map(mod(2), irange(0, 9)))
  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  """
  return (lambda n: n % m)

def clock(m):
  """
  like mod(m) except instead of 0 the function returns <m>.

  >>> list(map(clock(12), irange(0, 23)))
  [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  """
  return (lambda n: (n % m) or m)

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

# sign of a number (-1, 0, +1)
sign = lambda x: (0 < x) - (x < 0)  # = compare(x, 0)

# negation
neg = lambda x: -x

# logical implication: p -> q
def implies(p, q):
  """
  logical implication: (p -> q) = ((not p) or q)

  >>> list(p for p in irange(1, 100) if not implies(is_prime(p), p % 6 in (1, 5)))
  [2, 3]
  """
  return (not p) or q

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

# <seq> has <n> distinct values
def distinct_values(seq, n=None):
  seq = list(seq)
  if n is None: n = len(seq)
  return len(set(seq)) == n

# this is the same as: seq_multiplicity(seq, n=1)
def seq_all_different(*seqs, **kw):
  """
  check all elements of <seq> are pairwise distinct.

  if multiple sequences are provided elements must be
  distinct across all sequences.

  >>> seq_all_different([0, 1, 2, 3])
  True
  >>> seq_all_different([2, 1, 2, 3])
  False
  >>> seq_all_different(p % 100 for p in primes)
  False
  """
  fn = kw.pop('fn', None)
  if kw: raise TypeError(str.format("seq_all_different: unknown arguments {kw}", kw=seq2str(kw.keys())))
  seen = set()
  for seq in seqs:
    for x in seq:
      if fn: x = fn(x)
      if x in seen: return False
      seen.add(x)
  return True

# generalised version of seq_all_different()
def seq_multiplicity(*seqs, **kw):
  """
  check all elements of <seq> have multiplicity <n>.

  if multiple sequences are provided then the total
  multiplicity across all sequences is considered.

  >>> seq_multiplicity([0, 1, 2, 3], n=1)
  True
  >>> seq_multiplicity([2, 1, 2, 3], [0, 1, 0, 3], n=2)
  True
  """
  n = kw.pop('n', 1)
  fn = kw.pop('fn', None)
  if kw: raise TypeError(str.format("seq_multiplicity: unknown arguments {kw}", kw=seq2str(kw.keys())))
  seen = multiset()
  for seq in seqs:
    for x in seq:
      if fn: x = fn(x)
      if seen.get(x, 0) == n: return False
      seen.add(x)
  return seen.all_same(n)

# same as distinct_values(args), or distinct_values(args, len(args))
def is_pairwise_distinct(*args, **kw):
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
  #for i in xrange(len(args) - 1):
  #  if args[i] in args[i + 1:]: return False
  #return True
  return seq_all_different(args, **kw)

pairwise_distinct = is_pairwise_distinct
all_different = is_pairwise_distinct

# returns a Record() with various information on sequence <seq>
def seq_all_same_r(seq, **kw):
  """
  check to see if a sequence consists of values that are all the same
  (testing using equality (==)).

  if a 'value' parameter is passed, elements are checked to see if
  it is the same as this value, otherwise the elements should be the
  same as each other.

  a Record() is returned with the following attributes:

    same - true if all values in the sequence have the same value
    value - the value of all elements (or None)
    empty - if the sequence was empty

  if the sequence has no elements, a 'value' of None is returned, and
  'empty' is set to true.

  if the sequence contains different values a 'value' of None is
  returned.

  if the sequence has fewer than 2 elements, 'same' is trivially true.
  """
  i = iter(seq)
  n = 0
  # is value specified?
  if 'value' in kw:
    v = kw['value']
  else:
    # otherwise, use the first value
    try:
      v = next(i)
      n = 1
    except StopIteration:
      # empty sequence
      return Record(same=True, empty=True, value=None)
  # check the rest of the sequence
  for x in i:
    if x != v:
      return Record(same=False, empty=False, value=None)
    n = 1
  return Record(same=True, empty=(n == 0), value=v)

def seq_all_same(seq, **kw):
  """
  >>> seq_all_same([1, 2, 3])
  False
  >>> seq_all_same([1, 1, 1, 1, 1, 1])
  True
  >>> seq_all_same([1, 1, 1, 1, 1, 1], value=4)
  False
  >>> seq_all_same(Primes(expandable=1))
  False
  """
  return seq_all_same_r(seq, **kw).same

# same as distinct_values(args, 1)
def all_same(*args, **kw):
  """
  check all arguments have the same value

  >>> all_same(1, 2, 3)
  False

  >>> all_same(1, 1, 1, 1, 1, 1)
  True

  >>> all_same()
  True
  """
  return seq_all_same_r(args, **kw).same

def zip_eq(*ss, **kw):
  """
  check sequences have the same elements.

  the 'strict' argument is passsed to zip (which supported in some Python
  versions, and throws an error if the inputs are not of equal length)

  the 'first' parameter limits checks for the first <k> elements

  if 'reverse' is set the comparison starts from end

  >>> zip_eq((1, 2, 3), [1, 2, 3])
  True
  >>> zip_eq((1, 2, 3), [1, 2, 3], irange(1, 3), map(int, "123"))
  True
  >>> zip_eq((1, 2, 3, 4), [1, 2, 4, 8])
  False
  >>> zip_eq((1, 2, 3, 4), [1, 2, 4, 8], first=2)
  True
  >>> zip_eq((0, 2, 4, 8), [1, 2, 4, 8], reverse=1, first=2)
  True
  """
  reverse = kw.pop('reverse', 0)
  strict = kw.pop('strict', None)
  k = kw.pop('first', None)
  if kw: raise TypeError(str.format("zip_eq: unknown arguments {kw}", kw=seq2str(kw.keys())))
  if reverse: ss = (reversed(s) for s in ss)
  z = (zip(*ss, strict=strict) if strict is not None else zip(*ss))
  if k is not None: z = first(z, count=k, fn=iter)
  for vs in z:
    if not seq_all_same(vs): return False
  return True

def ordered(*args, **kw):
  """
  return args as a tuple in order.

  this is useful for making a key for a dictionary.

  >>> ordered(2, 1, 3)
  (1, 2, 3)
  >>> ordered(2, 1, 3, reverse=1)
  (3, 2, 1)
  >>> ordered(42)
  (42,)
  """
  return tuple(sorted(args, **kw))

# is sequence <seq> sorted?
def is_sorted(seq, strict=0, fn=operator.lt):
  f = (fn if strict else lambda x, y: x == y or fn(x, y))
  return all(f(x, y) for (x, y) in tuples(seq, 2))

is_increasing = lambda seq, strict=0: is_sorted(seq, strict=strict, fn=operator.lt)
is_decreasing = lambda seq, strict=0: is_sorted(seq, strict=strict, fn=operator.gt)

def encl(s, b="{}", fn=str):
  """
  enclose string <s> by bracketing it with the elements of <b>.

  <s> is initially processed by <fn>.

  >>> encl('xyz')
  '{xyz}'
  >>> encl(42, b='[]')
  '[42]'
  >>> encl('xyz', '|')
  '|xyz|'
  """
  return b[0] + fn(s) + b[-1]

# I would prefer join() to be a string constructor:
#   str.from_seq(seq, sep='', enc=''), or just: str.join(seq, sep='', enc='')
# but for now we define a utility function
def join(seq, sep='', enc='', fn=str):
  """
  construct a string by joining the items in sequence <seq> as
  strings, separated by separator <sep>, and enclosed by the pair
  <enc>.

  the default separator is the empty string so you can just use:

    join(seq)

  instead of:

    ''.join(seq)

  >>> join(['a', 'b', 'cd'])
  'abcd'
  >>> join(['a', 'b', 'cd'], sep=',', enc='{}')
  '{a,b,cd}'
  >>> join([5, 700, 5])
  '57005'

  """
  r = str.join(sep, (fn(x) for x in seq))
  if enc: r = str.join('', (enc[0], r, enc[-1]))
  return r

def joinf(sep='', enc='', fn=str):
  "return a joining function"
  return (lambda x: join(x, sep=sep, enc=enc, fn=fn))

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
  sep = kw.pop('sep', '')
  enc = kw.pop('enc', '')
  if kw: raise TypeError(str.format("concat: unknown arguments {kw}", kw=seq2str(kw.keys())))
  if len(args) == 1:
    try:
      return join(args[0], sep=sep, enc=enc)
    except TypeError:
      pass
    except:
      raise
  return join(args, sep=sep, enc=enc)

# reverse a sequence or a map (maybe -> rev())
# for a unicode string if will be considered as a sequence of codepoints
# you might want to convert it to a sequence of graphemes instead
def reverse(s, fn=None):
  """
  return the reverse of a sequence (str, tuple, list) or map (dict).

  note: when reversing a map, data may be lost if the original map
  does not have distinct values.

  >>> reverse([1, 2, 3])
  [3, 2, 1]
  >>> reverse(first(primes, 6))
  [13, 11, 7, 5, 3, 2]
  >>> reverse("stratagem")
  'megatarts'
  >>> reverse(dict(a=1, b=2, c=3))
  {1: 'a', 2: 'b', 3: 'c'}

  """
  ## if it has a 'rev' attribute call that (note: list.reverse() modifies the list)
  #if hasattr(s, 'rev'): return (s.rev() if fn is None else fn(s.rev()))
  # if it is a dict, return a reverse map
  if isinstance(s, dict): return type(s)((v, k) for (k, v) in s.items())
  # if it is not already a sequence, turn it into one
  if not isinstance(s, Sequence): s = list(s)
  # if it is a string, return a string
  if fn is None:
    if isinstance(s, basestring):
      fn = join
    elif isinstance(s, tuple):
      fn = tuple
    else:
      fn = list
  return fn(reversed(s))

rev = reverse

# translate text <t>, using map <m> (and optional symbols <s>)
def translate(t, m, s="", embed=1):
  """
  translate the text in <t> according to map <m> (and symbols <s>)

  <t> is a string (sequence of letters), if there are sections of the
  string enclosed in curly braces only those sections will be
  translated, otherwise the whole string is processed (providing the
  'embed' parameter is not disabled).

  <m> can be:
    - a dict of <letter> -> <replacement> mappings
    - a sequence of letters to replace, in which case <s> should
      give the corresponding substitutions
    - a function called for each replacement, that should provide
      the replacement value

  substitutions can be multiple letters

  >>> translate("A={A} B={B} C={C}", dict(A=1, B=2, C=3))
  'A=1 B=2 C=3'
  >>> translate("9567 + 1085 = 10652", "75160892", "DEMNORSY")
  'SEND + MORE = MONEY'
  >>> translate("1->{1}; 2->{2}; 3->{3}", (lambda x: sq(int(x))))
  '1->1; 2->4; 3->9'
  """
  t = str(t)
  # construct the map
  if callable(m):
    f = m
  else:
    if not isinstance(m, dict): m = dict(zip(m, s))
    f = (lambda x: m.get(x, x))
  fn = (lambda t: join(map(f, t)))
  if (not embed) or ('{' not in t): return fn(t)
  return re.sub(r'{(.*?)}', (lambda x: fn(x.group(1))), t)


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
  base = kw.pop('base', 10)
  if kw: raise TypeError(str.format("nconcat: unknown arguments {kw}", kw=seq2str(kw.keys())))
  # allow a sequence to be passed as a single argument
  if len(digits) == 1 and isinstance(digits[0], (Sequence, Iterable)): digits = digits[0]
  # this is faster than using: reduce(lambda a, b: a * base + b, digits, 0)
  n = 0
  for d in digits:
    n *= base
    n += d
  return n
  # or: (slower, and only works with digits < base)
  #return int(concat(*digits), base=base)

def nsplitter(n, k=None, base=10, validate=0):
  """split integer <n> into digits, starting with the least significant digit"""
  if base < 2: raise ValueError(str.format("invalid base: {base!r}", base=base))
  if validate: (n, base) = (as_int(n), as_int(base))
  n = abs(n)
  if k is None:
    # the "natural" number of digits in n
    while True:
      (n, r) = divmod(n, base)
      yield r
      if n == 0: break
  else:
    # the least significant k digits of n
    for _ in irange(1, k):
      (n, r) = divmod(n, base)
      yield r

def nsplit(n, k=None, base=10, fn=tuple, reverse=0, validate=0):
  """split an integer into digits (using base <base> representation)

  if <k> is specified it gives the number of digits to return, if the
  number has too few digits the the result is zero padded at the beginning,
  if the number has too many digits then the result includes only the
  rightmost digits.

  the sign of the integer is ignored.

  if <reverse> is set to a true value then the digits are presented in
  reverse order.

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
  >>> nsplit(111**2, 3)
  (3, 2, 1)

  """
  ds = nsplitter(n, k=k, base=base, validate=validate)
  if reverse: return fn(ds)
  return fn(list(ds)[::-1])

def dsum(n, k=None, base=10, validate=0):
  """
  calculate the digit sum of an integer (when represented in the
  specified base).

  the sign of the integer is ignored

  >>> dsum(123456789)
  45
  >>> dsum(123456789, base=2)
  16
  """
  return sum(nsplitter(n, k=k, base=base, validate=validate))

# population count, Hamming weight, bitsum(), bit_count()
if getattr(int, "bit_count", None):
  dsum2 = int.bit_count
else:
  def dsum2(n): "fast alternative to dsum(n, base=2)"; return bin(abs(n)).count('1', 2)

# equivalent to: len(nsplit(n))
# (we could use logarithms for "smallish" numbers)
def ndigits(n, base=10, validate=0):
  """
  return the number of digits in a number, when represented in the specified base.

  >>> ndigits(factorial(70))
  101
  """
  #return sum(1 for _ in nsplitter(n, base=base, validate=validate))
  return icount(nsplitter(n, base=base, validate=validate))

# maybe -> nrev()
def nreverse(n, k=None, base=10, validate=0):
  """
  reverse an integer (as a <k> digit number using base <base> representation)

  >>> nreverse(12345)
  54321
  >>> nreverse(-12345)
  -54321
  >>> nreverse(0xedacaf, base=16) == 0xfacade
  True
  >>> nreverse(100)
  1
  >>> nreverse(1, 3)
  100
  """
  if n < 0:
    return -nreverse(-n, base=base, validate=validate)
  else:
    return nconcat(nsplitter(n, k=k, base=base, validate=validate), base=base)

nrev = nreverse


fnmatch = lazy_import("fnmatch.fnmatchcase", name='fnmatch')

# match a value (as a string) to a template
# NOTE: match is a soft keyword in Python 3.10+
def match(v, t):
  """
  match a value (as a string) to a template (see fnmatch.fnmatchcase).

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

@static(special={'inf': inf, '+inf': inf, '-inf': -inf})
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
  v = number.special.get(s.strip().lower(), None)
  if v: return v
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

# rotate a sequence (move k elements from the beginning to the end)
def rotate(s, k=1):
  """
  rotate a sequence by moving <k> elements from the beginning to the end

  >>> rotate([1, 2, 3, 4], 1)
  [2, 3, 4, 1]
  >>> rotate([1, 2, 3, 4], -1)
  [4, 1, 2, 3]
  """
  return (s if k == 0 else s[k:] + s[:k])

# or you can use itertools.izip_longest(*[iter(l)]*n) for padded chunks
def chunk(seq, n=2, pad=0, value=None, fn=tuple):
  """
  iterate through iterable <seq> in chunks of size <n>.

  (for overlapping tuples see tuples())

  >>> list(chunk(irange(1, 8)))
  [(1, 2), (3, 4), (5, 6), (7, 8)]
  >>> list(chunk(irange(1, 8), 3))
  [(1, 2, 3), (4, 5, 6), (7, 8)]
  """
  i = iter(seq)
  while True:
    s = fn(itertools.islice(i, 0, n))
    if not s: break
    x = (n - len(s) if pad else 0)
    yield (s if x == 0 else s + fn([value] * x))

# find contiguous blocks of values (according to fn)
def clump(seq, fn=None):
  """
  generate (<value>, <count>) pairs for contiguous blocks of repeated values
  in sequence <seq> (according to function <fn>).

  >>> list(clump([1, 1, 1, 2, 2, 3]))
  [[1, 1, 1], [2, 2], [3]]
  >>> list(clump("bookkeeper"))
  [['b'], ['o', 'o'], ['k', 'k'], ['e', 'e'], ['p'], ['e'], ['r']]
  >>> list(clump(map(tri, irange(1, 10)), fn=mod(2)))
  [[1, 3], [6, 10], [15, 21], [28, 36], [45, 55]]
  """
  seq = iter(seq)
  try:
    v = next(seq)
  except StopIteration:
    return
  xs = [v]
  if fn is not None: v = fn(v)
  for x in seq:
    v_ = (x if fn is None else fn(x))
    if v_ == v:
      xs.append(x)
    else:
      yield xs
      v = v_
      xs = [x]
  yield xs

# set union of a bunch of sequences
def union(ss, fn=set):
  """construct a set that is the union of the sequences in <ss>"""
  return fn().union(*ss)

# disjoint set union of a bunch of sequences (or None)
# any value may appear in only one of the sequences
def disjoint_union(ss, fn=set):
  """
  construct a set that is the union of the sequences in <ss>.

  each value in the returned set only appears in one of the sequences
  (although it may appear multiple times in that sequence).

  if this is not possible a value of None is returned.

  >>> disjoint_union([[1], [2], [3], [4]]) == {1, 2, 3, 4}
  True
  >>> disjoint_union([[1], [2], [3], [2]]) is None
  True
  """
  rs = None
  for xs in ss:
    if not rs:
      rs = fn(xs)
    else:
      # this seems to be faster than updating rs and checking cardinality
      for x in xs:
        if x in rs: return
        rs.add(x)
  return rs

# set intersection of a bunch of sequences
def intersect(ss, fn=set):
  """construct a set that is the intersection of the sequences in <ss>"""
  i = iter(ss)
  try:
    s = fn(next(i))
  except StopIteration:
    pass
  else:
    return s.intersection(*i)
  raise ValueError("empty intersection")

# return an element of a container
def peek(s, k=0, **kw):
  """
  return an element of a container.

  empty containers return the specified 'default' value, or raise a
  ValueError.

  if k is specified the first k values chosen are discarded
  (so, for a sequence, you will get s[k]).

  note that if the container is an iterator, items will be consumed.

  >>> peek(set([1]))
  1
  >>> peek([1, 2, 3])
  1
  >>> peek("banana")
  'b'
  >>> peek("banana", 4)
  'n'
  >>> peek(primes, 10)
  31
  >>> peek(p for p in primes if p % 17 == 1)
  103
  """
  if not isinstance(s, dict):
    # try to index into the container
    try:
      return s[k]
    except (KeyError, IndexError, TypeError):
      pass
  # try iterating through the container
  for (i, x) in enumerate(s):
    if i == k:
      return x
  try:
    return kw['default']
  except KeyError:
    pass
  raise ValueError(str.format("invalid index {k}", k=k))

# functions to create a selector for elements/attributes from an object
# passing multi=1 forces a multivalued return, even if only one element is specified
def item(*ks, **kw):
  multi = kw.pop('multi', 0)
  if kw: raise TypeError(str.format("item: unknown arguments {kw}", kw=seq2str(kw.keys())))
  f = operator.itemgetter(*ks)
  if len(ks) == 1 and multi: return (lambda x: (f(x),))
  return f

def attr(*ks, **kw):
  multi = kw.pop('multi', 0)
  if kw: raise TypeError(str.format("attr: unknown arguments {kw}", kw=seq2str(kw.keys())))
  f = operator.attrgetter(*ks)
  if len(ks) == 1 and multi: return (lambda x: (f(x),))
  return f

items = lambda n: map(item, xrange(n)) # (x, y, z) = items(3)

# select items according to space/comma separated template
# item_from("p", "V, L, p") -> item(2)
def item_from(select, template, **kw):
  split = lambda s: (re.split(r'[\s,]+', s.strip("()[]{}")) if isinstance(s, basestring) else s)
  fields = dict((k, v) for (v, k) in enumerate(split(template)))
  return item(*(fields[k] for k in split(select)), **kw)

def diff(a, b, *rest, **kw):
  """
  return the subsequence of <a> that excludes elements in <b>.

  >>> diff((1, 2, 3, 4, 5), (3, 5, 2))
  (1, 4)
  >>> join(diff('newhampshire', 'wham'))
  'nepsire'
  """
  fn = kw.pop('fn', tuple)
  if kw: raise TypeError(str.format("diff: unknown arguments {kw}", kw=seq2str(kw.keys())))
  if rest: b = set(b).union(*rest)
  return fn(x for x in a if x not in b)


# unique combinations:
# like uniq(combinations(s, k)) but more efficient
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

# the multiset is implemented as a dict mapping <item> -> <count>
class multiset(dict):
  """
  an implementation of multisets.

  it can be used as an alternative to collections.Counter(), but note
  the following differences:

    len() counts the number of elements (not the number of distinct elements)

    iterating through a multiset provides all elements (not just distinct
    elements)
  """

  def __init__(self, *vs, **kw):
    """
    create a multiset from one of the following:

      a dict of <item> -> <count> values

      a sequence of (<item>, <count>) values

      a sequence of individual items (may have repeats)

    multiple initialisation arguments may be provided, the
    items from each are added into the multiset.

    so these are different ways of making the same multiset:

      multiset("banana")
      multiset(a=3, b=1, n=2)
      multiset([('a', 3), ('b', 1), ('n', 2)])
      multiset(['b', 'a', 'n', 'a', 'n', 'a'])
      multiset(dict(a=3, b=1, n=2))

    for more control over the initialisation of the multiset you can
    use: from_dict(), from_pairs(), from_seq() class methods or the
    corresponding: update_from_dict(), update_from_pairs(),
    update_from_seq() object methods.
    """
    dict.__init__(self)
    # deal with any initialisation objects
    for v in vs:
      if isinstance(v, dict):
        # from a dict
        self.update_from_dict(v)
      else:
        # from a sequence
        v = list(v)
        try:
          s = multiset().update_from_pairs(v)
          self.update_from_dict(s)
        except (TypeError, ValueError): # maybe more, or maybe just Error
          self.update_from_seq(v)
    # add in any keyword items
    if kw:
      for (x, n) in kw.items():
        self.add(x, n)

  def update_from_seq(self, vs, count=1):
    """update a multiset from a sequence of items"""
    for x in vs:
      self.add(x, count=count)
    return self

  def update_from_pairs(self, vs):
    """update a multiset from a sequence of (<item>, <count>) pairs"""
    for (x, n) in vs:
      self.add(x, as_int(n))
    return self

  def update_from_dict(self, d):
    """update a multiset from a dict of <item> -> <count> values"""
    return self.update_from_pairs(d.items())

  @classmethod
  def from_dict(cls, *vs):
    """
    create a multiset from a dict of <item> -> <count> values
    (or multiple dicts).
    """
    m = multiset()
    for v in vs:
      m.update_from_dict(v)
    return m

  @classmethod
  def from_pairs(self, *vs):
    """
    create a multiset from a sequence of (<item>, <count>) pairs
    (or multiple sequences).
    """
    m = multiset()
    for v in vs:
      m.update_from_pairs(v)
    return m

  @classmethod
  def from_seq(self, *vs, **kw):
    """
    create a multiset from a sequence of items (or multiple sequences).

    A keyword argument of 'count' specifies the multiplicity of each
    element of the sequence inserted into the multiset.
    """
    count = kw.pop('count', 1)
    if kw: raise TypeError(str.format("multiset.from_seq: unknown arguments {kw}", kw=seq2str(kw.keys())))
    m = multiset()
    for v in vs:
      m.update_from_seq(v, count=count)
    return m

  # count all elements in the multiset
  def size(self):
    """
    the cardinality of the multiset.
    i.e. a count all the elements in a multiset.

    to count the number of distinct element types use: s.distinct_size().

    this function is used to implement the len() method on multisets.

    >>> multiset("banana").size()
    6
    >>> len(multiset("banana"))
    6
    >>> multiset("banana").distinct_size()
    3
    """
    return sum(dict.values(self))

  # len(multiset) == multiset.size()
  __len__ = size

  def is_empty(self):
    return dict.__len__(self) == 0

  def is_nonempty(self):
    return dict.__len__(self) > 0

  # is_nonempty is faster than using __len__
  __bool__ = __nonzero__ = is_nonempty

  # return an element
  def peek(self, k=0, **kw):
    return peek(self, k=k, **kw)

  def is_duplicate(self, n=1):
    "does this multiset contain elements with multiplicity greater than <n>?"
    return any(v > n for v in dict.values(self))

  def all_same(self, n=None):
    "does this multiset contain only values with the same multiplicity (<n> if specified)"
    if n is None:
      return seq_all_same(dict.values(self))
    else:
      return all(v == n for v in dict.values(self))

  # all elements of the multiset
  # (for unique elements use: [[ s.keys() ]])
  def elements(self):
    """
    iterate through all elements of the multiset.

    for distinct elements use: s.keys()

    this function is used if a multiset is used as an iterator.

    >>> sorted(multiset("banana"))
    ['a', 'a', 'a', 'b', 'n', 'n']
    >>> sorted(multiset("banana").keys())
    ['a', 'b', 'n']
    """
    for (k, v) in dict.items(self):
      for _ in xrange(v):
        yield k

  __iter__ = elements

  # the distinct elements
  # alias for keys()
  distinct_elements = dict.keys

  # the number of the distinct elements
  def __distinct_size(self):
    """the number of distinct elements in the multiset"""
    return len(dict.keys(self))

  # specialised version for PyPy3
  def __distinct_size_pypy3(self):
    """the number of distinct elements in the multiset"""
    return len(dict(self))

  distinct_size = (__distinct_size_pypy3 if (_pypy and _python > 2) else __distinct_size)

  # return a count of the item
  def count(self, item):
    """return the number of times an item occurs in the multiset"""
    return dict.get(self, item, 0)

  # add an item
  def add(self, item, count=1):
    """
    add an item to a multiset.

    count can be negative to remove items.
    """
    try:
      count += self[item]
      if count == 0:
        del self[item]
        return self
    except KeyError:
      pass
    if count < 0: raise ValueError(str.format("negative count: {item} -> {count}", item=item, count=count))
    if count > 0: self[item] = count
    return self

  # remove an item
  def remove(self, item, count=1):
    """remove an item from the multiset"""
    return self.add(item, -count)

  # like self.items(), but in value order
  def most_common(self, n=None):
    """
    return the items of the multiset in order of the most common.

    if n is specifed only the first n items are returned.
    """
    s = sorted(dict.items(self), key=(lambda t: t[::-1]), reverse=1)
    return (s if n is None else s[:n])

  # provide some useful operations on multisets

  # update self with some other multisets (item counts are summed)
  def update(self, *rest):
    """
    update the multiset with some other multisets (or objects that can
    be interpreted as multisets).

    item counts are summed.
    """
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      self.update_from_dict(m)
    return self

  # combine self and some other multisets (item counts are summed)
  def combine(self, *rest):
    """
    return a new multiset that is the result of the original multiset
    updated with some other multisets (or objects that can be
    interepreted as multisets).

    item counts are summed.
    """
    return multiset(self).update(*rest)

  # union update of self and some other multiset (maximal item counts are retained)
  def union_update(self, *rest):
    """
    update a multiset with the union of itself and some other
    multisets (or objects that can be interpreted as multisets).

    maximal item counts are retained.
    """
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      for (item, count) in m.items(): self[item] = max(count, self.get(item, 0))
    return self

  # union of self and some other multiset (maximal item counts are retained)
  def union(self, *rest):
    """
    return a new multiset that is the result of the union of the
    original multiset and some other multisets (or objects that can be
    interpreted as multisets).

    maximal item counts are retained.
    """
    return multiset(self).union_update(*rest)

  # intersection of self and some other multisets (minimal item counts are retained)
  def intersection(self, *rest):
    """
    return a new multiset that is the result of the intersection of
    the original multiset and some other multisets (or objects that
    can be interpreted as multisets).

    minimal item counts are retained.
    """
    r = multiset(self)
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      r = multiset.from_pairs((item, min(count, r.get(item, 0))) for (item, count) in m.items())
    return r

  # is this multiset a subset of m?
  def issubset(self, m, strict=0):
    """test if the multiset is contained in multiset <m>"""
    if not isinstance(m, dict): m = multiset(m)
    # check sizes
    r = compare(self.size(), m.size())
    if r == 1 or (strict and r == 0): return False
    # check items
    for (item, count) in self.items():
      if count > m.get(item, 0): return False
    return True

  # is this multiset m a superset of self?
  def issuperset(self, m, strict=0):
    """test if the multiset contains multiset <m>"""
    if not isinstance(m, dict): m = multiset(m)
    return m.issubset(self, strict=strict)

  # differences between self and m
  # return (self - m, m - self)
  def differences(self, m):
    """
    return the differences between self and another multiset m.

    returns (self - m, m - self)
    """
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
    """return (self - m)"""
    return self.differences(m)[0]

  # absolute difference in item counts of the two multisets
  def symmetric_difference(self, m):
    """
    symmetric difference of this multiset with multiset <m>.

    the difference in item counts is retained.
    """
    (d1, d2) = self.differences(m)
    return d1.update(d2)

  def is_disjoint(self, *rest):
    """test if the multiset is disjoint from a bunch of other multisets"""
    for m in rest:
      if not isinstance(m, dict): m = multiset(m)
      if any(x in self for x in m): return False
    return True

  # multiply item counts
  def multiply(self, n):
    """
    return a new mutliset derived from the original multiset by
    multiplying item counts by <n>.
    """
    return multiset.from_pairs((k, n * v) for (k, v) in dict.items(self))

  def subsets(self, size=None, min_size=0, max_size=None):
    """generate subsets of a multiset"""
    if size is not None:
      min_size = max_size = size
    elif max_size is None:
      max_size = len(self)
    es = list(self.elements())
    for k in irange(min_size, max_size):
      for ss in uC(es, k):
        yield multiset.from_seq(ss)

  def copy(self):
    """return a copy of the multiset"""
    return multiset.from_dict(self)

  def min(self, **kw):
    """
    return the minimum item value of a multiset (or <default>).

    equivalent to: min(self)
    """
    if (not self) and 'default' in kw: return kw['default']
    if 'key' in kw:
      return _builtin_min(dict.keys(self), key=kw['key'])
    else:
      return _builtin_min(dict.keys(self))

  def max(self, **kw):
    """
    return the maximum item value of a multiset (or <default>).

    equivalent to: max(self)
    """
    if (not self) and 'default' in kw: return kw['default']
    if 'key' in kw:
      return _builtin_max(dict.keys(self), key=kw['key'])
    else:
      return _builtin_max(dict.keys(self))

  def sum(self, fn=_builtin_sum):
    """
    return the sum of items in a multiset.

    equivalent to: sum(self)
    """
    return fn(v * k for (k, v) in dict.items(self))

  def avg(self, div=fdiv, fn=_builtin_sum):
    """
    return the average (arithmetic mean) of the items in a multiset.
    """
    return div(self.sum(fn=fn), self.size())

  def multinomial(self):
    """
    The number of different arrangements of this multiset.

    >>> multiset("banana").multinomial()
    60
    """
    return multinomial(self.values())

  # generate elements in order
  def sorted(self, key=None, reverse=False):
    for k in sorted(dict.keys(self), key=key, reverse=reverse):
      for _ in xrange(self.get(k)):
        yield k

  # restriction of a multiset to a specific set of keys
  def restrict(self, ks, strict=0):
    return restrict(self, ks, strict=strict)

  def map2str(self, sort=1, enc='()', sep=', ', arr='='):
    """call map2str() on the multiset"""
    return map2str(self, sort=sort, enc=enc, sep=sep, arr=arr)

  # generate item pairs
  def to_pairs(self):
    return tuple(sorted(dict.items(self)))

  def to_dict(self):
    return dict(self)

  # allow operator overloading on multisets
  # (let me know if these don't do what you expect)
  # + = update
  # - = difference
  # & = intersection
  # | = union
  # * = multiply
  # < = subset
  # > = superset
  __add__ = combine
  __sub__ = difference
  __and__ = intersection
  __or__ = union
  __iadd__ = update
  __ior__ = union_update
  __mul__ = multiply
  __le__ = issubset
  __ge__ = issuperset
  __lt__ = lambda self, m: self.issubset(m, strict=1)
  __gt__ = lambda self, m: self.issuperset(m, strict=1)

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
#  [timing] total time: 68.9407666s (68.94s)
#
#  >>> with Timer(): icount(subsets("mississippi", select="mP"))
#  107899
#  [timing] total time: 0.5661372s (566.14ms)
#
def mP(d, n, r=()):
  if n == 0:
    yield r
  else:
    for (k, v) in d.items():
      if v > 0:
        d[k] -= 1
        for t in mP(d, n - 1, r + (k,)): yield t
        d[k] += 1

def mpermutations(s, k=None):
  s = multiset(s)
  if k is None: k = len(s)
  return mP(s, k)

# a simple implementation of derangements
def derangements(s, k=None):
  s = list(s)
  if k is None: k = len(s)
  for p in itertools.permutations(s, k):
    if not any(x == s[i] for (i, x) in enumerate(p)):
      yield p

# can be cached() if necessary
def subfactorial(n):
  if n == 0: return 1
  return n * subfactorial(n - 1) + (-1 if n % 2 else 1)

# subsets (or subseqs) wraps various methods (which can save an import)
@static(select_fn=dict(), prepare_fn=dict())
def subsets(s, size=None, min_size=0, max_size=None, select='C', prepare=None, fn=tuple):
  """
  generate tuples representing the subsequences of a (finite) iterator.

  'min_size' and 'max_size' can be used to limit the size of the
  subsequences or 'size' can be specified to produce subsequences of a
  particular size.

  the way the elements of the subsequences are selected can be
  controlled with the 'select' parameter:
     'C' = combinations (default)
     'P' = permutations
     'D' = derangements
     'R' = combinations with replacement
     'M' = product
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
    for x in select(s, k): yield fn(x)

# provide selection functions (where available)
# [[ maybe 'R' should be 'M', and 'M' should be 'X' ]]
def _subsets_init():
  for (k, v, p) in (
      ('C', getattr(itertools, 'combinations', None), None),
      ('P', getattr(itertools, 'permutations', None), None),
      ('D', derangements, None),
      ('R', getattr(itertools, 'combinations_with_replacement', None), None),
      ('M', (lambda s, k: product(s, repeat=k)), None),
      ('uC', uC, None),
      ('mC', uC, (lambda s: sorted(multiset(s)))),
      ('mP', mP, multiset),
    ):
    if v:
      subsets.select_fn[k] = v
      setattr(subsets, k, v)
      if p: subsets.prepare_fn[k] = p

_subsets_init()

# aliases
powerset = subsets
subseqs = subsets

# like filter() but also returns the elements that don't satisfy the predicate
# see also partition() recipe from itertools documentation
# (but note that itertools.partition() returns (false, true) lists)
@static(rtype=None)
def filter2(p, i, fn=list):
  """
  use a predicate to partition an iterable into those elements that
  satisfy the predicate, and those that do not.

  returns the partition of the original sequence as:

    (<values for which p is true>, <values for which p is false>)

  which can also be accessed from return value r as:

    r.true
    r.false

  alias: partition()

  >>> tuple(filter2(lambda n: n % 2 == 0, irange(1, 10)))
  ([2, 4, 6, 8, 10], [1, 3, 5, 7, 9])
  """
  t = list((x, p(x)) for x in i)
  if filter2.rtype is None: filter2.rtype = namedtuple('Filter2', 'true false')
  return filter2.rtype(fn(x for (x, v) in t if v), fn(x for (x, v) in t if not v))

# alias if you prefer the term partition (but don't confuse it with partitions())
partition = filter2

def is_equal(x, y):
  """
  is_equal(x, y) is the same as (x == y)

  >>> is_equal(1, 2)
  False
  >>> is_equal(42, 42)
  True
  >>> is_equal([1, 2, 3], (1, 2, 3))
  False
  """
  return (x == y)

@static(rtype=None)
def filter_unique(seq, f=identity, g=identity, st=None):
  """
  for objects <x> in the sequence <seq> consider the map f(x) -> g(x)
  and return a partition of <seq> into those objects where f(x)
  implies a unique value for g(x), and those objects where f(x)
  implies multiple values for g(x).

  if the predicate <st> is specified, only objects from the sequence
  that satisfy the predicate are considered.

  returns the partition of the original sequence as:

    (<unique values>, <non-unique values>)

  which can also be accessed from return value r as:

    r.unique
    r.non_unique

  See: Enigma 265 <https://enigmaticcode.wordpress.com/2015/03/14/enigma-265-the-parable-of-the-wise-fool/#comment-4167>

  alias: partition_unique()

  "If I told you the first number you could deduce the second"
  >>> filter_unique([(1, 1), (1, 3), (2, 2), (3, 1), (3, 2), (3, 3)], (lambda v: v[0])).unique
  [(2, 2)]

  "If I told you the first number you could not deduce if the second was odd or even"
  >>> filter_unique([(1, 1), (1, 3), (2, 2), (3, 1), (3, 2), (3, 3)], (lambda v: v[0]), (lambda v: v[1] % 2)).non_unique
  [(3, 1), (3, 2), (3, 3)]

  """
  # group values by f
  r = group(seq, st=st, by=f)
  # collect unique/non-unique items
  (unq, non) = (list(), list())
  if g is identity:
    # special case if g is not specified
    for (k, vs) in r.items():
      (unq if seq_all_same(vs) else non).extend(vs)
  else:
    # general case
    for (k, vs) in r.items():
      (unq if seq_all_same(map(g, vs)) else non).extend(vs)
  #printf("unq = {unq}\nnon = {non}")
  if filter_unique.rtype is None: filter_unique.rtype = namedtuple('FilterUnique', 'unique non_unique')
  return filter_unique.rtype(unq, non)

# alias if you prefer the term partition (but don't confuse it with partitions())
partition_unique = filter_unique


def _collect(s, accept, reject, every):
  for x in s:
    if (accept is None or accept(x)) and (reject is None or not reject(x)):
      yield x
    elif every:
      raise ValueError(str.format("collect: failed to collect item: {x}", x=x))

def collect(s, accept=None, reject=None, every=0, fn=list):
  """
  collect items from sequence <s> that are accepted by the <accept>
  function (if defined), and not rejected by the <reject> function (if
  defined).

  return the items that pass the tests (using <fn>)

  if every=1 then every item must be collected, otherwise None is
  returned.
  """
  try:
    return fn(_collect(s, accept, reject, every))
  except ValueError:
    return None

def group(seq, by=identity, st=None, f=identity, fn=None):
  """
  group the items of sequence <seq> together using the <by> function.

  items in the same group return the same value when passed to <by>.

  if the <st> function is specified, only items that satisfy it will
  be considered.

  if the <f> function is specified, the function is applied to
  selected values before they are added to the groups.

  a dict() is returned where the keys of the dict are the values of
  the <by> function applied to the items of the sequence, and the
  values of the dict are the grouped items (collected using <fn>,
  which by default will collect the items in a list, in the order of
  the original sequence <s>).

  >>> group(irange(0, 9), by=mod(2))
  {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}

  # to reverse a dict into a multivalued map
  >>> d = dict((n, n % 2) for n in irange(0, 9))
  >>> group(d.items(), by=item(1), f=item(0), fn=sorted)
  {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
  """
  d = dict()
  for x in seq:
    if st is None or st(x):
      k = by(x)
      v = f(x)
      vs = d.get(k)
      if vs is None:
        d[k] = [v]
      else:
        vs.append(v)
  if fn:
    for (k, vs) in d.items():
      d[k] = fn(vs)
  return d

# see ulambda() for a workaround for more complicated unpacking
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
  return (lambda args, kw=None: (fn(*args, **kw) if kw else fn(*args)))

# unpacked form of zip (which also serves as an inverse to zip)
unzip = unpack(zip)

# cartesian product of a sequence, cproduct = unpack(itertools.product)
def cproduct(ss, **kw):
  """
  the cartesian product of a sequence.

  so:
    itertools.product(*(<generator>)) --> cproduct(<generator>)
    itertools.product(As, Bs, Cs) --> cproduct([As, Bs, Cs])

  >>> set(cproduct(chunk(irange(1, 4), 2))) == {(1, 3), (1, 4), (2, 3), (2, 4)}
  True
  """
  return itertools.product(*ss, **kw)  # = call(itertools.product, ss, kw)

# here's workaround for more complicated parameter unpacking in Python 3
#
# in Python 2.7 we could do:
#
#   fn = lambda (x, (y, z)): x + y + z
#
# instead we can do this:
#
#   fn = ulambda("(x, (y, z))", "x + y + z")
#
#   fn = ulambda("(x, (y, z)): x + y + z")
#
def ulambda(args, expr=None):
  """
  provide an equivalent to the Python 2 expression:

    lambda {args}: {expr}

  in Python 3

  where {args} specifies a complex parameter unpacking of arguments

  e.g.:

  >>> dist = ulambda("(x1, y1), (x2, y2): hypot(x2 - x1, y2 - y1)")
  >>> dist((1, 2), (5, 5))
  5.0
  """
  if expr is None:
    (args, _, expr) = (x.strip() for x in args.partition(":"))

  if _python == 2:
    # in Python 2 it is straightforward lambda
    expr = str.format("lambda {args}: {expr}", args=args, expr=expr)
  else:
    # in Python 3 any of the following achieve the same
    #expr = sprintf("lambda *_x_: [{expr} for [{args}] in [_x_]][0]")
    #expr = sprintf("lambda *_x_: peek({expr} for [{args}] in [_x_])")
    expr = str.format("lambda *_x_: next({expr} for [{args}] in [_x_])", expr=expr, args=args)
  return eval(expr)

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
      p = true
  n = 0
  for x in i:
    if p(x):
      n += 1
      if n == t: break
  return n

# icount recipes
icount_exactly = lambda i, p=None, n=None: icount(i, p, n + 1) == n
icount_at_least = lambda i, p=None, n=None: icount(i, p, n) == n
icount_at_most = lambda i, p=None, n=None: icount(i, p, n + 1) < n + 1

# find: like index(), but return -1 instead of throwing an error
def find(seq, v):
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
  >>> find(primes, 10007)
  1229

  Note that this function works by attempting to use the index() method
  of the sequence. If it implements index() in a non-compatible way
  this function won't work.
  """
  try:
    return seq.index(v)
  except ValueError:
    return -1
  except AttributeError:
    pass
  if isinstance(seq, dict):
    # search the keys
    # (or we could use find() in the values, and return the correspond index in keys)
    for (k, x) in s.items():
      if x == v: return k
  else:
    # search the sequence
    for (i, x) in enumerate(seq):
      if x == v: return i
  # not found
  return -1

def rfind(seq, v):
  """find the last index of a value in a sequence, return -1 if not found"""
  i = find(seq[::-1], v)
  return (-1 if i == -1 else len(s) - i - 1)

# trim elements from a sequence
def trim(seq, head=0, tail=0, fn=None):
  """
  return a new sequence derived from input sequence <s>, but with <head>
  elements removed from the front and <tail> elements removed from the
  end.

  >>> trim([1, 2, 3, 4, 5], head=2)
  [3, 4, 5]
  >>> trim([1, 2, 3, 4, 5], tail=2)
  [1, 2, 3]
  >>> trim([1, 2, 3, 4, 5], head=2, tail=2)
  [3]
  >>> trim('progress', head=2, tail=2)
  'ogre'
  """
  if head > 0 or tail > 0:
    if fn is None:
      if isinstance(seq, basestring):
        fn = join
      elif isinstance(seq, tuple):
        fn = tuple
    seq = list(seq)
    if head > 0: del seq[:head]
    if tail > 0: del seq[-tail:]
  return (fn(seq) if fn else seq)

def _partitions(seq, n):
  """
  partition a sequence <seq> of distinct elements into subsequences of length <n>.

  <seq> should be sequenceable type (tuple, list, str).
  <n> should be a factor of the size of the sequence.

  >>> list(_partitions((1, 2, 3, 4), 2))
  [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
  """
  if not (len(seq) > n):
    yield (seq,)
  else:
    for x in itertools.combinations(seq[1:], n - 1):
      p = (seq[0],) + tuple(x)
      for ps in _partitions(diff(seq[1:], x), n):
        yield (p,) + ps


def ipartitions(seq, n):
  """
  partition a sequence by index.

  >>> list(ipartitions((1, 0, 1, 0), 2))
  [((1, 0), (1, 0)), ((1, 1), (0, 0)), ((1, 0), (0, 1))]
  """
  for p in _partitions(tuple(xrange(len(seq))), n):
    yield tuple(tuple(seq[i] for i in x) for x in p)


def partitions(seq, n, pad=0, value=None, distinct=None):
  """
  partition a sequence <seq> into subsequences of length <n>.

  if <pad> is true then the sequence will be padded (using <value>)
  until its length is a integer multiple of <n>.

  if sequence <seq> contains distinct elements then <distinct> can be
  set to True, if it is not set then <seq> will be examined for repeated
  elements.

  >>> list(partitions((1, 2, 3, 4), 2))
  [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]
  """
  if not isinstance(seq, (tuple, list, str)): seq = tuple(seq)
  (d, r) = divmod(len(seq), n)
  if r != 0:
    if not pad: raise ValueError("invalid sequence length {l} for {n}-tuples".format(l=len(seq), n=n))
    seq = tuple(seq) + (value,) * (n - r)
  if d == 0 or (d == 1 and r == 0):
    yield (seq,)
  else:
    if distinct is None: distinct = is_pairwise_distinct(*seq)
    fn = (_partitions if distinct else ipartitions)
    # or in Python 3: [[ yield from fn(seq, n) ]]
    for z in fn(seq, n): yield z


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
      if (not distinct) or v not in s:
        s_ = list(s)
        s_.append(v)
        if fn is None or fn(*s_):
          # choose the rest [[Python 3: yield from ...]]
          for z in choose(vs, fns[1:], s_, distinct): yield z


def first(s, count=1, skip=0, fn=list):
  """
  return the first <count> items in iterator <s> (skipping the initial
  <skip> items) as a list (or other object specified by <fn>).

  <count> can be a callable object, in which case items are collected
  from <i> while <count> returns a true value when it is passed each
  item (after skipping the first <skip> items).

  <skip> can also be a callable, in which case items are skipped while
  <skip> returns a true value when it is passed each item.

  this would be a way to find the first 10 primes:
  >>> first(primes, count=10)
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  >>> first(p for p in primes if p % 17 == 1)
  [103]

  this finds squares less than 200
  >>> first(powers(0, inf, 2), count=lt(200))
  [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196]
  """
  if callable(count):
    if skip == 0:
      r = itertools.takewhile(count, s)
    elif callable(skip):
      r = itertools.takewhile(count, itertools.dropwhile(skip, s))
    else:
      r = itertools.takewhile(count, itertools.islice(s, skip, None))
  elif count == inf:
    r = s
  else:
    r = itertools.islice(s, skip, skip + count)
  return (r if fn is None else fn(r))

# return the single value if s contains only a single value (else None)
# NOTE: similar to the Python expression : [[ [x] = s ]]
def singleton(s, skip=0, default=None):
  """
  if the container <s> contains only a single value return it,
  otherwise return None (or the <default> parameter)

  >>> singleton([], default=0)
  0
  >>> singleton({1}, default=0)
  1
  >>> singleton([1, 2, 3], default=0)
  0
  """
  r = first(s, 2, skip)
  return (r[0] if len(r) == 1 else default)

def repeat(fn, v=0, k=inf):
  """
  generate repeated applications of function <fn> to value <v>.

  the initial value is returned first, followed by the result of
  repeatedly applying the specified function to the previous value.

  if a limit <k> is specified then the function will be applied
  the specified number of times, so (k + 1) values will be returned
  (corresponding to the application of the function 0 .. k times).

  >>> list(repeat(inc(1), 0, 10))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  """
  i = 0
  while True:
    yield v
    if i == k: break
    i += 1
    v = fn(v)

def uniq(seq, fn=None, verbose=0):
  """
  generate unique values from <seq> (maintaining order).

  i.e. repeated values are suppressed.

  >>> list(uniq([5, 7, 0, 0, 5]))
  [5, 7, 0]
  >>> join(uniq('mississippi'))
  'misp'
  >>> list(uniq(irange(1, 9), fn=(lambda x: x // 3)))
  [1, 3, 6, 9]
  """
  seen = set()
  for x in seq:
    r = (x if fn is None else fn(x))
    if r not in seen:
      yield x
      seen.add(r)
  if verbose: printf("[uniq: found {n} unique items]", n=len(seen))

def uniq1(seq, fn=None):
  """
  collapse repeated consecutive values (according to <fn>) in <seq>
  down to single values.

  i.e. repeated _consecutive_ values are suppressed.

  >>> list(uniq1((1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5)))
  [1, 2, 3, 4, 5]
  >>> join(uniq1('mississippi'))
  'misisipi'
  >>> join(uniq1('bookkeeper'))
  'bokeper'
  """
  for vs in clump(seq, fn=fn):
    yield vs[0]


# root: calculate the (positive) nth root of a (positive) number
# we use math.pow rather than **/pow() to avoid generating complex numbers
root = lambda x, n: (x if not x else math.pow(x, 1.0 / n))

def _cbrt(x):
  """
  return the cube root of a number (as a float).

  see also: math.cbrt() (Python 3.11)

  >>> cbrt(27.0)
  3.0
  >>> cbrt(-27.0)
  -3.0
  """
  r = root(abs(x), 3.0)
  return (-r if x < 0 else r)

# use math.cbrt() [available from 3.11]
cbrt = getattr(math, 'cbrt', _cbrt)

# cb = lambda x: x**3
def cb(x): "cb(x) = x**3"; return x**3

# for large numbers with large prime factors use prime_factor_h() or sympy.ntheory.factorint()
# basis = [2, 3, 5]
_prime_factor_ds = (1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6)  # deltas
_prime_factor_js = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3)  # next index
def prime_factor(n, limit=inf):
  """
  generate (<prime>, <exponent>) pairs in the prime factorisation of
  positive integer <n>.

  no pairs are returned for 1 (or for non-positive integers).

  if 'limit' is specified no prime factors greater than this will
  be returned.

  for numbers with large prime factors it will take a long time to
  find them. in this case you probably want to use prime_factor_h()
  instead.

  >>> list(prime_factor(60))
  [(2, 2), (3, 1), (5, 1)]
  >>> list(prime_factor(factorial(12)))
  [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]
  >>> list(prime_factor(factorial(12) + 1))
  [(13, 2), (2834329, 1)]
  """
  if n > 1:
    ds = _prime_factor_ds
    js = _prime_factor_js
    x = 2
    j = 0
    while True:
      if x > limit: return
      e = 0
      while True:
        (d, r) = divmod(n, x)
        if r > 0: break
        e += 1
        n = d
      if e > 0: yield (x, e)
      x += ds[j]
      if x * x > n: break
      j = js[j]
    if n > limit: return
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

# divsors (for non-negative integers) are based on:
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

# is <x> a divisor of <n>?
def is_divisor(n, x, proper=0):
  """
  determine if <x> is a divisor of <n> (both are non-negative integers).

  if 'proper' is set then the divisor <x> must be smaller than <n>.

  >>> is_divisor(42, 7)
  True
  >>> is_divisor(43, 7)
  False
  >>> is_divisor(7, 7)
  True
  >>> is_divisor(7, 7, proper=1)
  False
  >>> is_divisor(1, 0)
  False
  >>> is_divisor(0, 0)
  True
  """
  if x < 0 or x > n: return False
  if n == 0: return (not proper) and (x == 0)
  return (x > 0 and n % x == 0 and ((not proper) or x < n))

# you can use the following to look for multiples but note argument order is opposite of div()
# and this will only work for multiples from 1 upwards (or from 2 if 'proper' is set)
#is_multiple = lambda n, x, proper=0: is_divisor(x, n, proper=proper)

def divisor_pairs(n):
  """
  generate divisors (a, b) of positive integer n, such that a <= b and a * b = n.

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


def multiples(ps, k=1):
  """
  given a list of (<m>, <n>) pairs, return all numbers that can be formed by multiplying
  together the <m>s, with each <m> occurring up to <n> * <k> times.

  the multiples are returned as a sorted list

  the practical upshot of this is that the divisors of a number <x> can be found using
  the expression: multiples(prime_factor(x))

  >>> multiples(prime_factor(180))
  [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180]
  """
  s = [1]
  for (m, n) in ps:
    if k > 1: n *= k
    t = list()
    p = m
    for _ in xrange(n):
      t.extend(x * p for x in s)
      p *= m
    s.extend(t)
  s.sort()
  return s


def divisors(n, k=1, fn=prime_factor, validate=0):
  """
  return the divisors of positive integer pow(<n>, <k>) as a sorted list.

  the function used to find prime factors can be specified as <fn>. if you
  are factorising large numbers with large prime factors, then you will
  probably want to provide a function based on prime_factor_h().

  >>> divisors(36)
  [1, 2, 3, 4, 6, 9, 12, 18, 36]
  >>> divisors(101)
  [1, 101]
  >>> ds = divisors(factorial(23) - 1, fn=(lambda n: prime_factor_h(n, mr=1)))
  >>> print(seq2str(ds))
  (1, 51871, 498390560021687969, 25852016738884976639999)
  """
  if validate: n = as_int(n, include="0+")
  if n == 0 and k > 0: return [0]
  return multiples(fn(n), k=k)


def divisors_pairs(n, k=1, fn=prime_factor, every=0, validate=0):
  """
  generate divisors pairs (a, b) with a <= b, such that a * b = pow(n, k).

  pairs are generated in order, by determining the factors of n.

  this is probably faster than divisor_pairs() if you want all divisors.

  if the 'every' parameter is set, then pairs with a > b are also generated.
  """
  if validate: n = as_int(n, "0+")
  if n == 0 and k > 0:
    yield (0, 0)
    return
  nk = (n**k if k != 1 else n)
  for a in divisors(n, k=k, fn=fn):
    b = nk // a
    if a > b and (not every): break
    yield (a, b)

def divisors_tuples(n, k, s=()):
  """
  find ordered <k>-tuples that multiply to give <n>.

  >>> list(divisors_tuples(1335, 3))
  [(1, 1, 1335), (1, 3, 445), (1, 5, 267), (1, 15, 89), (3, 5, 89)]
  """
  if k == 1:
    if not (s and n < s[-1]):
      yield s + (n,)
  else:
    for (a, b) in divisors_pairs(n):
      if not (s and a < s[-1]):
        for z in divisors_tuples(b, k - 1, s + (a,)): yield z

# see also: is_prime_mr(), Primes.is_prime(), gmpy2.is_prime()
def is_prime(n, validate=0):
  # type: (int) -> bool
  """
  return True if the non-negative integer <n> is prime.

  if <validate> the argument will be validated as a non-negative integer.

  note: for numbers up to 2**64 is_prime_mr() is a fast, accurate prime
  test. (And for larger numbers it is probabilistically accurate).

  >>> is_prime(101)
  True
  >>> is_prime(1001)
  False
  """
  if validate: n = as_int(n, include="0+")
  if n < 2: return False  # 0, 1 -> F
  if n < 4: return True   # 2, 3 -> T
  r = n % 6
  if r != 1 and r != 5: return False  # (n % 6) != (1, 5) -> F

  for (p, e) in prime_factor(n):
    return p == n
  return False

prime = is_prime


# Miller-Rabin primality test (originally suggested by Brian Gladman)

randrange = lazy_import('random.randrange')

def _is_composite(a, d, n, s):
  if a == 0: return 0
  x = pow(a, d, n)
  if x == 1: return 0
  for _ in xrange(s):
    if x == n - 1: return 0
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

  for numbers less than 2**64, the prime test is completely accurate,
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
  if r > 0 and any(_is_composite(randrange(2, n - 1), d, n, s) for _ in range(r)):
    # definitely composite
    return 0

  # otherwise, probably prime
  return 1


# find prime factors using Pollard's Rho method
# [see: https://programmingpraxis.files.wordpress.com/2012/09/primenumbers.pdf ]
def _rho_factor(n, mrr=0):
  if n % 2 == 0: return 2
  c = 1
  while True:
    (t, h, d) = (2, 2, 1)
    while d == 1:
      t = (t * t + c) % n
      h = (h * h + c) % n
      h = (h * h + c) % n
      d = gcd(h - t, n)
    if d == n:
      pass
    elif is_prime_mr(d, mrr):
      return d
    else:
      n = d
    c += 1

# NOTE: factors are not neccessarily returned in order
def prime_factor_rho(n, mrr=0):
  """
  generate (<prime>, <exponent>) pairs in the prime factorisation of
  positive integer <n>.

  note that factors are not necessarily returned in numerical order.

  <mrr> is the number of additional rounds performed in the
  is_prime_mr() test for prime factors.

  >> sorted(prime_factor_rho(factorial(23) + 1))
  [(47, 2), (79, 1), (148139754736864591, 1)]
  """
  while n > 1:
    # check for prime
    if is_prime_mr(n, mrr):
      yield (n, 1)
      break
    # find a factor
    p = _rho_factor(n)
    (e, n) = drop_factors(n, p)
    yield (p, e)


def prime_factor_h(n, ps=None, end=None, nf=0, mr=0, mrr=0):
  """
  find prime factors using various heuristics

    ps = can be a prime sieve to check (end = upper limit on primes)
    nf = number of tolerated failures (after which we switch to heuristics)
    mr = enable Pollard Rho/Miller Rabin (mrr = number of Miller-Rabin rounds)

  Primes found using the sieve will be generated in numerical order.
  Primes found heuristically may not be generated in order.

  Depending on the arguments the factorisation may be incomplete
  (e.g. if a sieve is specified and mr=0, large factors outside the
  sieve may not be found).

  The following example uses the prime sieve <primes> for factors up to
  1000, and then probabalistic tests to find the rest:

  >> list(prime_factor_h(factorial(18) + 1, ps=primes, end=1000, mr=1))
  [(19, 1), (23, 1), (29, 1), (61, 1), (67, 1), (123610951, 1)]

  """
  if ps is None and mr == 0: raise ValueError("no heuristics specified")  # otherwise we won't find anything
  f = 0  # number of failed primes for n
  psi = (None if ps is None else ps.generate(end=end))
  pmax = 0

  # first use primes from the sieve
  while psi is not None and n > 1:

    # is this a prime in the sieve?
    if n < ps.max:
      if f == 0 and ps.is_prime(n):
        yield (n, 1)
        return

    # try the next prime in the sieve
    try:
      p = next(psi)
    except StopIteration:
      # the sieve is exhausted, so n is a product of primes larger than the sieve
      pmax = end or ps.max
      if n < pmax * pmax:
        # n must be a prime
        yield (n, 1)
        return
      else:
        # move to probabalistic tests
        psi = None
    else:
      # check prime p
      (e, n) = drop_factors(n, p)
      if e > 0:
        yield (p, e)
        f = 0
      else:
        f += 1
        if f == nf:
          # move to probabalistic tests
          pmax = end or ps.max
          psi = None

  # now try heuristic tests on what is left
  if mr and n > 1:
    m = 1  # multiplicity of factors

    # check to see if the number is an exact power
    k = 2
    while True:
      r = iroot(n, k)
      if pow(r, k) == n:
        m *= k
        n = r
      elif k == 2:
        k = 3
      else:
        k += 2
      # limit the search
      if k > 20 or r < pmax: break

    # n could now be in the sieve
    if ps is not None and n < pmax * pmax:
      yield (n, m)
      return

    # look for probabalistic factors (not necessarily in order)
    for (p, e) in prime_factor_rho(n, mrr=mrr):
      yield (p, e * m)

def tau(n, fn=prime_factor):
  """
  count the number of divisors of a positive integer <n>.

  tau(n) = len(divisors(n))  # (but faster)

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

def mobius(n, fn=prime_factor):
  """
  return the Mobius value for positive integer <n>.

  mobius(n) =  1; if n is square free and has an even number of prime factors
  mobius(n) = -1; if n is square free and has an odd number of prime factors
  mobius(n) =  0; if n is not square free
  """
  if n < 1: return None
  r = 1
  for (p, e) in fn(n):
    if e > 1: return 0
    r = -r
  return r


def farey(n, ends=0):
  """
  generate the Farey sequence F(n) - the sequence of coprime
  pairs (a, b) where 0 < a < b <= n. pairs are generated
  in numerical order when considered as fractions a/b.

  the pairs (0, 1) and (1, 1) usually present at the start
  and end of the sequence are not generated by this function,
  unless 'ends' is set to True.

  >>> list(p for p in farey(20) if sum(p) == 20)
  [(1, 19), (3, 17), (7, 13), (9, 11)]
  """
  if ends: yield (0, 1)
  (a, b, c, d) = (0, 1, 1, n)
  while d > 1:
    k = (n + b) // d
    (a, b, c, d) = (c, d, k * c - a, k * d - b)
    yield (a, b)
  if ends: yield (1, 1)

def coprime_pairs(n=None, order=0):
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
  fn = ((lambda p: p[0] <= n) if n else true)
  if order:
    # use a heap to order the pairs
    from heapq import (heapify, heappush, heappop)
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
    for p in ((b + b - a, b), (a + a + b, a), (b + b + a, b)):
      if fn(p): _push(ps, p)

# Pythagorean Triples:
# see: https://en.wikipedia.org/wiki/Formulas_for_generating_Pythagorean_triples

# generate primitive pythagorean triples (x, y, z) with hypotenuse not exceeding Z
# if Z is None, then triples will be generated indefinitely
# if order is true, then triples will be returned in order
def _pythagorean_primitive(Z=None, order=0):
  fn = (true if Z is None else le(Z))
  if order:
    # use a heap
    from heapq import (heapify, heappush, heappop)
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
      (c3 - b2 + a2, c2 - b + a2, c2 - b2 + a),
      (c3 + b2 - a2, c2 + b - a2, c2 + b2 - a),
      (c3 + b2 + a2, c2 + b + a2, c2 + b2 + a),
    ):
      if fn(z): _push(ts, ((z, x, y) if y < x else (z, y, x)))
    ## alternatively: Brian's (more compact, but slower) formulation
    #t = 2 * (a + b + c)
    #(u, v, w) = (t - 4 * b, t, t - 4 * a)
    #for (z, y, x) in ((u + c, u + b, u - a), (v + c, v - b, v - a), (w + c, w - b, w + a)):
    #  if fn(z): _push(ts, ((z, x, y) if y < x else (z, y, x)))

# generate pythagorean triples (x, y, z) with hypotenuse not exceeding Z
def _pythagorean_all(Z, order=0):
  if order:
    # use a heap to save the multiples
    from heapq import (heapify, heappush, heappop)
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
      yield (x, y, z)
      for k in irange(2, Z // z):
        yield (k * x, k * y, k * z)

# generate pythagorean triples
# n - specifies the maximum hypotenuse allowed
# primitive - if set only primitive triples are generated
# order - if set triples are generated in order
# if primitive is false, then a value for n must be specified
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
    if n is None: raise ValueError("max hypotenuse not specified")
    return _pythagorean_all(n, order)


def fib(*s, **kw):
  """
  generate Fibonacci type sequences (or other recurrence relations)

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
  fn = kw.pop('fn', sum)
  if kw: raise TypeError(str.format("fib: unknown arguments {kw}", kw=seq2str(kw.keys())))
  s = list(s)
  while True:
    s.append(fn(s))
    yield s.pop(0)


# if we don't overflow floats (happens around 2^53) this works...
#   def is_power(n, m):
#     i = int(n**(1.0 / m) + 0.5)
#     return (i if i**m == n else None)
# but here we use a binary search, which should work on arbitrary large integers
#
# NOTE: that this will return 0 if n = 0 and None if n is not a perfect k-th power,
# so [[ power(n, k) ]] will evaluate to True only for positive n
# if you want to allow n to be 0 you should check: [[ power(n, k) is not None ]]
#
def iroot(n, k):
  """
  compute the largest integer x such that pow(x, k) <= n.

  i.e. x is the integer k-th root of n.

  it is the exact root if: pow(x, k) == n
  (which is what is_power() does)
  """
  # binary search
  if n < 0 or k < 1: return
  if n >> k == 0: return int(n > 0)
  a = 1 << ((n.bit_length() - 1) // k)
  b = a << 1
  #assert (a**k <= n and b**k > n)
  # if this assertion fails we need:
  #while not (b**k > n): (a, b) = (b, b << 1)
  while b - a > 1:
    r = (a + b) // 2
    x = r**k
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
  >>> n = (2**60 + 1)
  >>> (is_power(n**2, 2) is not None, is_power(n**2 + 1, 2) is not None)
  (True, False)
  >>> (is_power(n**3, 3) is not None, is_power(n**3 + 1, 3) is not None)
  (True, False)
  """
  r = iroot(n, k)
  if r is None: return None
  return (r if r**k == n else None)


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

# sq = lambda x: x * x
def sq(x): "sq(x) = x**2"; return x * x
def sumsq(xs): "sumsq(xs) = sum(sq(x) for x in xs)"; return sum(x * x for x in xs)

# calculate intf(sqrt(n))
@static(impl=getattr(math, 'isqrt', None))
def isqrt(n):
  # type: (int) -> int | NoneType
  """
  calculate intf(sqrt(n)), for integers n.

  See also: math.isqrt (Python 3.8), gmpy2.isqrt().

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
  if isqrt.impl: return isqrt.impl(n) # use math.isqrt() if available

  # use the math.isqrt algorithm
  c = (n.bit_length() - 1) >> 1
  a = 1
  d = 0
  s = c.bit_length()
  while s:
    s -= 1
    e = d
    d = c >> s
    a = (a << d - e - 1) + (n >> (c << 1) - e - d + 1) // a
  return a - (a * a > n)

# square root floor and ceiling functions
sqrtf = isqrt
sqrtc = lambda x: (isqrt(x) if x < 1 else 1 + isqrt(x - 1))

# it would be more Pythonic to encapsulate is_square in a class with the initialisation
# in __init__, and the actual call in __call__, and then instantiate an object to be
# the is_square() function (i.e. [[ is_square = _is_square_class(80) ]]), but it is
# more efficient (and perhaps more readable) to just use normal variables, although
# if you're using PyPy the class based version is just as fast (if not slightly faster)
# experimentally mod = 80, 48, 72, 32 are good values (24, 16 also work OK)
@static(mod=720, residues=None, cache_enabled=0, cache=dict())
def is_square(n, validate=0):
  # type: (int | NoneType, bool) -> int | NoneType
  """
  check integer <n> is a perfect square.

  if <n> is a perfect square, returns the integer square root.
  if <n> is not a perfect square, returns None.

  if <validate> is set, then the input value will be validated as
  a non-negative integer (and an ValueError thrown if it isn't),
  otherwise the input is assumed to be an integer value, or None.

  results can be cached by setting: is_square.cache_enabled = 1

  >>> is_square(49)
  7
  >>> is_square(50) is not None
  False
  >>> is_square(0)
  0
  """
  if validate: n = as_int(n, include="0+")   # if validate check for non-negative integer
  if n is None or n < 0: return None
  if n < 2: return n
  # early rejection: check <square> mod <some value> against a precomputed cache
  # e.g. <square> mod 80 = 0, 1, 4, 9, 16, 20, 25, 36, 41, 49, 64, 65 (rejects 88% of numbers)
  # mod 720 (= factorial(6)) rejects 93% of candidates
  if not is_square.residues: is_square.residues = set((i * i) % is_square.mod for i in xrange(is_square.mod))
  if (n % is_square.mod) not in is_square.residues: return None
  # otherwise use isqrt and check the result
  z = is_square.cache.get(n)
  if z is None:
    r = isqrt(n)
    z = (r if r * r == n else None)
    if is_square.cache_enabled: is_square.cache[n] = z
  return z

# is <n> the square of a rational number?
def is_square_q(n, F=None):
  if n is None: return None
  if F is None: F = Rational()
  n = F(n)
  p = is_square(n.numerator)
  if p is None: return None
  q = is_square(n.denominator)
  if q is None: return None
  return F(p, q)

def sum_of_squares(n, k=2, min_v=0, sep=0, ss=[]):
  """
  return ordered k-sequences of non-negative integers (a, b, ...) such that:

    n = a**2 + b**2 + ...

  min_v - specifies the minimum allowable value in the returned sequences
  sep - specified the minimum separation between values

  >>> list(sum_of_squares(50, 2))
  [[1, 7], [5, 5]]
  >>> list(sum_of_squares(50, 2, sep=1))
  [[1, 7]]
  >>> list(sum_of_squares(637, 3))
  [[0, 14, 21], [3, 12, 22], [5, 6, 24], [12, 13, 18]]
  """
  if k == 1:
    r = is_square(n)
    if not (r is None or r < min_v):
      yield ss + [r]
  elif k == 2:
    i = isqrt(n)
    j = 0
    while not (i < j):
      r = compare(i * i + j * j, n)
      if r == 0 and not (j < min_v or i - j < sep): yield (ss + [j, i] if ss else [j, i])
      if r != -1: i -= 1
      if r != 1: j += 1
  else:
    for x in irange(min_v, inf):
      x2 = x * x
      if x2 * (k - 1) > n: break
      for z in sum_of_squares(n - x2, k - 1, x + sep, sep, ss + [x]): yield z

# generate powers from a range
def powers(a, b, k=2, step=1, fn=None):
  """
  generate powers pow(n, k) for n in irange(a, b)

  >>> list(powers(1, 10))
  [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
  >>> list(powers(1, 10, 3))
  [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
  """
  for n in irange(a, b, step=step):
    x = n**k
    yield (x if fn is None else fn(x))

# compose functions in order (forward functional composition, "and then")
# so: fcompose(f, g, h)(x) == h(g(f(x)))
def fcompose(f, *gs):
  """
  forward functional composition ("and then")

  fcompose(f, g, h)(x) == h(g(f(x)))

  >>> fcompose(is_square, is_not_none)(49)
  True
  >>> fcompose(is_square, is_not_none)(50)
  False
  """
  # special case for 1 or 2 functions
  n = len(gs)
  if n == 0:
    return f
  if n == 1:
    g = gs[0]
    return (lambda *args, **kw: g(f(*args, **kw)))
  # general case
  def fn(*args, **kw):
    r = f(*args, **kw)
    for g in gs:
      r = g(r)
    return r
  return fn

# compose functions in reverse order (reverse functional composition, "after")
# so: rcompose(f, g, h)(x) = f(g(h(x)))
def rcompose(*fns):
  """
  reverse functional composition

  rcompose(f, g, h)(x) == f(g(h(x)))
  """
  return fcompose(*(reversed(fns)))

is_not_none = (lambda x: x is not None)
is_square_p = (lambda x: is_square(x) is not None) # = fcompose(is_square, is_not_none)

# 819 rejects 95% (other good values: 63 (86%), 117 (87%), 189 (89%), 351 (90%), 504 (91%), 819 (95%))
@static(mod=819, residues=None, cache_enabled=0, cache=dict())
def is_cube(n, validate=0):
  """
  check positive integer <n> is a perfect cube.

  to check for positive/negative values use: is_cube_z().

  results can be cached by setting: is_cube.cache_enabled = 1

  >>> is_cube(27)
  3
  >>> is_cube(49) is not None
  False
  >>> is_cube(0)
  0
  """
  if validate: n = as_int(n, include="0+")
  if n < 0: return None
  if n < 2: return n
  if not is_cube.residues: is_cube.residues = set((i * i * i) % is_cube.mod for i in xrange(is_cube.mod))
  if (n % is_cube.mod) not in is_cube.residues: return None
  z = is_cube.cache.get(n)
  if z is None:
    z = is_power(n, 3)
    if is_cube.cache_enabled: is_cube.cache[n] = z
  return z

is_cube_p = (lambda x: is_cube(x) is not None) # = fcompose(is_cube, is_not_none)

def is_cube_z(n, validate=0):
  """
  check integer <n> is a perfect cube.

  >>> is_cube_z(27)
  3
  >>> is_cube_z(-27)
  -3
  >>> is_cube_z(0)
  0
  """
  if n < 0:
    r = is_cube(-n, validate=validate)
    return (None if r is None else -r)
  else:
    return is_cube(n, validate=validate)

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


def tri(n):
  """
  tri(n) is the nth triangular number.

  tri(n) = n * (n + 1) / 2.

  Note: trif() is available for float arguments.

  >>> tri(1)
  1
  >>> tri(100)
  5050
  """
  return n * (n + 1) // 2

T = tri

# triangular numbers as floats
trif = lambda x: 0.5 * x * (x + 1)


def trirt(x):
  """
  return the triangular root of <x> (as a float)

  >>> trirt(5050)
  100.0
  >>> round(trirt(2), 8)
  1.56155281
  """
  return 0.5 * (math.sqrt(8 * x + 1) - 1.0)

def is_triangular(n):
  """
  check positive integer <n> is a triangular number.

  if <n> is a triangular number, returns integer <k> such that T(k) == n.
  if <n> is not a triangular number, returns None.

  >>> is_triangular(5050)
  100
  >>> is_triangular(49) is not None
  False
  """
  if n % 9 not in {0, 1, 3, 6}: return
  x = is_square(8 * n + 1)
  return (None if x is None else x // 2)

is_triangular_p = (lambda x: is_triangular(x) is not None)

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
  if not (0 <= d < base): raise ValueError(str.format("repdigit: invalid digit: {d!r}", d=d))
  return d * ((base**n) - 1) // (base - 1)

# Python 3.6: ...(*vs, root=math.sqrt)
def hypot(*vs, **kw):
  """
  return hypotenuse of a right angled triangle with shorter sides <a> and <b>.

  hypot(a, b) = sqrt(a^2 + b^2)

  multiple arguments can be specified to return Euclidean distance in
  higher dimensions.

  a keyword argument of 'root' may be specified to provide the
  function used to calculate the root of the sum of the squares.

  See also: math.hypot() (Python 3.8).

  >>> hypot(3, 4)
  5.0
  >>> hypot(3, 4, 12)
  13.0
  >>> hypot(3, 4, root=is_square)
  5
  """
  root = kw.pop('root', math.sqrt)
  if kw: raise TypeError(str.format("hypot: unknown arguments {kw}", kw=seq2str(kw.keys())))
  return root(sum(v * v for v in vs))

# alias for: hypot(..., root=is_square)
ihypot = lambda *vs: hypot(*vs, root=is_square)

# root of combined squares:
# for positive value the square of the value is added
# for negative vales the square of the value is subtracted
# and the square root of the total (or None) is returned
# for example (for positive x, y, z)
# hypot(x, y) = rcs(x, y)
# if x^2 + y^2 = z^2, we have: y = rcs(z, -x); x = rcs(z, -y); z = rcs(x, y)
def rcs(*vs, **kw):
  root = kw.pop('root', math.sqrt)
  if kw: raise TypeError(str.format("rcs: unknown arguments {kw}", kw=seq2str(kw.keys())))
  t = 0
  for v in vs:
    if v < 0:
      t -= v * v
    else:
      t += v * v
  return (None if t < 0 else root(t))

ircs = lambda *vs: rcs(*vs, root=is_square)

# return roots of the form n/d in the appropriate domain
def _roots(domain, include, F, *nds):
  (pos, neg, zero) = (x in include for x in '+-0')
  for (n, d) in nds:
    # calculate the root
    if domain in "CF":
      x = F(n / d)
    elif domain == "Q":
      x = F(n, d)
    elif domain == "Z":
      x = as_int(F(n, d), default=None)
      if x is None: continue
    # return the root
    if domain == "C":
      yield x
    elif x > 0:
      if pos: yield x
    elif x < 0:
      if neg: yield x
    else:
      if zero: yield x

# find roots of a quadratic equation
# domain = "Z" (integer), "Q" (rational), "F" (float), "C" (complex float)
def quadratic(a, b, c, domain="Q", include="+-0", F=None):
  """
  find roots of the equation:

     a.x^2 + b.x + c = 0

  in the specified domain:

    "Z" finds integer solutions
    "Q" finds rational solutions
    "F" finds float solutions
    "C" finds complex solutions

  >>> sorted(quadratic(1, 1, -6, domain="Z"))
  [-3, 2]
  """
  if domain not in "CFQZ": raise ValueError(str.format("quadratic: invalid domain {domain!r}", domain=domain))

  if a == 0:
    # linear equation
    if b == 0: raise ValueError("quadratic: invalid linear equation")
    if F is None: F = (Rational() if domain in 'QZ' else (complex if domain == 'C' else float))
    return _roots(domain, include, F, (-c, b))

  # discriminant
  D = b * b - 4 * a * c
  if D < 0 and domain != "C": return _roots(domain, '', None)

  if domain in "CF":
    F = (complex if domain == 'C' else float)
    d = -2 * a
    if D == 0:
      return _roots(domain, include, F, (b, d))
    else:
      r = D**0.5
      return _roots(domain, include, F, (b + r, d), (b - r, d))

  elif domain in "QZ":
    if F is None: F = Rational()
    r = is_square_q(D, F=F)
    if r is not None:
      d = -2 * a
      if r == 0:
        return _roots(domain, include, F, (b, d))
      else:
        return _roots(domain, include, F, (b + r, d), (b - r, d))

  return _roots(domain, '', None)


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
  r = int(x)
  return (r - 1 if x < r else r)


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
  r = int(x)
  return (r + 1 if x > r else r)

def intr(x):
  """
  round to nearest integer.

  if x is exactly between two integers (i.e. x = n + 0.5) then the
  answer is the integer further away from 0.

  >>> intr(0.0)
  0
  >>> intr(2.5)
  3
  >>> intr(-2.5)
  -3
  """
  if x < 0:
    x = -x
    return -int((x + x + 1) // 2)
  else:
    return int((x + x + 1) // 2)

def divf(a, b):
  """
  floor division. (similar to Python's a // b).
  always returns an int.

  >>> divf(100, 10)
  10
  >>> divf(101, 10)
  10
  >>> divf(101.1, 10)
  10
  >>> divf(4.5, 1)
  4
  """
  return int(a // b)

def floor(x, m=1):
  """
  return largest multiple of <m>, not greater than <x>.
  """
  # [[ m is 1 ]] gives a SyntaxWarning
  if m == 1: return intf(x)
  return m * int(x // m)


def divc(a, b):
  """
  ceiling division.
  always returns an int.

  >>> divc(100, 10)
  10
  >>> divc(101, 10)
  11
  >>> divc(101.1, 10)
  11
  >>> divc(4.5, 1)
  5
  """
  return -int(-a // b)

cdiv = divc

def divr(a, b):
  """
  round the value of a/b to the nearest integer.

    divr(a, b) = intr(fdiv(a, b))

  >>> divr(0, 1)
  0
  >>> divr(5, 2)
  3
  >>> divr(10, -4)
  -3
  """
  if b < 0: (a, b) = (-a, -b)
  if a < 0:
    a = -a
    return -int((a + a + b) // (b + b))
  else:
    return int((a + a + b) // (b + b))

def ceil(x, m=1):
  """
  return lowest multiple of <m>, not less than <x>
  """
  if m == 1: return intc(x)
  return m * -int(-x // m)

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
  if b == 0 or a is None: return None
  (d, r) = divmod(a, b)
  if r != 0: return None
  return d

def ediv(a, b):
  "return (a // b) if b exactly divides a, otherwise raise a ValueError."
  (d, r) = divmod(a, b)
  if r != 0: raise ValueError("inexact division")
  return d

def is_duplicate(*s):
  """
  check to see if arguments (as strings) contain duplicate characters.

  >>> is_duplicate("hello")
  True
  >>> is_duplicate("world")
  False
  >>> is_duplicate(99**2)
  False
  """
  s = join(s)
  return len(set(s)) != len(s)
  # or using regexps
  #return True if re.search(r'(.).*\1', str(s)) else False

duplicate = is_duplicate

# this avoids creating a reversed copy of the sequence
# Note: for unicode strings this will be checking the sequence of codepoints
# you might want to convert the string to a sequence of graphemes first
def is_palindrome(s):
  """
  check to see if sequence <s> is palindromic.

  >>> is_palindrome([1, 2, 3, 2, 1])
  True
  >>> is_palindrome("ABBA")
  True
  >>> first(n for n in irange(0, inf) if not is_palindrome(nsplit(11**n)))
  [5]
  """
  j = len(s)
  if j < 2: return True
  i = 0
  j -= 1
  while i < j:
    if s[i] != s[j]: return False
    i += 1
    j -= 1
  return True

# is a number palindromic in base b
def is_npalindrome(n, base=10):
  """
  check if integer <n> is palindromic in base <base>.

  >>> is_npalindrome(1230321)
  True
  >>> is_npalindrome(10**5000 + 1)
  True
  >>> is_npalindrome(57005, base=9)
  True
  """
  n = abs(n)
  if n % base == 0: return (n == 0)
  (a, b) = (n, 0)
  while a > b:
    (d, r) = divmod(a, base)
    b *= base
    b += r
    if a == b: return True
    a = d
  return (a == b)

# originally called product(), but renamed to avoid name confusion with itertools.product()
def multiply(seq, r=1, mod=None):
  """
  return the product of the numeric sequence <seq>.

  if <r> is specified this is used as the initial value of the product
  (and is the value returned when <seq> is empty).

  if <mod> is specified, the result at each stage is calculate mod <mod>.

  See also: math.prod() (Python 3.8).

  >>> multiply(irange(1, 7))
  5040
  >>> multiply([2] * 8)
  256
  >>> multiply(irange(100, 200), mod=1234)
  18
  """
  if mod is None:
    for x in seq:
      r *= x
  else:
    for x in seq:
      r *= x
      r %= mod
  return r

def avg(seq, div=fdiv):
  """
  calculate the arithmetic mean (average) of the values in <seq>.

  for sequences this is equivalent to: sum(seq) / len(seq)

  the function used for division can be provided with the 'div' parameter.

  >>> avg(irange(1, 10))
  5.5
  """
  t = k = 0
  for x in seq:
    t += x
    k += 1
  return div(t, k)

# vector dot product: dot(xs, ys, strict=0, fnp=multiply, fns=sum)
def dot(*vs, **kw):
  """
  this function takes a sequence of vectors provided as arguments,
  and calculates the product of the elements in the same position
  in each vector, and then sums these products.

  for two vectors this is the same as the vector dot product:

    dot((a1, a2, a3, ...), (b1, b2, b3, ...))
      = a1 * b1 + a2 * b2 + a3 * b3 + ...)

  if the 'strict' argument is present it will be passed to zip()
  (which in supported Python versions will throw an error if the
  inputs are not of equal length), otherwise the length of vectors
  processed will be defined by the shortest input vector.

  the functions used for the product and sum functions can be defined
  with the parameters 'fnp' (default is: multiply) and 'fns' (default
  is: sum).

  see also: math.sumprod() (Python 3.12)

  >>> dot((1, 3, -5), (4, -2, -1))
  3
  >>> call(dot, [(1, 3, -5)] * 2)
  35
  >>> call(dot, [(1, 3, -5)] * 3)
  -97
  """
  strict = kw.pop('strict', None)
  fns = kw.pop('fns', sum)
  fnp = kw.pop('fnp', multiply)
  if kw: raise TypeError(str.format("dot: unknown arguments {kw}", kw=seq2str(kw.keys())))
  z = (zip(*vs, strict=strict) if strict is not None else zip(*vs))
  return fns(map(fnp, z))

# multiple argument versions of basic operations
def add(*vs): "add(a, b, c, ...) = a + b + c + ..."; return sum(vs)
def mul(*vs): "mul(a, b, c, ...) = a * b * c * ..."; return multiply(vs)
def bit_or(*vs):
  "bit_or(a, b, c, ...) = a | b | c | ..."
  r = 0
  for v in vs: r |= v
  return r
def bit_xor(*vs):
  "bit_xor(a, b, c, ...) = a ^ b ^ c ^ ..."
  r = 0
  for v in vs: r ^= v
  return r
def bit_and(r, *vs):
  "bit_and(a, b, c, ...) = a & b & c & ..."
  for v in vs:
    r &= v
    if not r: break
  return r

def _gcd(a, b):
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

# or use math.gcd() [available from 3.5; from 3.9 = mgcd]
gcd = getattr(math, 'gcd', _gcd)


def _lcm(a, b):
  """
  lowest common multiple (on positive integers).

  >>> lcm(123, 456)
  18696
  >>> lcm(5, 7)
  35
  """
  return (a // gcd(a, b)) * b

# or use math.lcm() [available from 3.5; from 3.9 = mlcm]
lcm = getattr(math, 'lcm', _lcm)


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

# multiplicative inverse of <n> mod <m>
def _invmod(n, m):
  """
  return the multiplicative inverse of n mod m
  (or None if there is no inverse)

  i.e. the value x such that (n * x) % m = 1

  e.g. the inverse of 2 (mod 9) is 5, as (2 * 5) % 9 = 1
  >>> invmod(2, 9)
  5
  """
  (x, y, g) = egcd(n, m)
  return ((x % m) if g == 1 else None)

# from Python 3.8, pow() can do this for us
if _pythonv > (3, 7):
  def invmod(n, m): return catch(pow, n, -1, m)
  invmod.__doc__ = _invmod.__doc__
else:
  invmod = _invmod

# find square roots of <a> mod <m>
# this is OK for relatively small m, but more efficient (and complex)
# approaches are available (e.g. sympy.ntheory.sqrt_mod_iter)
def sqrtmod(a, m):
  """
  find square roots of a mod m.

  i.e. values x such that (x * x) is congurent to a (mod m).

  >>> sorted(sqrtmod(1, 16))
  [1, 7, 9, 15]
  >>> sorted(sqrtmod(17, 43))
  [19, 24]
  >>> sorted(sqrtmod(-1, 25))
  [7, 18]
  """
  a %= m
  for x in irange(0, m // 2):
    if (x * x) % m == a:
      # x is a root
      yield x
      # -x (mod m) is also a root
      if x > 0 and m > 2 * x: yield m - x

# multiple GCD
def mgcd(a, *rest):
  """
  GCD of multiple (two or more) integers.

  see also: math.gcd() (Python 3.9)

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

  see also: math.lcm() (Python 3.9)

  >>> mlcm(2, 3, 5, 9)
  90
  """
  return reduce(lcm, rest, a)

def is_coprime(*vs):
  return mgcd(*vs) == 1

# multiple divmod
# hours, minutes, seconds: (h, m, s) = mdivmod(x, 60, 60)
# days, hours, minutes, seconds: (d, h, m, s) = mdivmod(x, 24, 60, 60)
# days, hours, minutes, seconds, fractional seconds: (d, h, m, s, f) = mdivmod(x, 24, 60, 60, 1)
def mdivmod(x, *vs):
  rs = list()
  for v in reversed(vs):
    (x, r) = divmod(x, v)
    rs.insert(0, r)
  rs.insert(0, x)
  return rs

# for those times when Rational() is overkill
@static(Fraction=None)
def fraction(*args, **kw):
  """
  return the numerator and denominator of the fraction a/b in lowest terms

  if more than 2 arguments are specified the sum of the arguments as
  (numerator, denominator) pairs is determined, so:

  fraction(a, b, c, d, e, f, ...) -> a/b + c/d + e/f + ...

  >>> fraction(286, 1001)
  (2, 7)
  >>> fraction(1, 2,  1, 3,  1, 6)  # 1/2 + 1/3 + 1/6 = 1
  (1, 1)
  >>> fraction(1, 2,  3, 4,  5, 6)  # 1/2 + 3/4 + 5/6 = 25/12
  (25, 12)
  """
  if not args: return (0, 1)  # 0
  # go through the fractions in pairs
  ps = chunk(args, 2)
  (a, b) = next(ps)
  for (c, d) in ps:
    (a, b) = (a * d + b * c, b * d)
  if b == 0: raise ZeroDivisionError("fraction can't have zero denominator")
  if b < 0: (a, b) = (-a, -b)
  g = gcd(a, b)
  if g > 1: (a, b) = (a // g, b // g)
  return (a, b)

@static(rtype=None)
def Fraction(*args):
  """
  same as fraction(), but returns an object where the numerator and denominator
  can be referred to as obj.num and obj.den.
  """
  (a, b) = fraction(*args)
  if Fraction.rtype is None: Fraction.rtype = namedtuple('Fraction', 'num den')
  return Fraction.rtype(a, b)

def format_fraction(n, d, base=10):
  s = int2base(n, base=base)
  if d == 1: return s
  return s + "/" + int2base(d, base=base)

def ratio(*ns):
  """
  return ratio of integers in <ns> in lowest terms.

  >>> ratio(6, 8)
  (3, 4)
  >>> ratio(6, 8, 10)
  (3, 4, 5)
  """
  g = mgcd(*ns)
  return (ns if g == 1 else tuple(v // g for v in ns))

def ratio_q(*qs):
  """
  return ratio of fractions in <qs> as integers in lowest terms.

  >>> ratio_q(rational(2, 3), rational(10, 3))
  (1, 5)
  """
  # turn the fractions into integers
  m = call(mlcm, (q.denominator for q in qs))
  return call(ratio, (int(m * q) for q in qs))

# find an appropriate rational class
# (could also try "sympy.Rational", but not for speed)
# be aware when using gmpy2.mpq:
#   >> x = mpq(64)
#   >> y = mpq(x, 2)
# will end up setting both x and y to 32 (they are the same object)
# I have submitted a bug against gmpy2 (#334)
# instead do this:
#   >> x = mpq(64)
#   >> y = x / 2
# if fix_gmpy2=1 is set, a workaround will be used for gmpy2 < 2.1.4
@static(src="gmpy2.mpq gmpy.mpq fractions.Fraction", impl=dict(), fix_gmpy2=None)
def Rational(src=None, verbose=None):
  """
  select a class for representing rational numbers.

  >> Q = Rational(verbose=1)
  [Rational: using gmpy2.mpq]
  >> Q = Rational(src="fractions.Fraction", verbose=1)
  [Rational: using fractions.Fraction]
  """
  s = f = None
  if src is None:
    try:
      (s, f) = Rational.impl['*']
    except KeyError:
      src = Rational.src
  if f is None:
    for s in src.split():
      try:
        f = Rational.impl[s]
        break
      except KeyError:
        pass
      try:
        f = import_fn(s)
      except (ImportError, KeyError):
        continue
      Rational.impl[s] = f
      if '*' not in Rational.impl and src == Rational.src:
        Rational.impl['*'] = (s, f)
      # gmpy2 is fixed in v2.1.4+
      if s == 'gmpy2.mpq' and Rational.fix_gmpy2 is None:
        Rational.fix_gmpy2 = (not catch(require, "gmpy2.version", "2.1.4"))
      break
  if verbose is None: verbose = ('v' in _PY_ENIGMA)
  if verbose: printf("[Rational: using {s}]", s=(s if f else f))
  # fix for gmpy2.mpq() behaviour (issue #334) - may be fixed in gmpy2.version() > 2.1.2
  if Rational.fix_gmpy2 and s == 'gmpy2.mpq':
    if verbose: printf("[Rational: applying fix for {s}]")
    f = (lambda x, y=None, fn=f: (fn(x) if y is None else fn(x) / y))
  return f

# create a function that will calculate a/b, and return an int if the result is an integer
# or a rational object if the result is a rational
class Rdiv(object):
  def __init__(self, F=None, src=None, verbose=None):
    self.F = F
    self.src = src
    self.verbose = verbose
  def __call__(self, a, b):
    (d, r) = divmod(a, b)
    if r == 0: return d
    if self.F is None: self.F = Rational(src=self.src, verbose=self.verbose)
    return self.F(a) / b  # because mpq(x, y) changes x

rdiv = Rdiv()

@static(impl=None)
def rational(*args, **kw):
  """
  create an object representing a rational number.

  the class used is selected using Rational(), so for more control use:

    >> rational = Rational(verbose=1)
  or:
    >> rational = Rational(src="<preferred-implementations>", verbose=1)

  to see what implementation is being used.
  (or set the 'v' flag in the environment variable PY_ENIGMA).

  >>> rational(1, 49) * 49 == 1
  True
  """
  if rational.impl is None: rational.impl = Rational()
  return rational.impl(*args, **kw)

def factorial(a, *bs):
  """
  return a! / b!.

  >>> factorial(6)
  720
  >>> factorial(10, 7)
  720

  number of anagrams of "mississippi" (len = 11; 4x i, 4x s, 2x p)
  >>> factorial(11, 4, 4, 2)
  34650
  """
  if not bs: return math.factorial(a)
  r = None
  bs = sorted(bs, reverse=1)
  b = bs[0]
  if a - b < 100:
    r = multiply(irange(b + 1, a))
    bs.pop(0)
  if r is None: r = math.factorial(a)
  for b in bs:
    if b == 1: break
    (r, z) = divmod(r, math.factorial(b))
    if z != 0: raise ValueError("inexact division")
  return r

# multinomial coefficient
def multinomial(ks, n=None):
  """
  calculate multinomial coefficient.

  e.g. number of anagrams of "mississippi" (len = 11; 4x i, 4x s, 2x p)
  >>> multinomial([4, 4, 2, 1])
  34650
  >>> multinomial([4, 4, 2], 11)
  34650
  """
  if n is None: n = sum(ks)
  return factorial(n, *ks)


def nPr(n, r):
  """
  permutations functions: n P r.

  the number of ordered r-length selections from n elements
  (elements can only be used once).

  see also: math.perm() (Python 3.8).

  >>> nPr(10, 3)
  720
  """
  if r > n:
    return 0
  else:
    return math.factorial(n) // math.factorial(n - r)

P = nPr

def nCr(n, r):
  """
  combinatorial function: n C r.

  the number of unordered r-length selections from n elements
  (elements can only be used once).

  see also: math.comb() (Python 3.8).

  >>> nCr(10, 3)
  120
  """
  if r > n:
    return 0
  else:
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)

C = nCr

# NOTE: this corresponds to [[ select='R' ]] in subsets(), not [[ select='M' ]]
def M(n, k):
  """
  multichoose function: n M k.

  the number of unordered k-length selections from n elements where
  elements may be chosen multiple times.

  M(n, k) = icount(subsets(irange(1, n), size=k, select='R'))

  >>> M(10, 3)
  220
  """
  return C(n + k - 1, k)


@static(rtype=None)
def recurring(a, b, recur=0, base=10, digits=None):
  """
  find recurring representation of the fraction <a> / <b> in the specified base.
  return strings (<integer-part>, <non-recurring-part>, <recurring-part>)
  if you want rationals that normally terminate represented as non-terminating set <recur>

  >>> tuple(recurring(1, 7))
  ('0', '', '142857')
  >>> tuple(recurring(3, 2))
  ('1', '5', '')
  >>> tuple(recurring(3, 2, recur=1))
  ('1', '4', '9')
  >>> tuple(recurring(5, 17, base=16))
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
    j = r.get(a)
    if j is not None:
      # have we had this dividend before?
      (i, nr, rr) = (int2base(i, base, digits=digits), s[:j], s[j:])
      if neg and (nr or rr or i != '0'): i = '-' + i
      if recurring.rtype is None: recurring.rtype = namedtuple('Recurring', 'i nr rr')
      return recurring.rtype(i, nr, rr)
    else:
      # no, we haven't
      r[a] = n
      n += 1
      (d, a) = divmod(base * a, b)
      if recur and a == 0: (d, a) = (d - 1, b)
      if not (d == a == 0):
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
  dp = kw.pop('dp', '.')
  enc = kw.pop('enc', '()')
  dots = kw.pop('dots', '...')
  if kw: raise TypeError(str.format("format_recurring: unknown arguments {kw}", kw=seq2str(kw.keys())))
  if len(args) == 1: args = args[0]
  (i, nr, rr) = args
  rr = (enc[0] + rr + enc[-1] + dots if rr else '')
  return (i + dp + nr + rr if nr or rr else i)

# recurring -> fraction
def recurring2fraction(i, nr, rr, base=10, digits=None):
  """
  turn the decimal representation <i>.<nr>(<rr>)...
  into a fraction in its lowest terms.

  >>> recurring2fraction('0', '', '142857')
  (1, 7)
  >>> recurring2fraction('1', '5', '')
  (3, 2)
  >>> recurring2fraction('1', '4', '9')
  (3, 2)
  >>> recurring2fraction('0', '', '4B', base=16)
  (5, 17)
  """
  (p, q) = (len(nr), len(rr))
  i = base2int(i, base=base, digits=digits)
  if q:
    if p:
      d = (base**(p + q)) - (base**p)
      n = base2int(nr, base=base, digits=digits) * ((base**q) - 1) + base2int(rr, base=base, digits=digits)
    else:
      d = (base**q) - 1
      n = base2int(rr, base=base, digits=digits)
  elif p:
    d = base**p
    n = base2int(nr, base=base, digits=digits)
  else:
    return (i, 1)
  (a, b) = fraction(n, d)
  return ((i * b - a, b) if i < 0 else (i * b + a, b))

# see: Enigma 348
def reciprocals(k, b=1, a=1, m=1, M=inf, g=0, rs=[]):
  """
  generate k whole numbers (d1, d2, ..., dk) such that 1/d1 + 1/d2 + ... + 1/dk = a/b
  the numbers are generated as an ordered list
  m = minimum allowed number
  M = maximum allowed number
  g = minimum allowed gap between numbers

  e.g. 3 reciprocals that sum to 1:
  1/2 + 1/3 + 1/6 = 1
  1/2 + 1/4 + 1/4 = 1
  1/3 + 1/3 + 1/3 = 1
  >>> list(reciprocals(3, 1))
  [[2, 3, 6], [2, 4, 4], [3, 3, 3]]
  """
  # are we done?
  if k == 1:
    (d, r) = divmod(b, a)
    if r == 0 and not (d < m or d > M):
      yield rs + [d]
  elif k == 2:
    # special case k = 2
    if M == inf:
      dmin = divc(b + 1, a)
    else:
      dmin = divc(b * M, a * M - b)
    dmax = divf(b + b, a)
    for d in irange(max(m, dmin), min(M, dmax)):
      (e, r) = divmod(d * b, d * a - b)
      if r == 0 and not (e < d + g or e > M):
        yield rs + [d, e]
  else:
    if M == inf:
      # general case
      dmin = divc(b + 1, a)
    else:
      # but if M is given we can find a better dmin [suggested by frits]
      xs = list(M - g * i for i in xrange(k - 1))
      xd = multiply(xs)
      xn = sum(xd // x for x in xs)
      dmin = divc(b * xd, a * xd - b * xn)
    dmax = divf(k * b, a)
    # find a suitable reciprocal
    for d in irange(max(m, dmin), min(M, dmax)):
      # solve for the remaining fraction [[Python 3: yield from ... ]]
      for ds in reciprocals(k - 1, b * d, a * d - b, d + g, M, g, rs + [d]): yield ds


# command line arguments

# fetch command line arguments from sys
@static(argv=None)
def get_argv(force=0, args=None):
  if force or get_argv.argv is None:
    # TODO: decode command line args to unicode
    if args is None: args = sys.argv[1:]
    get_argv.argv = args
  return get_argv.argv

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
  return str.format(fmt, **vs)

# Python 3 has str.format_map(vs)
def __sprintf3(fmt, vs):
  return str.format_map(fmt, vs)

# in Python v3.6.x we are getting f"..." strings which can do this job
#
# NOTE: you lose the ability to do this:
#
#   printf("... {d[x]} ...", d={ 'x': 42 })  ->  "... 42 ..."
#
# instead you have to do this:
#
#   printf("... {d['x']} ...", d={ 'x': 42 })  ->  "... 42 ..."
#
# but you gain the ability to use arbitrary expressions:
#
#   printf("... {a} + {b} = {a + b} ...", a=2, b=3)  ->  "... 2 + 3 = 5 ..."
#
def __sprintf36(fmt, vs):
  return eval('f' + repr(fmt), vs)

@static(fn=None)
def _sprintf(fmt, vs, frame):
  # we can't use collections.ChainMap() here as eval() in __sprintf36() barfs
  d = dict(frame.f_globals)
  d.update(frame.f_locals)
  d.update(vs)
  return _sprintf.fn(fmt, d)

if _pythonv > (3, 5):
  _sprintf.fn = __sprintf36
elif _python > 2:
  _sprintf.fn = __sprintf3
else:
  _sprintf.fn = __sprintf

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
  print format string <fmt> with interpolated local variables and
  keyword arguments.

  the final newline can be suppressed by ending the string with '\\'
  (which you may need to escape).

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


def catch(fn, *args, **kw):
  """
  evaluate the function with the given arguments,
  but if it throws an exception return None instead.

  >>> catch(divmod, 7, 0) is None
  True
  """
  try:
    return fn(*args, **kw)
  except Exception:
    #print("catch: caught exception!"); print(sys.exc_info())
    return

# inclusive range iterator
# irange(a, b) -> [a, a + 1, ..., b]
# irange(n) -> irange(0, n - 1) -> [0, ..., n - 1]
@static(inf=inf) # so b=irange.inf can be used
def irange(a, b=None, step=1):
  """
  irange(a, b) =
  an integer range iterator that includes both endpoints, <a> and <b>.

  it will generate, in order, the integers: [a, a + k, a + 2k, ..., b]
  where <k> is the step.

  note that it is possible to choose endpoint/step combinations where
  the sequence of integers generated does not include b, or is empty.

  if <b> is specified as inf (or -inf for negative steps) the iterator
  will generate values indefinitely.

  irange(n) =
  if only one value <n> is specified for the endpoints, then endpoints
  of 0 and (n - 1) are used (these are swapped if <step> is
  negative), so that irange(n) produces n integers from 0 to n - 1.

  Note: Python's standard range iterator is available as xrange() if you
  want to emphasise the exclusion of the final endpoint.

  >>> list(irange(1, 9))
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(irange(9, 1, step=-1))
  [9, 8, 7, 6, 5, 4, 3, 2, 1]
  >>> list(irange(0, 10, step=3))
  [0, 3, 6, 9]
  >>> list(irange(10))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> list(irange(10, step=-1))
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

  """
  if not step: raise ValueError("irange: step cannot be zero")
  if b == inf:
    if step < 0: return xrange(0)
  elif b == -inf:
    if step > 0: return xrange(0)
  else:
    if b is None: (a, b) = ((0, a - 1) if step > 0 else (a - 1, 0))
    return xrange(a, b + (1 if step > 0 else -1), step)
  return itertools.count(start=a, step=step)

# inclusive range iterator that allows a fractional step
def irangef(a, b, step=1):
  """
  inclusive range iterator that allows the endpoints and the
  step to be fractional values.

  note that if float approximations are used for the step and/or
  endpoint then the final value may not be generated.

  >>> list(irangef(1, 2.5, step=0.5))
  [1.0, 1.5, 2.0, 2.5]
  """
  n = (inf if b == inf else divf(b - a, step))
  for i in irange(0, n):
    yield a + i * step

# flatten a list of lists
def flatten(s, fn=list):
  """
  flatten a list of lists (actually an iterator of iterators).

  the function: chain(*s) = flatten(s) is provided as a convenience.

  >>> flatten([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
  [1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> flatten(((1, 2), (3, 4, 5), (6, 7, 8, 9)), fn=tuple)
  (1, 2, 3, 4, 5, 6, 7, 8, 9)
  >>> flatten([['abc'], ['def', 'ghi']])
  ['abc', 'def', 'ghi']
  """
  return fn(j for i in s if i is not None for j in i)

# chain(a, b, c) = flatten([a, b, c])
# so: unpack(chain) = flatten
def chain(*ss, **kw):
  """
  a convenience function for calling flatten():

  chain(a, b, c, ...) = flatten([a, b, c, ...], fn=iter)

  >>> chain("abc", (1, 2, 3), None, [4, 5, 6], fn=tuple)
  ('a', 'b', 'c', 1, 2, 3, 4, 5, 6)
  """
  fn = kw.pop("fn", iter)
  if kw: raise TypeError(str.format("chain: unknown arguments {kw}", kw=seq2str(kw.keys())))
  return flatten(ss, fn=fn)

# interleave values from a bunch of iterators
# flatten(zip(*ss), fn=iter) works if arguments are the same length
def interleave(*ss, **kw):
  ss = list(iter(s) for s in ss)
  n = len(ss)
  while n > 0:
    i = 0
    while i < n:
      try:
        yield next(ss[i])
        i += 1
      except StopIteration:
        ss.pop(i)
        n -= 1

# do we flatten this?
def _flatten_test(s):
  # don't flatten strings
  if isinstance(s, (basestring, bytes)):
    return None
  # do flatten other sequences
  if isinstance(s, (Sequence, Iterable)):
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

# in Python 3 we could use functools.singledispatch
# or in Python 3.10+ we could use structural pattern matching
# to implement polymorphic functions (update(), delete(), append())
# however there does not seem to be a performance advantage
# and it will fail with older versions of Python

# return a copy of object <s>, but with value <v> at index <k> for (k, v) in <ps>
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

  >>> update((1, 2, 3), [(2, 4)])
  (1, 2, 4)
  """
  if vs is not None: ps = zip(ps, vs)
  # allow updating of immutable types: tuple, string
  fn = None
  if isinstance(s, tuple):
    fn = type(s)
    s = list(s)
  elif isinstance(s, basestring):
    fn = ''.join
    s = list(s)
  else:
    # use copy() method if available
    # otherwise create a new object initialised from the old one
    cpy = getattr(s, 'copy', None)
    s = (cpy() if cpy else type(s)(s))
  # use update() method if available
  upd = getattr(s, 'update', None)
  if upd:
    upd(ps)
  else:
    # otherwise update the pairs individually
    for (k, v) in ps:
      s[k] = v
  # return the new object
  return (fn(s) if fn else s)

# return a copy of object <s> with values at indices <ks> removed
def delete(s, ks=()):
  """
  return an updated version of object <s> with items at keys <ks> removed.

  >>> delete(dict(a=1, b=2, c=3), 'bc') == dict(a=1)
  True
  >>> delete("bananas", [0, 2, 4, 6])
  'aaa'
  """
  fn = None
  if isinstance(s, list):
    s = list(s)
    ks = sorted(ks, reverse=1)
  elif isinstance(s, tuple):
    fn = type(s)
    s = list(s)
    ks = sorted(ks, reverse=1)
  elif isinstance(s, basestring):
    fn = ''.join
    s = list(s)
    ks = sorted(ks, reverse=1)
  else:
    # use copy() method if available
    # otherwise create a new object initialised from the old one
    cpy = getattr(s, 'copy', None)
    s = (cpy() if cpy else type(s)(s))
  # remove specified keys
  for k in ks:
    del s[k]
  # return the new object
  return (fn(s) if fn else s)

# this unifies adding an element (or elements) to a container:
# - string: '123' + '4'      -> append('123', '4')
# - list:   [1, 2, 3] + [4]  -> append([1, 2, 3], 4)
# - tuple:  (1, 2, 3) + (4,) -> append((1, 2, 3), 4)
# - set:    {1, 2, 3} | {4}  -> append({1, 2, 3}, 4)
def append(s, *vs):
  """
  make a new container, the same as <s> but with additional values <vs> added.

  if the container has a sense or order, items are added at the end.

  >>> append((1, 2, 3), 4)
  (1, 2, 3, 4)
  >>> append([1, 2, 3], 4)
  [1, 2, 3, 4]
  >>> append({1, 2, 3}, 4) == {1, 2, 3, 4}
  True
  >>> append('123', '4')
  '1234'
  """
  if isinstance(s, list):
    r = type(s)(s)
    r.extend(vs)
    return r
  if isinstance(s, tuple):
    return s + vs
  if isinstance(s, basestring):
    return s + str.join('', vs)
  if isinstance(s, set):
    r = s.copy()
    r.update(vs)
    return r
  if isinstance(s, frozenset):
    return s.union(vs)
  if isinstance(s, multiset):
    return s.copy().update_from_seq(vs)
  raise ValueError(str.format("append() can't handle container of type {x}", x=type(s)))

# this unifies removing an element (or elements) from a container:
def remove(s, *vs):
  """
  make a new container, the same as <s> but with values in <vs> removed.

  items appearing in <vs> that are not in <s> are ignored.

  if the container has a sense of order, the earliest items are removed

  >>> remove((1, 2, 3), 2)
  (1, 3)
  >>> remove([1, 2, 3, 2], 2)
  [1, 3, 2]
  >>> remove({1, 2, 3}, 2) == {1, 3}
  True
  >>> remove('1232', '2')
  '132'
  """
  if isinstance(s, list):
    r = type(s)(s)
    for v in vs:
      try:
        r.remove(v)
      except ValueError:
        continue
    return r
  if isinstance(s, tuple):
    return type(s)(remove(list(s), *vs))
  if isinstance(s, basestring):
    for v in vs:
      s = s.replace(v, '', 1)
    return s
  if isinstance(s, (set, frozenset, multiset)):
    return s.difference(vs)
  raise ValueError(str.format("remove() can't handle container of type {x}", x=type(s)))

# restriction of a container <s> that only includes items at keys <ks>
def restrict(s, ks, strict=0):
  """
  create the restriction of container <s> that only includes keys in <ks>

  keys that do not occur in the original container are ignored, unless
  the <strict> flag is set, in which case an exception is raised.

  >>> map2str(restrict(dict(a=1, b=2, c=3, d=4), {'a', 'b', 'x'}))
  '(a=1, b=2)'
  >>> restrict(['zero', 'one', 'two', 'three'], [1, 3])
  ['one', 'three']
  >>> restrict("abracadabra", [0, 3, 5, 7, 10])
  'aaaaa'
  """
  if isinstance(s, dict):
    r = type(s)()
    for k in ks:
      try:
        r[k] = s[k]
      except KeyError:
        if strict: raise
    return r
  if isinstance(s, (list, tuple, basestring)):
    r = list()
    for k in ks:
      try:
        r.append(s[k])
      except IndexError:
        if strict: raise
    return (join(r) if isinstance(s, basestring) else type(s)(r))

# adjacency matrix for an n (columns) x m (rows) grid
# entries are returned as lists in case you want to modify them before use
def grid_adjacency(n, m, deltas=None, include_adjacent=1, include_diagonal=0, include_self=0, fn=None):
  """
  this function generates the adjacency matrix for a grid with n
  columns and m rows, represented by a linear array of size n * m.

  the element in the (i, j)th position in the grid is at index (i + n * j)
  in the array.

  it returns an array, where the entry at index k is the collection of
  indices into the linear array that are adjacent to the square at index k.

  if 'fn' is specified then it is used to collect the indices,
  otherwise they are returned as a list.

  the default behaviour is to treat the squares immediately N, S, E, W
  of the target square as being adjacent, although this can be controlled
  with the 'deltas' parameter, it can be specified as a list of (x, y)
  deltas to use instead.

  if 'deltas' is not specified the 'include_adjacent', 'include_diagonal'
  and 'include_self' flags are used to specify which squares are adjacent
  to the target square:
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
  for y in xrange(0, m):
    for x in xrange(0, n):
      s = list()
      for (dx, dy) in deltas:
        (x1, y1) = (x + dx, y + dy)
        if not (x1 < 0 or y1 < 0 or x1 + 1 > n or y1 + 1 > m):
          s.append(x1 + y1 * n)
      r[x + y * n] = (fn(s) if fn else s)
  return r


# cumulative sum
def csum(seq, s=0, fn=operator.add, empty=0):
  """
  generate cumulative partial sums from sequence <seq>.

  's' is the initial value, and 'fn' is the function used to update it
  with each element of the sequence.

  if 'empty' is set to a true value then the initial value 's' will be
  initially returned.

  see also: itertools.accumulate() (Python 3.2).

  >>> list(csum(irange(1, 10)))
  [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
  >>> list(csum(irange(1, 10), fn=operator.mul, s=1))
  [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]

  """
  if empty: yield s
  for x in seq:
    s = fn(s, x)
    yield s


# cumulative slice
def cslice(seq, empty=0):
  """
  generate an iterator that is the cumulative slices of a sequence.

  >>> list(cslice([1, 2, 3]))
  [[1], [1, 2], [1, 2, 3]]
  >>> list(cslice('python'))
  ['p', 'py', 'pyt', 'pyth', 'pytho', 'python']
  """
  for i in irange((0 if empty else 1), len(seq)):
    yield seq[:i]


# overlapping tuples from a sequence
def tuples(seq, n=2, circular=0, fn=tuple):
  """
  generate overlapping <n>-tuples from sequence <seq>.
  (for non-overlapping tuples see chunk()).

  if 'circular' is set to true, then values from the beginning of <seq>
  will be used to complete tuples when the end is reached.

  see also: itertools.pairwise() (Python 3.10).

  >>> list(tuples('ABCDE'))
  [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
  >>> list(tuples(irange(1, 5), 3))
  [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
  >>> list(tuples(irange(1, 5), 3, circular=1))
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)]
  """
  if n < 1: raise ValueError(str.format("invalid tuple length: {n!r}", n=n))
  i = iter(seq)
  if circular and n > 1:
    # we need extract the first (n - 1) items and add them to the end
    xs = first(i, n - 1)
    if not xs: return
    m = len(xs)
    if m < n - 1:
      i = itertools.chain(xs, i, (xs[k % m] for k in xrange(n - 1)))
    else:
      i = itertools.chain(xs, i, xs)

  t = list()
  try:
    # collect the first tuple
    for _ in xrange(n):
      t.append(next(i))
    while True:
      # return the tuple
      yield fn(t)
      # move the next value in to the tuple
      t.append(next(i))
      t.pop(0)
  except StopIteration:
    return

def last(seq, count=1, fn=list):
  """
  find the last <count> items in sequence <seq>.

  >>> last([1, 2, 3, 4])
  [4]
  >>> last(Primes(30), 3)
  [19, 23, 29]
  """
  try:
    x = seq[-count:]
    if len(x) < count: return
  except TypeError:
    x = None
    for x in tuples(seq, count, fn=list): pass
    if x is None: return
  return (x if fn == list else fn(x))

def contains(seq, subseq):
  """
  return the position in <seq> that <subseq> occurs as a contiguous subsequence
  or -1 if it is not found

  >>> contains("abcdefghijkl", "def")
  3
  >>> contains("abcdefghijkl", "hik")
  -1
  >>> contains(primes, [11, 13, 17, 19])
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

    a = pow(2, k) - 1
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

def bit_from_positions(ps):
  """
  construct an integer with bit positions in <ps> set

  >>> bit_from_positions({1, 3, 7, 11})
  2186
  """
  return call(bit_or, (1 << p for p in ps))

def bit_positions(x):
  """
  return the positions of bits set in integer <x>

  >>> list(bit_positions(2186))
  [1, 3, 7, 11]
  """
  i = 0
  while x:
    if x & 1: yield i
    x >>= 1
    i += 1

# for "coin puzzles", see also: Denominations()

# simple express:
def express(t, ds, qs=None, min_q=0):
  """
  express total <t> using denominations <ds>.

  optional: using quantities chosen from <qs>
  or: minimum quantity <min_q> (non-negative integer)

  <ds> and <qs> should be increasing sequences.

  generated values are the quantities for each denomination in <ds>.

  >>> list(express(20, (3, 5, 7)))
  [[0, 4, 0], [1, 2, 1], [2, 0, 2], [5, 1, 0]]

  >>> list(express(20, (3, 5, 7), min_q=1))
  [[1, 2, 1]]

  >>> list(express(20, (3, 5, 7), qs=(0, 1, 2)))
  [[1, 2, 1], [2, 0, 2]]

  the number of ways to change 1 pound into smaller coins
  >>> icount(express(100, (1, 2, 5, 10, 20, 50)))
  4562
  """
  ds = list(ds)
  if not (ds and ds[0] > 0): raise ValueError(str.format("invalid denominations {ds!r}", ds=ds))
  if qs:
    return express_quantities(t, ds, qs)
  if min_q > 0:
    return express_denominations_min(t, ds, min_q)
  else:
    return express_denominations(t, ds)

# express total <t> using denominations <ds>
def express_denominations(t, ds, ss=[]):
  if t == 0:
    if not ds:
      yield ss
    else:
      yield ss + [0] * len(ds)
  elif ds:
    d = ds[0]
    (k, r) = divmod(t, d)
    if len(ds) == 1:
      if r: return
      qs = [k]
    else:
      qs = irange(0, k)
    for q in qs:
      for r in express_denominations(t - d * q, ds[1:], ss + [q]): yield r

# express total <t> using denominations <ds>, min quantity <min_q>
def express_denominations_min(t, ds, min_q):
  # allocate the minimum quantities
  t -= min_q * sum(ds)
  if t == 0:
    yield [min_q] * len(ds)
  elif t > 0:
    # solve for the remaining amount
    for ss in express_denominations(t, ds):
      # add in the initial quantities
      yield list(q + min_q for q in ss)

# express total <t> using denominations <ds>, quantities chosen from <qs>
def express_quantities(t, ds, qs, ss=[]):
  if t == 0:
    if not ds:
      yield ss
    elif 0 in qs:
      yield ss + [0] * len(ds)
  elif ds:
    d = ds[0]
    for q in qs:
      if d * q > t: break
      for r in express_quantities(t - d * q, ds[1:], qs, ss + [q]): yield r


# An implementation of the Boecker-Liptak Money Changing algorithm from:
#
# https://bio.informatik.uni-jena.de/bib2html/downloads/2007/BoeckerLiptak_FastSimpleAlgorithmMoneyChanging_Algorithmica_2007.pdf
# https://pdfs.semanticscholar.org/14ac/14a15ebc31b58a4ac04328f9824f743a1e4e.pdf
#
# this implementation uses the "Round Robin" algorithm before optimisations.

# make the Extended Residue Table
def _residues(vs):
  # empty table
  res = [None] * len(vs)
  # initial row
  v0 = vs[0]
  r = [inf] * v0
  r[0] = 0
  # fill out each row
  for (i, vi) in enumerate(vs):
    if i > 0:
      d = gcd(v0, vi)
      for p in xrange(d):
        m = min(r[q] for q in xrange(p, v0, d))
        if m < inf:
          for c in xrange(v0 // d):
            m += vi
            j = m % v0
            if r[j] < m:
              m = r[j]
            else:
              r[j] = m
    res[i] = list(r)
  return res

# generate possible expressions of t
def _find_all(t, vs, i, c, res):
  v0 = vs[0]
  if i == 0:
    c[0] = t // v0
    yield tuple(c)
  else:
    vi = vs[i]
    m = lcm(v0, vi)
    d = m // vi
    for j in xrange(0, d):
      c[i] = j
      u = t - j * vi
      b = res[i - 1][u % v0]
      while u >= b:
        for x in _find_all(u, vs, i - 1, c, res): yield x
        u -= m
        c[i] += d


class Denominations(object):
  """
  An implementation of the Boecker-Liptak Money Changing algorithm.

  The denominations passed in are sorted into increasing order, and
  accessible via the 'denominations' attribute.

  Quantities returned by 'express()' are in the same order as the
  'denominations' attribute.

  >>> sorted(Denominations([3, 5, 7]).express(20))
  [(0, 4, 0), (1, 2, 1), (2, 0, 2), (5, 1, 0)]

  >>> sorted(Denominations([3, 5, 7]).express(20, min_q=1))
  [(1, 2, 1)]

  the number of ways to change 1 pound into smaller coins:
  >>> Denominations([1, 2, 5, 10, 20, 50]).count(100)
  4562

  using at least 1 of each type of coin:
  >>> Denominations([1, 2, 5, 10, 20, 50]).count(100, min_q=1)
  15

  the largest non-McNugget number:
  >>> Denominations([6, 9, 20]).frobenius()
  43
  """
  def __init__(self, *denominations):
    # preferred initialisation is to pass a sequence of denominations
    if len(denominations) == 1: denominations = denominations[0]
    # first sort the denominations
    try:
      ds = tuple(sorted(denominations))
    except TypeError:
      ds = ()
    if not (len(ds) > 1 and ds[0] > 0 and seq_all_different(ds)):
      raise ValueError(str.format("invalid denominations: {denominations}", denominations=denominations))
    self.denominations = ds
    # compute the extended residue table for the given denominations
    self.residues = _residues(ds)

  # generate different ways to express <amount>
  def express(self, amount, min_q=0):
    """
    generate the different ways to express the given amount.

    if min_q is specified (non-negative integer), at least that many
    instances of each denomination must be used.
    """
    n = len(self.denominations)
    if min_q == 0:
      for t in _find_all(amount, self.denominations, n - 1, [0] * n, self.residues):
        yield t
    elif min_q > 0:
      amount -= min_q * sum(self.denominations)
      if amount == 0:
        yield (min_q,) * n
      elif amount > 0:
        for t in _find_all(amount, self.denominations, n - 1, [0] * n, self.residues):
          yield tuple(x + min_q for x in t)

  # count the number of ways to express <amount>
  def count(self, amount, min_q=0):
    """count the number of ways of expressing an amount"""
    return icount(self.express(amount, min_q=min_q))

  # return the Frobenius number (the largest amount that cannot be changed)
  def frobenius(self):
    """return the largest amount not expressible using the denominations"""
    m = max(self.residues[-1])
    return (None if m == inf else m - self.denominations[0])

# return a function to generate k-sequences of positive integers with a particular total
def Decompose(k=None, increasing=1, sep=1, min_v=1, max_v=inf, fn=identity):
  """
  return a function to generate k-sequences of non-negative integers
  that sum to a chosen total

    k = length of sequences to generate
    increasing = +1 (increasing sequences [default]); -1 (decreasing sequences); or 0
    sep = separation between numbers; 0 allows repeats [default: 1]
    min_v = minimum permissible value (non-negative integer)
    max_v = maximum permissible value (non-negative integer, or inf)
    fn = return type (default is to return tuples)
  """
  # decompose t into k increasing numbers, in range [min_v, max_v]
  # d = delta between numbers (for inc/dec seqs)
  # R = function to calculate remaining values
  # M = function to calculate next minimum value
  # r = reverse return values
  # fn = return type
  # ns = numbers collected so far
  def _decompose(t, k, min_v, max_v, d, R, M, r, fn, ns=()):
    if k == 0:
      if t == 0: yield fn(())
    elif k == 1:
      if not (t < min_v or t > max_v):
        ns += (t,)
        yield fn(ns[::-1] if r else ns)
    else:
      k_ = k - 1
      for n in irange(min_v, min(max_v, R(t, k, k_, min_v))):
        for z in _decompose(t - n, k_, M(n, d), max_v, d, R, M, r, fn, ns + (n,)): yield z

  if increasing == 0:
    # generate increasing sequences with the appropriate sep value
    # and then permute the answers (which may contain repeats if sep=0)
    f = Decompose(k, increasing=1, sep=sep, min_v=min_v, max_v=max_v, fn=fn)
    perm = (mpermutations if sep == 0 else itertools.permutations)
    return (lambda t, k=k: flatten((perm(ns, k) for ns in f(t, k)), fn=iter))
  else:
    d = abs(sep)
    if d == 0:
      R = (lambda t, k, k_, m: t // k)
      M = (lambda n, d: n)
    elif d == 1:
      R = (lambda t, k, k_, m: (t - (k * k_) // 2) // k)
      M = (lambda n, d: n + 1)
    else:
      R = (lambda t, k, k_, m: (t - (d * k * k_) // 2) // k)
      M = (lambda n, d: n + d)
    r = (increasing < 0)
    return (lambda t, k=k, min_v=min_v: _decompose(t, k, min_v, max_v, d, R, M, r, fn))

# all-in-one
def decompose(t, k, increasing=1, sep=1, min_v=1, max_v=inf, fn=identity):
  """
  decompose <t> in <k>-sequences of non-negative integers that sum to <t>

    t = total sum of each sequence
    k = length of sequences to generate
    increasing = +1 (increasing sequences); -1 (decreasing sequences); or 0
    sep = separation between numbers (if increasing != 0); 0 allows repeats
    min_v = minimum permissible value (non-negative integer)
    max_v = maximum permissible value (non-negative integer, or inf)
    fn = return type (default is to return tuples)

  >>> sorted(decompose(10, 3, increasing=1, min_v=1))
  [(1, 2, 7), (1, 3, 6), (1, 4, 5), (2, 3, 5)]
  >>> sorted(decompose(8, 3, increasing=1, min_v=0))
  [(0, 1, 7), (0, 2, 6), (0, 3, 5), (1, 2, 5), (1, 3, 4)]
  >>> sorted(decompose(8, 3, increasing=1, sep=0, min_v=1))
  [(1, 1, 6), (1, 2, 5), (1, 3, 4), (2, 2, 4), (2, 3, 3)]
  >>> sorted(decompose(5, 3, increasing=0, sep=0, min_v=1))
  [(1, 1, 3), (1, 2, 2), (1, 3, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1)]
  """
  return call(Decompose(increasing=increasing, sep=sep, min_v=min_v, max_v=max_v, fn=fn), (t, k))

###############################################################################

# exact set cover (using Knuth's Algorithm X)

# in-place algorithmX implementation (X, soln are modified)
def algorithmX(X, Y, soln):
  if not X:
    yield soln
  else:
    c = min(X.keys(), key=(lambda k: len(X[k])))
    # copy X[c], as X is modified (could use sorted(X[c]) for stability)
    for r in list(X[c]):
      soln.append(r)

      # cols = select(X, Y, r)
      cols = list()
      for j in Y[r]:
        for i in X[j]:
          for k in Y[i]:
            if k != j:
              X[k].remove(i)
        cols.append(X.pop(j))

      for z in algorithmX(X, Y, soln): yield z

      # deselect(X, Y, r, cols)
      for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
          for k in Y[i]:
            if k != j:
              X[k].add(i)

      soln.pop()

# input: ss = sequence of collections of sets [ [a0, a1, ...], [b1, b2, ...], [c1, c2, ...] ... ]
# output: sequence of sets (a, b, c, ...) one from each collection
def exact_cover(sss, tgt=None):
  """
  given a collection of sets (of sets):
    [a0, a1, ...]
    [b1, b2, ...]
    ...

  an exact cover is a selection of sets:
    [a, b, ...]

  where a is chosen from the first collection, b from the second, etc.

  and each element of the target set appears in exactly one of the
  sets in the cover.

  if the target set is not specified, it is the collection of all elements
  contained in any of the provided sets.
  """
  # map elements to indices
  if tgt is None: tgt = union(union(ss) for ss in sss)
  tgt = sorted(tgt)
  n = len(tgt)
  m = dict((x, i) for (i, x) in enumerate(tgt))

  # set up Y, one row for each position
  Y = list()
  for (j, ss) in enumerate(sss, start=n):
    for s in ss:
      y = list(m[x] for x in s)
      y.append(j)
      Y.append(y)

  # set up X as a dict of sets
  X = dict((k, set()) for k in irange(0, j))
  for (i, y) in enumerate(Y):
    for k in y:
      X[k].add(i)

  # find exact covers using algorithmX
  k = len(sss)
  for rs in algorithmX(X, Y, list()):
    # turn the selected rows of Y, back into sets
    r = [None] * k
    for i in rs:
      y = Y[i]
      r[y[-1] - n] = list(tgt[i] for i in y[:-1])
    yield r

# exact multiset cover (see: Enigma 1712, Teaser 2690)
def mcover(m, tgt, reject=None):
  """
  find exact multiset covers.

  <m> is a map of keys to multisets of values
  <tgt> is a multiset of values to target
  <reject> is a function that can be used to reject partial solutions

  solutions are returned as an (unordered) list of keys, where the
  combined multisets of values corresponding to those keys give
  exactly the target multiset.
  """
  # a variation on Knuth's Algorithm X
  def _mcover(m, tgt, X, ss):
    # are we done?
    if not tgt:
      yield ss
    else:
      # choose a column to work on
      c = min(tgt.keys(), key=(lambda k: (len(X[k]), -tgt[k])))
      # consider subsets with this value
      xs = sorted(X[c])
      for (j, n) in enumerate(xs, start=1):
        s = m[n]
        # update the target
        (tgt_, rs) = tgt.differences(s)
        if rs: continue
        # is this sequence acceptable?
        ss_ = ss + [n]
        if reject and reject(ss_): continue
        # remove the value (and any prior values) from consideration
        discard = set(xs[:j])
        # remove any values in columns that have reached 0
        for (k, v) in X.items():
          if v and tgt_.count(k) == 0:
            discard.update(v)
        # recurse with a new target and X
        X_ = dict((k, v.difference(discard)) for (k, v) in X.items())
        for z in _mcover(m, tgt_, X_, ss_): yield z

  # X tells us what elements of the target are involved in which values
  if not isinstance(tgt, multiset): tgt = multiset(tgt)
  ks = set(tgt.keys())
  X = dict((k, set()) for k in ks)
  for (v, s) in m.items():
    if ks.issuperset(s.keys()):
      for k in s.keys():
        X[k].add(v)
  # check each target element appears
  if not all(X.values()): return ()
  # solve using the variation on Algorithm X
  return _mcover(m, tgt, X, list())

###############################################################################

# Numerical approximations

# a simple record type class for results
# (Python 3.3 introduced types.SimpleNamespace)
class Record(object):

  # best called as Record.update(self, ...)
  def update(self, **vs):
    """update values in a record"""
    self.__dict__.update(vs)
    return self

  # __init__ is the same as update
  def __init__(self, **vs):
    Record.update(self, **vs)

  def __iter__(self):
    d = self.__dict__
    for k in sorted(d.keys()):
      yield (k, d[k])

  def __repr__(self):
    return self.__class__.__name__ + map2str((k, repr(v)) for (k, v) in self)

  # best called as Record.map(self, ...)
  def map(self):
    return self.__dict__


# a golden section search minimiser
# f - function to minimise
# a, b - bracket to search
# t - tolerance
# m - metric
def gss_minimiser(f, a, b, t=1e-9, m=None):
  # apply any metric
  fn = (f if m is None else (lambda x: m(f(x))))
  R = 0.5 * (math.sqrt(5.0) - 1.0)
  C = 1.0 - R
  (x1, x2) = (R * a + C * b, C * a + R * b)
  (f1, f2) = (fn(x1), fn(x2))
  i = 0  # count iterations
  while b - a > t:
    i += 1
    if f1 > f2:
      (a, x1, f1) = (x1, x2, f2)
      x2 = C * a + R * b
      f2 = fn(x2)
    else:
      (b, x2, f2) = (x2, x1, f1)
      x1 = R * a + C * b
      f1 = fn(x1)
  (v, fv) = ((x1, f1) if f1 < f2 else (x2, f2))
  return Record(v=v, fv=fv, t=t, i=i)


find_min = gss_minimiser
find_min.__name__ = 'find_min'
find_min.__doc__ = """
  find a minimum value of a (well behaved) function over an interval.

  f = function to minimise (should take a single float argument)
  a, b = the interval to minimise over (a < b)
  t = the tolerance to work to
  m = the metric we want to minimise (default is None = the value of the function)

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is minimised
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_min(lambda x: sq(x - 2), 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
"""

# NOTE: using functools.partial and setting __name__ and __doc__ doesn't work (in Python 2.7 and 3.3)
# see: http://bugs.python.org/issue12790
def find_max(f, a, b, t=1e-9):
  """
  find a maximum value of a (well behaved) function over an interval.

  f = function to maximise (should take a single float argument)
  a, b = the interval to search (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is maximised
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_max(lambda x: 9 - sq(x - 2), 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  """
  return gss_minimiser(f, a, b, t=t, m=neg)

# we can also use the minimiser to find roots
# there are more rapidly converging root finding algorithms,
# but unless f() is very expensive to call this will suffice
def find_zero(f, a, b, t=1e-9, ft=1e-6):
  """
  find a zero of a (well behaved) function over an interval.

  f = function to find the zero of (should take a single float argument)
  a, b = the interval to search (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is zero
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_zero(lambda x: sq(x) - 4, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  >>> r = find_zero(lambda x: sq(x) + 4, 0.0, 10.0) # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
    ...
  ValueError: value not found
  """
  # find a minimum of the absolute value
  r = find_min(f, a, b, t, m=abs)
  # and check the function value is close enough to 0
  if ft < abs(r.fv): raise ValueError("value not found") # try a smaller t, or a larger ft
  r.ft = ft
  return r

def find_value(f, v, a, b, t=1e-9, ft=1e-6):
  """
  find a value of a (well behaved) function over an interval.

  f = function to find the value of (should take a single float argument)
  a, b = the interval to search (a < b)
  t = the tolerance to work to

  the result is returned as a record with the following fields:
  v = the calculated value at which the function is the specified value
  fv = the value of the function at v
  t = the tolerance used

  >>> r = find_value(lambda x: sq(x) + 4, 8.0, 0.0, 10.0)
  >>> round(r.v, 6)
  2.0
  """
  r = find_zero((lambda x: f(x) - v), a, b, t, ft)
  r.fv += v
  return r

# 2D geometry: a point is represented by (x, y)

def line_intersect(p1, p2, p3, p4, internal=0, div=fdiv):
  """
  Find the intersection of 2 lines defined by points:

    line 1 passes through p1 and p2 (= (x1, y1) and (x2, y2))
    line 2 passes through p3 and p4 (= (x3, y3) and (x4, y4))

  internal can be set to: 1, 2, 1+2 to check the intersection is internal
  to the specified line segments, if not an exception is raised

  div is set to an appropriate division function (default is fdiv()
  for floats, but the result of Rational() could be used)

  return value is a Record object:
    pt = (x, y) value of intersection
    x = x
    y = y
    q1 = fraction along line 1 (0 = p1, 1 = p2)
    q2 = fraction along line 2 (0 = p3, 1 = p4)
  """
  ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = (p1, p2, p3, p4)
  (dx21, dx43, dx31) = (x2 - x1, x4 - x3, x3 - x1)
  (dy21, dy43, dy31) = (y2 - y1, y4 - y3, y3 - y1)
  z = dx21 * dy43 - dy21 * dx43
  if z == 0: raise ValueError("invalid lines")
  # calculate parameter for line 1 = (x1 + r1(x2 - x1), y1 + r1(y2 - y1))
  q1 = div(dx31 * dy43 - dy31 * dx43, z)
  if internal & 1 and (q1 < 0 or q1 > 1): raise ValueError("external intersection on line 1")
  # calculate intersection point
  (x, y) = (x1 + q1 * dx21, y1 + q1 * dy21)
  # calculate parameter for line 2 = (x3 + q2(x4 - x3), y3 + q2(y4 - x3))
  if abs(dx43) > abs(dy43):
    q2 = div(x - x3, dx43)
  else:
    q2 = div(y - y3, dy43)
  if internal & 2 and (q2 < 0 or q2 > 1): raise ValueError("external intersection on line 2")
  # return intersection point (pt) (also: x, y, q1, q2)
  return Record(pt=(x, y), x=x, y=y, q1=q1, q2=q2)

# return a line segment the same length as (p1, p2) that is its perpendicular bisector
# i.e. (p1, p2) forms the diagonal of a square, the other diagonal is the return value (p3, p4)
def line_bisect(p1, p2, div=fdiv):
  """
  Return a line segment that is a perpendicular bisector of the
  line segment defined by p1 and p2 (= (x1, y1) and (x2, y2)).

  The value returned (p3, p4) (= (x3, y3), (x4, y4)) is a line segment
  that forms a diagonal of a square, where the other diagonal is (p1, p2).
  """
  ((x1, y1), (x2, y2)) = (p1, p2)
  s = fdiv(x1 + x2 + y1 + y2, 2)
  return ((s - y1, s - x2), (s - y2, s - x1))

# return the closest distance between a line that passes through p1 and p2
# and the point p0
def line_distance(p1, p2, p0=(0, 0)):
  """
  Return the minimum distance between the point p0 (= (x0, y0)
  and a line that passes through points p1 (= (x1, y1))
  and p2 (= (x2, y2)).

  If p0 is not specified (0, 0) is used.
  """
  ((x1, y1), (x2, y2), (x0, y0)) = (p1, p2, p0)
  (xd, yd) = (x2 - x1, y2 - y1)
  return fdiv(abs(xd * (y1 - y0) - yd * (x1 - x0)), hypot(xd, yd))

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
  x = as_int(x)
  if not (0 < x < 5000): raise ValueError("integer out of range: {x}".format(x=x))
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

# digits = (default) digits for use in converting bases
@static(digits=str_digit + str_upper)
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
  if len(args) > 1: raise TypeError(str.format("invalid base digits: {args!r}", args=args))
  if args: base_digits.digits = (args[0] or _DIGITS)
  return base_digits.digits

def int2base(i, base=10, width=None, pad=None, group=None, sep=",", digits=None):
  """
  convert an integer <i> to a string representation in the specified
  base <base>.

  if the <width> parameter is specified the number of digits will be
  padded to value of <width> using the <pad> character. if <width> is
  positive pad characters will be added on the left, if negative they
  are added on the right. The default pad character is the digit 0.

  if the <group> parameter is specified the digits are grouped into
  blocks of <group> digits and separated by the string <sep> (this
  happens after the digits are padded to any specified <width>). if
  <group> is positive the groups start from the right, if negative
  they start from the left.

  By default this routine only handles single digits up 36 in any
  given base, using standard digits 0-9 and then letters A-Z, but
  the <digits> parameter can be specified to give the symbols for
  larger bases. And if there are more digits in the specified base
  then there are available symbols, then the returned string will
  be of the form "{<n>:<n>:...}" where <n> is the digit value
  expressed in decimal (using digits 0-9).

  see also: gmpy2.digits()

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
  >>> int2base(84, base=2, width=9, group=3, sep=" ")
  '001 010 100'
  """
  if base < 2: raise ValueError("invalid base {base!r}".format(base=base))
  if digits is None: digits = base_digits()
  (p, r) = ('', None)
  if i < 0: (p, i) = ('-', -i)
  # if there aren't enough digits switch to {<digit>:<digit>:...} format
  if len(digits) < base:
    ds = nsplit(i, base=base, fn=list)
    # pad left (or right for negative values) with zero digit
    if width and width > 0:
      while len(ds) < width: ds.insert(0, 0)
    elif width and width < 0:
      while len(ds) < -width: ds.append(0)
    r = join(ds, sep=':', enc="{}")
    width = group = None  # group is just ignored for now
  elif i == 0:
    r = digits[0]
  else:
    r = list()
    while i > 0:
      (i, n) = divmod(i, base)
      r.insert(0, digits[n])
    r = join(r)
  if width is not None:
    if pad is None: pad = digits[0]
    r = (r.rjust(width, pad) if width > 0 else r.ljust(-width, pad))
  if group:
    if group < 0: r = join((join(x) for x in chunk(r, -group)), sep=sep)
    if group > 0: r = rev(join((join(x) for x in chunk(rev(r), group)), sep=rev(sep)))
  return (p + r if p else r)

def base2int(s, base=10, strip=0, digits=None):
  """
  convert a string representation of an integer in the specified base to an integer.

  if <strip> is set, then invalid characters in the conversion are ignored.

  >>> base2int('-42')
  -42
  >>> base2int('xyzzy', base=3, digits='zyx')
  190
  >>> base2int('HELLO', base=36)
  29234652
  """
  if base < 2: raise ValueError("invalid base {base!r}".format(base=base))
  if digits is None: digits = base_digits()
  if len(digits) > base: digits = digits[:base]
  s = str(s)
  if s == digits[0]: return 0
  i = neg = 0
  if s.startswith('-'):
    neg ^= 1
    i += 1
  n = 0
  for k in range(i, len(s)):
    d = s[k]
    try:
      v = digits.index(d)
    except ValueError as e:
      if strip: continue
      e.args = ("invalid digit {d!r} for base {base} in {s!r}".format(base=base, s=s, d=d),)
      raise
    n *= base
    n += v
  return (-n if neg else n)

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

# int2words implementation for lang='en' (English)

_numbers = {
  0: 'zero',
  1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
  10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 15: 'fifteen', 18: 'eighteen',
}

_tens = {
  1: 'teen', 2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'
}

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
    raise ValueError(str.format('Number too large (scale: {scale})', scale=scale))

# from http://en.wikipedia.org/wiki/Names_of_large_numbers
_illions = [
  'm', 'b', 'tr', 'quadr', 'quint', 'sext', 'sept', 'oct',
  'non', 'dec', 'undec', 'duodec', 'tredec', 'quattuordec', 'quindec',
  'sexdec', 'septendec', 'octodec', 'novemdec', 'vigint', 'unvigint',
  'duovigint', 'tresvigint', 'quattuorvigint', 'quinquavigint', 'sesvigint',
  'septemvigint', 'octovigint', 'novemvigint', 'trigint', 'untrigint',
  'duotrigint', 'trestrigint', 'quattuortrigint', 'quinquatrigint', 'sestrigint',
  'septentrigint', 'octotrigint', 'noventrigint', 'quadragint',
]

def __int2words(n, scale='short', sep='', hyphen=' '):
  """
  convert a large integer (one million or greater) to a string
  representing the number in English, using short or long scale.

  >>> __int2words(10**12, scale='short')
  'one trillion'
  >>> __int2words(10**12, scale='long')
  'one billion'
  """
  if scale == 'short':
    (g, p, k) = (3, 1000, 2)
  elif scale == 'long':
    (g, p, k) = (6, 1000000, 1)
  else:
    raise ValueError('Unsupported scale type: ' + scale)
  i = (len(str(n)) - 1) // g
  (d, r) = divmod(n, p**i)
  w = _illions[i - k] + 'illion'
  x = _int2words(d, scale, sep, hyphen) + ' ' + w
  if r == 0: return x
  if r < 100: return x + ' and ' + _int2words(r, scale, sep, hyphen)
  return x + sep + ' ' + _int2words(r, scale, sep, hyphen)

def int2words(n, scale='short', sep='', hyphen=' ', lang='en'):
  """
  convert an integer <n> to a string representing the number (in English).

  scale - 'short' (for short scale numbers), or 'long' (for long scale numbers)
  sep - separator between groups
  hyphen - separator between tens and units
  lang - language (only 'en' (for English) is currently accepted)

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
  >>> sorted(irange(1, 10), key=int2words)
  [8, 5, 4, 9, 1, 7, 6, 10, 3, 2]
  """
  if lang != 'en': raise ValueError(str.format("int2words: lang='{lang}' not implemented", lang=lang))
  return _int2words(int(n), scale, sep, hyphen)

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
  return (r if s == 1 else -r)

# convert a sequence to a string: "(a, b, c)"
def seq2str(s, sort=0, rev=0, enc="()", sep=", "):
  """
  convert a sequence to a string suitable for output.

  >>> seq2str([2, 1, 3])
  '(2, 1, 3)'
  >>> seq2str([2, 1, 3], sort=1)
  '(1, 2, 3)'
  >>> seq2str(first(primes, 10))
  '(2, 3, 5, 7, 11, 13, 17, 19, 23, 29)'
  >>> seq2str(map(seq2str, prime_factor(factorial(15) - 1)))
  '((17, 1), (31, 2), (53, 1), (1510259, 1))'
  """
  if sort: s = sorted(s)
  if rev: s = reversed(s)
  return join(s, sep=sep, enc=enc)

# convert a map to a string: "(a=1, b=2, c=3)"
def map2str(m, sort=1, enc="()", sep=", ", arr="="):
  """
  convert a map to a string suitable for output.

  the map may be a dict() type object or a collection of (key, value)
  pairs.

  >>> map2str(dict(a=1, b=2, c=3))
  '(a=1, b=2, c=3)'
  >>> map2str(multiset("banana"))
  '(a=3, b=1, n=2)'
  >>> map2str(zip("abc", irange(1, 3)))
  '(a=1, b=2, c=3)'
  """
  fn = (sorted if sort else identity)
  if isinstance(m, dict):
    # dict
    return join((concat(k, arr, m[k]) for k in fn(m.keys())), sep=sep, enc=enc)
  else:
    # (k, v) pairs
    return join((concat(k, arr, v) for (k, v) in fn(m)), sep=sep, enc=enc)

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
      x.value (or x()) --> returns expensive(1)
      x.evaluated --> True
      x.value (or x()) --> returns expensive(1) again, without re-evaluating it
      x.reset()
      x.evaluated --> False
      x.value (or x()) --> returns expensive(1), but re-evaluates it
    """
    self.fn = fn
    self.args = args
    self.kw = kw
    self.evaluated = False
    if self.immediate: self.evaluate()

  def __repr__(self):
    return self.__class__.__name__ + '(value=' + (repr(self.value) if self.evaluated else '<delayed>') + ')'

  def evaluate(self):
    self.value = call(self.fn, self.args, self.kw)
    self.evaluated = True
    return self.value

  def __call__(self):
    if not self.evaluated: self.evaluate()
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
  >>> a = Accumulator(fn=add)
  >>> for x in irange(1, 9): a.accumulate(x)
  >>> a.value
  45
  >>> fdiv(a.value, a.count)
  5.0
  """

  def __init__(self, fn=operator.add, fn1=identity, value=None, data=None, collect=0, count=0):
    """
    create an Accumulator.

    The accumulation function and initial value can be specified.
    """
    self.fn = fn # used to accumulate
    self.fn1 = fn1 # used to set initial value
    self.value = value
    if collect and data is None: data = []
    self.collect = collect
    self.data = data
    self.target = None
    self.count = count


  def __repr__(self):
    return self.__class__.__name__ + '(value=' + repr(self.value) + ', data=' + repr(self.data) + ', count=' + str(self.count) + ')'

  def accumulate(self, v=1):
    """
    Accumulate a value.

    If the current value is None then this value replaces the current value.
    Otherwise it is combined with the current value using the accumulation
    function which is called as fn(<current-value>, v).
    """
    self.count += 1
    v0 = self.value
    self.value = (self.fn1(v) if v0 is None else self.fn(v0, v))

  def accumulate_data(self, v, data, target=None):
    """
    Accumulate a value, and check the accumulated value against a target value,
    and if it matches record the data parameter.

    You can use this to record data where some function of the data is at an
    extremum value.

    If the 'collect' parameter was set during initialisation, then all
    values that hit current target are recorded in a list. Otherwise
    only the most recent value is recorded.
    """
    if target is None: target = v
    self.accumulate(v)

    # have we hit the target?
    if self.value == target:
      if self.collect:
        # we need to collect all data values with specified target measure
        if target != self.target:
          # the target has changed, start a new list
          self.data = [data]
          self.target = target
        else:
          # if the target is unchanged, append the current data
          self.data.append(data)

      else:
        # otherwise, just record the data verbatim
        self.data = data

  def accumulate_from(self, s):
    """accumulate values from iterable object <s>"""
    for v in s:
      self.accumulate(v)
    return self

  def accumulate_data_from(self, s, value=0, data=1):
    """
    accumulate values and data from iterable object <s>.

    <value>, <data> can be an index into elements from <s>
    or a function to extract the appropriate value from an element.
    """
    if not callable(value): value = item(value)
    if not callable(data): data = item(data)
    for x in s:
      self.accumulate_data(value(x), data(x))
    return self

  # rs = Accumulator.multi(fns=[min, max])
  @classmethod
  def multi(cls, *args, **kw):
    return MultiAccumulator(*args, **kw)

  # combine multiple Accumulator objects into a MultiAccumulator
  # rs = Accumulator.combine(Accumulator(fn=min), Accumulator(fn=max), ...)
  @classmethod
  def combine(cls, *args):
    self = MultiAccumulator()
    self.multi = rs
    return self

# multiple accumulators: e.g. MultiAccumulator(fns=[min, max])
class MultiAccumulator(object):

  def __init__(self, fns, *args, **kw):
    self.multi = list(Accumulator(fn, *args, **kw) for fn in fns)

  def __repr__(self):
    return self.__class__.__name__ + '(' + repr(self.multi) + ')'

  def perform(self, fn, *args, **kw):
    for x in self.multi:
      fn(x, *args, **kw)

  def accumulate(self, v):
    self.perform(Accumulator.accumulate, v)

  def accumulate_data(self, v, data):
    self.perform(Accumulator.accumulate_data, v, data)

  def accumulate_from(self, s):
    self.perform(Accumulator.accumulate_from, list(s))

  def accumulate_data_from(self, s, value=0, data=1):
    self.perform(Accumulator.accumulate_data_from, list(s), value=value, data=data)

  def __getitem__(self, i):
    return self.multi[i]

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

poly_to_pairs = enumerate

# remove extraneous zero coefficients
def poly_trim(p):
  while len(p) > 1 and p[-1] == 0: p.pop()
  return p

# multiply two polynomials
def poly_mul(p, q):
  (np, nq) = (len(p), len(q))
  if np < nq: (p, q) = (q, p)
  return poly_from_pairs(
    ((i + j, a * b) for (i, a) in enumerate(p) for (j, b) in enumerate(q)),
    [0] * (np + nq - 1)
  )

poly_zero = [0]
poly_unit = [1]

# multiply a sequence of polynomials
def poly_multiply(ps):
  r = poly_unit
  for p in ps:
    r = poly_mul(r, p)
  return r

# make a polynomial with the given roots
def poly_from_roots(rs):
  return poly_multiply([-r, 1] for r in rs)

# raise a polynomial to a (non-negative) integer power
def poly_pow(p, n):
  n = as_int(n, include='0+')
  r = poly_unit
  while n > 0:
    (n, m) = divmod(n, 2)
    if m: r = poly_mul(r, p)
    if n: p = poly_mul(p, p)
  return r

# add two polynomials
def poly_add(p, q):
  return poly_from_pairs(enumerate(p), list(q))

# add a sequence of polynomials
def poly_sum(ps):
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
def poly_divmod(p, q, div=rdiv):
  fn = (identity if q[-1] == 1 else (lambda x: div(x, q[-1])))
  (d, r) = (poly_zero, p)
  while r != poly_zero:
    k = len(r) - len(q)
    if k < 0: break
    m = poly_from_pairs([(k, fn(r[-1]))])
    d = poly_add(d, m)
    r = poly_sub(r, poly_mul(m, q))
  return (d, r)

# compose two polynomials: compose(p, q)(x) = p(q(x))
def poly_compose(p, q):
  r = poly_zero
  m = poly_unit
  for (i, a) in enumerate(p):
    if a:
      r = poly_add(r, list(a * c for c in m))
    m = poly_mul(m, q)
  return r

# print a polynomial in a more friendly form
def poly_print(p, var='x'):
  r = list()
  for (e, c) in enumerate(p):
    if c == 0: continue
    s = str(c)
    if not (c < 0): s = '+' + s
    s = '(' + s + ')'
    if e == 0:
      pass
    elif e == 1:
      s = s + var
    else:
      s = s + var + '^' + str(e)
    r.append(s)
  return join(r[::-1], sep=" ") or "(0)"

# evaluate a polynomial
def poly_value(p, x):
  v = 0
  for n in reversed(p):
    v *= x
    v += n
  return v

# derivative of a polynomial
def poly_derivative(p, k=1):
  for _ in irange(1, k):
    p = poly_from_pairs((e - 1, e * c) for (e, c) in enumerate(p) if e > 0)
  return p

# integral of a polynomial (with constant c)
def poly_integral(p, c=0, div=rdiv):
  k = c
  p = poly_from_pairs((e + 1, div(c, e + 1)) for (e, c) in enumerate(p))
  p[0] = k
  return p

# polynomial interpolation from a number of points
def poly_interpolate(ps, field=None):
  ps = list(ps)
  k = len(ps)
  if k == 0: return None
  if k == 1: return [ps[0][1]]
  k -= 1
  (A, B) = (list(), list())
  for (x, y) in ps:
    A.append([1, x] + [x**i for i in irange(2, k)])
    B.append(y)
  try:
    return poly_trim(Matrix.linear(A, B, field=field))
  except ValueError:
    return

# scale a polynomial to give integer coefficents
def poly_scale(p, F=None):
  if not p: return p
  if F is None: F = Rational()
  p = list(map(F, p))
  m = call(mlcm, (f.denominator for f in p))
  p = list(int(m * f) for f in p)
  g = mgcd(*p)
  if g > 1: p = list(x // g for x in p)
  return p

# find rational roots of a polynomial
# see: [ https://en.wikipedia.org/wiki/Rational_root_theorem ]
def poly_rational_roots(p, domain="Q", include="+-0", F=None):
  """
  find rational roots for the polynomial p (with rational coefficients).

  returns rational values x, such that: p(x) = 0

  the type of roots returned can be controlled with the 'domain' and
  'include' parameters:

    domain='Q' - find rational roots
    domain='Z' - find integer roots

    include='+' - include positive roots
    include='0' - include zero
    include='-' - include negative roots
  """
  if domain not in "QZ": raise ValueError(str.format("invalid domain: {domain!r}", domain=domain))
  if not p: return
  (pos, neg, zero) = (x in include for x in '+-0')
  if F is None: F = Rational()
  # first deal with a root at x=0
  if p[0] == 0:
    if zero: yield (0 if domain == "Z" else F(0, 1))
    while p and p[0] == 0: p = p[1:]
  if not p: return
  # make an equivalent polynomial with integer coefficients
  p = poly_scale(p, F=F)
  fs = product(divisors(abs(p[0])), divisors(abs(p[-1])))
  for x in uniq(map(unpack(F), fs)):
    if domain == "Z":
      x = as_int(x, default=None)
      if x is None: continue
    if pos and poly_value(p, x) == 0:
      yield x
    if neg and poly_value(p, -x) == 0:
      yield -x

# return (n, r) where p = q^n . r
def poly_div(p, q, div=rdiv):
  n = 0
  while True:
    (p_, z) = poly_divmod(p, q, div=div)
    if z != poly_zero: break
    n += 1
    p = p_
  return (n, p)

# EXPERIMENTAL: return factors of polynomial <p> using Kroneker's method
def poly_factor(p, F=None, div=None):
  if F is None: F = Rational()
  if div is None: div = Rdiv(F)

  # first find factors of x
  (n, p) = (0, list(p))
  while p[0] == 0:
    n += 1
    p.pop(0)
  if n > 0: yield ([0, 1], n)

  # find other linear factors
  for x in poly_rational_roots(p, domain='Q', F=F):
    f = [-int(x.numerator), int(x.denominator)]
    (n, p) = poly_div(p, f, div=div)
    if n > 0: yield (f, n)

  # look for factors of degree k
  k = 2
  while len(p) > 2 * k:
    q = poly_scale(p)
    # evaluate q at (k + 1) values, and record the divisors
    ds = list()
    for i in irange(0, k):
      vs = divisors(poly_value(q, i))
      if i > 0: vs += list(-x for x in vs)
      ds.append(vs)
    # choose potential values for polynomial factor at the values
    for vs in cproduct(ds):
      if mgcd(*vs) != 1: continue
      # interpolate the polynomial factor
      f = poly_interpolate(enumerate(vs), field=F)
      if len(f) - 1 < k: continue
      (n, p) = poly_div(p, f, div=div)
      if n > 0:
        yield (f, n)
        break
    else:
      k += 1

  # anything left?
  if len(p) > 1:
    f = poly_scale(p)
    yield (f, 1)
    p = [div(p[-1], f[-1])]
  if p != poly_unit:
    yield (p, 1)

# drop factors in qs from polynomial p
def poly_drop_factor(p, *qs):
  for q in qs:
    while True:
      (d, r) = poly_divmod(p, q)
      if r != poly_zero: break
      p = d
  return p

# calculate cyclotomic polynomials
@static(cache={1: [-1, 1]}, cache_enabled=1)
def poly_cyclotomic(n, fs=None, div=rdiv, fn=prime_factor):
  """
  return the nth cyclotomic polynomial

  >>> poly_cyclotomic(7)
  [1, 1, 1, 1, 1, 1, 1]
  >>> poly_cyclotomic(12)
  [1, 0, -1, 0, 1]
  >>> poly_cyclotomic(30)
  [1, 1, 0, -1, -1, -1, 0, 1, 1]
  """
  if n < 1: return None
  r = poly_cyclotomic.cache.get(n)
  if r is None:
    if fs is None: fs = list(fn(n))
    if len(fs) == 1:
      (p, e) = fs[0]
      if e == 1:
        # n is prime
        r = [1] * n
      else:
        # power of a prime
        q = n // p
        r = poly_from_pairs((k * q, 1) for k in irange(0, p - 1))
    elif fs[0] == (2, 1):
      # 2n, invert the odd positions in cyclotomic[n]
      r = list(poly_cyclotomic(n // 2, fs=fs[1:], div=div, fn=fn))
      for i in range(1, len(r), 2): r[i] = -r[i]
    else:
      # C[n] = multiply((x^d - 1) ^ mobius(n // d) for d in divisors(n))
      # we can specialise multiplication and division by (x^d - 1)
      (r, ds) = ([1], [])
      for d in multiples(fs):
        m = mobius(n // d, fn=fn)
        if m == 1:
          # r *= (x^d - 1)
          r = poly_sub([0] * d + r, r)
        elif m == -1:
          ds.append(d)
      for d in ds:
        # r /= (x^d - 1)
        r_ = list()
        for c in reversed(r[d:]):
          r_.insert(0, c)
          if len(r_) > d: r_[0] += r_[d]
        r = r_
    if poly_cyclotomic.cache_enabled: poly_cyclotomic.cache[n] = r
  return r


# wrap the whole lot up in a class

class Polynomial(list):
  """
  A class for manipulating polynomials in one variable.

  Polynomials are represented by a list of their coefficents:

    a + b.x + c.x^2 + d.x^3 + ... ->  [a, b, c, d, ...]
  """

  def __repr__(self, var='x'):
    return self.__class__.__name__ + "[" + poly_print(self, var=var) + "]"

  def print(self, var='x'):
    return poly_print(self, var=var)

  def __hash__(self):
    return hash(tuple(self))

  def __add__(self, other):
    if not isinstance(other, Polynomial): other = Polynomial([other])
    return self.__class__(poly_add(self, other))

  # this allows: <non-poly> + <poly> (e.g. 3 + p)
  __radd__ = __add__

  def __iadd__(self, other):
    return self + other

  def __mul__(self, other):
    if isinstance(other, Polynomial):
      # multiply polynomials
      return self.__class__(poly_mul(self, other))
    else:
      # multiply coefficients
      return self.__class__(other * c for c in self)

  # this allows: <non-poly> * <poly> (e.g. 3 * p)
  __rmul__ = __mul__

  def __neg__(self):
    return self.__class__(poly_neg(self))

  def __pow__(self, n):
    return self.__class__(poly_pow(self, n))

  def __sub__(self, other):
    if not isinstance(other, Polynomial): other = Polynomial([other])
    return self.__class__(poly_sub(self, other))

  # this allows: <non-poly> - <poly> (e.g. 3 - p)
  __rsub__ = lambda self, other: -self + other

  __call__ = poly_value

  def copy(self):
    "return a copy of the polynomial"
    return self.__class__(self)

  def degree(self):
    "return the degree of the polynomial"
    return len(self) - 1

  def coeff(self, k, default=0):
    "return the coefficient of x^k in the polynomial"
    if 0 <= k < len(self):
      return self[k]
    else:
      return default

  def to_pairs(self):
    "an iterator that returns (<exponent>, <coefficient>) pairs of the polynomial"
    for p in enumerate(self):
      if p[1] != 0:
        yield p

  def is_zero(self):
    "check if this polynomial is zero: p(x) = 0"
    return self == poly_zero

  def is_unit(self):
    "check if this polynomial is the unit polynomial: p(x) = 1"
    return self == poly_unit

  def map(self, fn):
    """
    return a polynomial that is the result of applying <fn> to the
    coefficents in this polynomial
    """
    return self.__class__(poly_map(self, fn))

  def scale(self):
    """
    return a polynomial with integer coefficients derived from this
    polynomial by multiplying the coefficients by a scalar value
    """
    return self.__class__(poly_scale(self))

  def derivative(self, k=1):
    "return the derivative of the polynomial"
    return self.__class__(poly_derivative(self, k))

  def integral(self, c=0, div=rdiv):
    "return the indefinite integral of the polynomial"
    return self.__class__(poly_integral(self, c=c, div=div))

  def quadratic_roots(self, v=0, domain="Q", include="+-0", F=None):
    p = (self - v if v else self)
    if p.degree() > 2: raise ValueError("polynomial degree too large")
    return quadratic(p[2], p[1], p[0], domain=domain, include=include, F=F)

  def rational_roots(self, v=0, domain="Q", include="+-0", F=None):
    "find rational roots of the polynomial = v"
    p = (self - v if v else self)
    return poly_rational_roots(p, domain=domain, include=include, F=F)

  def divmod(self, q, div=rdiv):
    (d, r) = poly_divmod(self, q, div=div)
    return (self.__class__(d), self.__class__(r))

  def compose(self, other):
    "return a polynomial which is the result of the applying this polynomial to another"
    return self.__class__(poly_compose(self, other))

  def factor(self, F=None, div=None):
    "generate factors of the polynomial"
    for (f, n) in poly_factor(self, F=F, div=div):
      yield (self.__class__(f), n)

  def drop_factor(self, *qs):
    return self.__class__(poly_drop_factor(self, *qs))

  @classmethod
  def from_pairs(cls, ps):
    return cls(poly_from_pairs(ps))

  @classmethod
  def from_roots(cls, rs):
    return cls(poly_from_roots(rs))

  @classmethod
  def unit(cls):
    return cls(poly_unit)

  @classmethod
  def zero(cls):
    return cls(poly_zero)

  # sum() is only documented for "numeric" values (although it works)
  # but you can use this instead...
  @classmethod
  def sum(cls, ps):
    "return the sum of a sequence of polynomials"
    return cls(poly_sum(ps))

  @classmethod
  def multiply(cls, ps):
    "return the product of a sequence of polynomials"
    return cls(poly_multiply(ps))

  @classmethod
  def interpolate(cls, ps, field=None):
    "return a polynomial that fits the (x, y) values in <ps>"
    r = poly_interpolate(ps, field=field)
    return (None if r is None else cls(r))

  @classmethod
  def cyclotomic(cls, n, div=rdiv, fn=prime_factor):
    "return the nth cyclotomic polynomial"
    p = poly_cyclotomic(n, div=div, fn=fn)
    return (None if p is None else cls(p))

###############################################################################

# Prime Sieves

_primes_array = bytearray
_primes_size = 1295
_primes_chunk = lambda n: 2 * n
_primes_array_optimise = 1  # turn this off to disable bitarray optimisations

class _PrimeSieveE6(object):
  """
  A prime sieve.

  The 'array' parameter can be used to specify a list like class to implement
  the sieve. Possible values for this are:

  list - use standard Python list
  bytearray - faster and uses less space (default)
  bitarray - (if you have it) less space that bytearray, but more time than list

  >>> _PrimeSieveE6(50).contents()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  >>> primes = _PrimeSieveE6(1000000)
  >>> primes.is_prime(10001)
  False
  >>> 10007 in primes
  True
  >>> sum(primes) == 37550402023
  True
  >>> list(primes.irange(2, 47))
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
  # p->i = p // 3 (or in general: p->i = (p + 1) // 3 - (p % 6 == 5))
  #
  # to check numbers up to (but not including) n we need a sieve of size: (n // 3) + (n % 6 == 2)

  def __init__(self, n, array=_primes_array, verbose=0):
    "make a sieve of primes up to <n>"
    # initial sieve
    self.sieve = array([0])
    self.max = 1
    self.num = None # record the number of primes
    # return n copies of True or False values
    self.T = (lambda n, v=array([1]): v * n) # used to extend the array
    self.F = (lambda n, v=array([0]): v * n) # used to exclude non-primes
    if _primes_array_optimise:
      if array.__name__ == 'bitarray':
        # bitarray can set all values in a slice to the same boolean value
        if verbose: printf("[{x}: optimising for array={a}]", x=self.__class__.__name__, a=array.__name__)
        self.F = (lambda n: 0)
    # other parameters
    self.expandable = 0  # set to 1 to allow automatic expansion
    self.verbose = verbose
    # now extend the sieve to the required size
    self.extend(n)

  def __repr__(self):
    return self.__class__.__name__ + '(max=' + repr(self.max) + ')'

  def extend(self, n):
    """
    extend the sieve up to (at least) <n>
    """
    if not (n > self.max): return
    #if self.verbose: printf("[{x}: expanding to {n}]", x=self.__class__.__name__)

    # extend the sieve to the right size
    s = self.sieve
    lo = len(s) + 1
    hi = ((n + 1) // 3) + (n % 6 == 1)
    s.extend(self.T(hi - lo + 1))

    # remove multiples of primes p from indices lo to hi
    for i in irange(1, isqrt(n) // 3):
      if s[i]:
        odd = (i & 1)
        p = (i * 3) + odd + 1
        k = 2 * p
        # remove multiples of p starting from p^2
        j = (p * p) // 3
        if j < lo: j += k * ((lo - j) // k)
        s[j::k] = self.F((hi - j - 1) // k + 1)
        ##printf("eliminating {p} {ns}", ns=list((z * 3) + (z & 1) + 1 for z in irange(j, hi-1, step=k)))
        # remove multiples with the other residue
        q = p + (2 if odd else 4)
        j = (p * q) // 3
        if j < lo: j += k * ((lo - j) // k)
        s[j::k] = self.F((hi - j - 1) // k + 1)
        ##printf("eliminating {p} {ns}", ns=list((z * 3) + (z & 1) + 1 for z in irange(j, hi-1, step=k)))

    self.max = n
    self.num = None
    if self.verbose:
      if s.__class__.__name__ == 'bitarray' and hasattr(s, 'buffer_info'): b = s.buffer_info()[1]
      elif hasattr(s, '__sizeof__'): b = s.__sizeof__()
      else: b = ''
      if b: b = sprintf(" ({b} bytes used)")
      printf("[{x}: expanded to {n}{b}]", x=self.__class__.__name__)

  # return the contents of the sieve (more space) [used to be called list()]
  def contents(self, fn=list):
    """
    return a collection of primes in the sieve (default is a list
    in numerical order).

    this will require more memory than using generate().
    """
    return fn(_PrimeSieveE6.generate(self))

  # return a generator (less space)
  def generate(self, start=0, end=None):
    """
    generate primes in the sieve (in numerical order).

    the range of primes can be restricted to starting at <start>
    and ending at <end> (primes less than <end> will be returned)

    (this will require less memory than contents())
    """
    if end is None or end > self.max: end = self.max + 1
    if start < 3 and end > 2: yield 2
    if start < 4 and end > 3: yield 3
    s = self.sieve
    # generate primes from <start> up to (but not including) <end>
    for i in xrange((start + 1) // 3 - (start % 6 == 5), (end + 1) // 3 - (end % 6 == 5)):
      if s[i]: yield (i * 3) + (i & 1) + 1

  # make this an iterable object
  __iter__ = generate

  # xrange(a, b) - generate primes in the range [a, b) - is the same as generate() now
  xrange = generate
  range = generate

  # irange = inclusive range
  def irange(self, a, b):
    if not (b is None or b == inf): b += 1
    return self.generate(a, b)

  # generate tuples of primes from the sieve (in the interval [a, b])
  def tuples(self, k, a=0, b=inf):
    # consider increasing values for the largest prime
    for p in self.irange(a, b):
      if k == 1:
        yield (p,)
      elif k > 1:
        # generate the smaller primes
        for ps in self.tuples(k - 1, a=a, b=p - 1):
          yield ps + (p,)

  # prime test (may throw IndexError if n is too large)
  def is_prime(self, n):
    """
    check to see if the number is a prime.

    (may throw IndexError for numbers larger than the sieve).
    """
    if n < 2: return False # 0, 1 -> F
    if n < 4: return True # 2, 3 -> T
    r = n % 6
    if r != 1 and r != 5: return False # (n % 6) != (1, 5) -> F
    if self.expandable: self.expand(n)
    return bool(self.sieve[n // 3])

  prime = is_prime

  # allows use of "in"
  __contains__ = is_prime

  # before, after: return the prime immediately before/after n
  def before(self, n):
    """
    return the largest prime less than <n>
    """
    if n < 3: return None
    if n < 4: return 2
    if n < 6: return 3
    if self.expandable: self.expand(n)
    i = (n + 1) // 3 - (n % 6 == 5)
    while True:
      i -= 1
      if self.sieve[i]: return (i * 3) + (i & 1) + 1

  def after(self, n):
    """
    return the smallest prime greater than <n>
    """
    if n < 2: return 2
    if n < 3: return 3
    i = (n + 1) // 3 + (n % 6 == 1)
    while True:
      if self.expandable and not (i < len(self.sieve)): self.expand()
      if self.sieve[i]: return (i * 3) + (i & 1) + 1
      i += 1

  def between(self, a, b, fn=None):
    """
    return primes in [a, b]
    """
    if self.expandable: self.extend(b)
    r = self.irange(a, b)
    return (r if fn is None else fn(r))

  # size = number of primes (currently) in the sieve
  def size(self):
    if self.num is None: self.num = icount(self.generate(0, self.max + 1))
    return self.num

  __len__ = size

  # generate prime factors of <n> using the sieve
  # (try setting mr=100 if checking large numbers)
  # (or mr=inf to perform all heuristic tests after the sieve is exhausted)
  def prime_factor(self, n, end=None, mr=0, mrr=0):
    """
    generate (<prime>, <exponent>) pairs in the prime factorisation of
    positive integer <n>, for primes in the sieve (less than <end>).

    if <mr> is set the program will use a Miller-Rabin probabilistic
    test after <mr> primes have failed to divide the residue to see if
    it is prime, and after the primes in the sieve are exhausted the
    Pollard Rho algorithm is used to look for remaining large prime
    factors.

    Note: By default this will only return primes up to the limit of
    the sieve, so may not be a complete factorisation of <n>.  However
    when <mr> is set it will also attempt to look for larger
    probabalistic prime factors.
    """
    if n == 1: return ()
    return prime_factor_h(n, self, end=end, nf=mr, mr=mr, mrr=mrr)

  # functions that can use self.prime_factor() instead of simple prime_factor()

  # return a list of the factors of n
  def factor(self, n, end=None, mr=0, mrr=0):
    """
    return a list of the prime factors of positive integer <n>.

    Note: This will only consider primes up to the limit of the sieve,
    this is a complete factorisation for <n> up to the square of the
    limit of the sieve.
    """
    return factor(n, fn=(lambda n: self.prime_factor(n, end=end, mr=mr, mrr=mrr)))

  def divisors(self, n, end=None, mr=0, mrr=0):
    return divisors(n, fn=(lambda n: self.prime_factor(n, end=end, mr=mr, mrr=mrr)))

  def divisors_pairs(self, n, end=None, mr=0, mrr=0):
    return divisors_pairs(n, fn=(lambda n: self.prime_factor(n, end=end, mr=mr, mrr=mrr)))

  def tau(self, n, end=None, mr=0, mrr=0):
    return tau(n, fn=(lambda n: self.prime_factor(n, end=end, mr=mr, mrr=mrr)))

  def is_square_free(self, n, end=None, mr=0, mrr=0):
    return is_square_free(n, fn=(lambda n: self.prime_factor(n, end=end, mr=mr, mrr=mrr)))

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
  def __init__(self, n, array=_primes_array, fn=_primes_chunk, verbose=0):
    """
    make a sieve of primes with an initial maximum of <n>.

    when the sieve is expanded the function <fn> is used to calculate
    the new maximum, based on the previous maximum.

    the default function doubles the maximum at each expansion.
    """
    _PrimeSieveE6.__init__(self, n, array=array, verbose=verbose)
    self.chunk = fn
    self.expandable = 1

  # expand the sieve up to n, or by the next chunk
  def extend(self, n=None):
    """
    extend the sieve to include primes up to (at least) n.

    if n is not specified that sieve will be expanded according to the
    function specified in __init__().
    """
    if n is None: n = self.chunk(self.max)
    _PrimeSieveE6.extend(self, n)
    return self

  # for backwards compatibility
  expand = extend

  # generate all primes, a chunk at a time
  # end = inf (or None), will expand the sieve for ever
  # end = self.max, will not expand the sieve
  def generate(self, start=0, end=inf):
    """
    generate primes without limit, expanding the sieve as necessary.

    eventually the sieve will consume all available memory.
    """
    if end is None: end = inf
    while start < end:
      # generate all primes currently in the sieve
      for p in _PrimeSieveE6.generate(self, start, end):
        yield p
      # expand the sieve for the next batch
      start = max(start, self.max + 1)
      if start < end: self.expand()

  # make this an iterable object
  __iter__ = generate

  # expand the sieve as necessary
  def range(self, a=0, b=None):
    """
    generate primes in the (inclusive) range [a, b].

    the sieve is expanded as necessary beforehand.
    """
    # have we asked for unlimited generation?
    if b is None or b == inf: return self.generate(a)
    # otherwise, upper limit is provided
    self.extend(b)
    return _PrimeSieveE6.range(self, a, b)

# create a suitable prime sieve
def Primes(n=None, expandable=0, array=_primes_array, fn=_primes_chunk, verbose=0):
  """
  Return a suitable prime sieve object.

  n - initial limit of the sieve (the sieve contains primes up to <n>)
  expandable - should the sieve expand as necessary
  array - list implementation to use
  fn - function used to increase the limit on expanding sieves

  If we are interested in a limited collection of primes, we can do
  this:

  >>> primes = Primes(50)
  >>> primes.contents()
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
  >>> primes.contents()
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

  But if we use it as a generator it will expand indefinitely, so we
  can only sum a restricted range:
  >>> sum(primes.range(0, 100))
  1060

  If you don't know how many primes you'll need you can just use
  Primes() and get an expandable sieve with primes up to 1024, and the
  limit will double each time the sieve is expanded.

  So, to sum the first 1000 primes:
  >>> sum(first(primes, 1000))
  3682913
  """
  # if n is None then make it expandable by default
  if n is None: (n, expandable) = (_primes_size, True)
  # return an appropriate object
  if expandable:
    return _PrimeSieveE6X(n, array=array, fn=fn, verbose=verbose)
  else:
    return _PrimeSieveE6(n, array=array, verbose=verbose)

# backwards compatibility
def PrimesGenerator(n=None, array=_primes_array, fn=_primes_chunk):
  "provided for backwatds compatability. use Primes() instead."
  return Primes(n, expandable=1, array=array, fn=fn)

# default expandable sieve
primes = Primes(1, expandable=1, array=_primes_array, fn=(lambda n: _primes_size if n < _primes_size else 2 * n))

###############################################################################

# Magic Square Solver:

deepcopy = lazy_import('copy.deepcopy')

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
      l = tuple(xrange(n))
      for i in xrange(n):
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
    """set the value of a square (linearly indexed from 0)"""
    self.square[i] = v
    self.numbers.remove(v)

  def get(self, i):
    """get the value of a square (linearly indexed from 0)"""
    return self.square[i]

  def output(self):
    """print the magic square"""
    m = max(self.square)
    n = (int(math.log10(m)) + 1 if m > 0 else 1)
    fmt = "[{:>" + str(n) + "s}]"
    for y in xrange(self.n):
      for x in xrange(self.n):
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
    return deepcopy(self)

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

# NOTE: SubstitutedExpression.split_sum() can do all this, and more, faster.

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

  if there are braces present in <text> then only those portions of the
  <text> enclosed in braces are substituted.

  >>> substitute(dict(zip('DEMNORSY', (7, 5, 1, 6, 0, 8, 9, 2))), "SEND + MORE = MONEY")
  '9567 + 1085 = 10652'

  """
  if text is None: return None
  if digits is None: digits = base_digits()
  return translate(text, (lambda x: digits[s2d[x]] if x in s2d else x))

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
  # check there aren't too many letters (we could issue a warning)
  words = list(terms)
  words.append(result)
  if len(union(words)) > base: return
  # fill out the parameters
  if l2d is None: l2d = dict()
  if digits is None: digits = xrange(base)
  digits = set(digits).difference(l2d.values())
  if d2i is None: d2i = { 0: set(w[0] for w in words if len(w) > 1) }
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
  Note: See SubstitutedExpression.split_sum() for a more powerful solver.

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
      if check and (not check(s)): continue
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
      s=map2str(s, sep=" ", enc="")
    )

  solution = output_solution

  def run(self, check=None, first=0):
    """find all solutions (matching the filter <check>) and output them"""
    for s in self.solve(check=check, verbose=1):
      if first: break

  # backwards compatability
  go = run

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
  def chain_run(cls, sums, base=10, digits=None, l2d=None, d2i=None):
    template = join(('(' + join(s[:-1], sep=' + ') + ' = ' + s[-1] + ')' for s in sums), sep=' ')
    printf("{template}")
    for s in cls.chain(sums, base=base, digits=digits, l2d=l2d, d2i=d2i):
      printf("{t} / {s}",
        t=substitute(s, template),
        s=map2str(s, sep=" ", enc="")
      )

  # backwards compatability
  chain_go = chain_run

  # class method to call from the command line
  @classmethod
  def run_command_line(cls, args):
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
      except Exception:
        printf("{cls.__name__}: invalid option: {arg}")
        return -1

    # check command line usage
    if not args:
      print(usage)
      return -1

    # extract the sums
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
#
# the experimental [[ denest=1 ]] parameter will produce less nested code
# to work around this issue.

# TODO: think about negative values - expressions resulting in an
# alphametic word must be non-negative
#
# TODO: consider ordering the symbols, so we can calculate words sooner.
#
# TODO: consider allowing a "wildcard" character, for symbols that can take
# on any available digit (but still not allow leading zeros). [Enigma 1579]
#
# TODO: consider allowing code to generate possible values for a word
# like: "[possible values] -> WORD", (so [Teaser 3019] would be:
# [ "primes.irange(1000, 9999) -> WILL", "primes.irange(10, 99) -> AM" ], or
# [Teaser 3018] could use [ "powers(31, 99, 2) -> ABCD" ])
#
# TODO: spotting expressions for independent groups and solving
# each group separately [Teaser 2990]

# find words in string <s>
def _find_words(s, r=1):
  words = set(re.findall(r'{(\w+?)}', s)) # re.UNICODE
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

# local variable used to represent symbol x:
# default is just _x
_sym = lambda x: '_' + x
# this will use ascii variables 'v_<hex-code-of-symbol>'
#_sym = cached(lambda x: 'v_' + join((int2base(ord(c), base=16) for c in x), sep="_"))

_set = lambda x: join(x, sep=", ", enc="{}")

# return an expression that evaluates word <w> in base <base>
def _word(w, base):
  (m, d) = (1, dict())
  for x in w[::-1]:
    d[x] = d.get(x, 0) + m
    m *= base
  return join((concat((_sym(k),) + (() if v == 1 else ('*', v))) for (k, v) in d.items()), sep=' + ')

@static(i=0)
def gensym(x):
  """
  generate a unique string starting with <x>.

  >> gensym('foo')
  'foo1'
  >> gensym('foo')
  'foo2'
  """
  gensym.i += 1
  return concat(x, gensym.i)

# file.writelines does NOT include newline characters
def writelines(fh, lines, sep=None, flush=1):
  if sep is None: sep = os.linesep
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
    s = re.sub(encl(re.escape(sep[1:]), '[]'), d, s)
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

# check for explicit "{WORD}" parameters
_is_explicit = lambda s: re.search(r'{\w+}', s)  # was: [[ ('{' in s) ]]

# fix up implicit parameters
# if <s> contains no {word} parameters, then enclose words from <symbols>
def _fix_implicit(s, symbols):
  if s is None: return None
  if _is_explicit(s): return s
  return re.sub(encl(symbols, '[]') + '+', (lambda m: encl(m.group(0))), s)

def _fix_implicit_seq(seq, symbols):
  if seq is None: return None
  return list(_fix_implicit(x, symbols) for x in seq)

def _expand_macros(s, macro, depth=20):
  if not s: return s
  while '@' in s:
    if depth == 0: raise ValueError("macro depth exceeded")
    try:
      s = re.sub(r'\@(\w+)', (lambda x: macro[x.group(1)]), s)
    except KeyError as e:
      x = ValueError(str.format("invalid macro: {k!r}", k=e.args[0]))
      # [Python 3] raise ... from None
      if hasattr(x, '__cause__'): x.__cause__ = None
      raise x
    depth -= 1
  return s

class SubstitutedExpression(object):
  """
  A solver for Python expressions with symbols substituted for numbers.

  It takes a Python expression and then tries all possible ways of assigning
  symbols (by default the capital letters) in it to digits and returns those
  assignments which result in the expression having a True value.

  This allows for more general expressions to be evaluated than specialised
  solvers, like SubstitutedSum(), allow.


  Enigma 1530 <https://enigmaticcode.wordpress.com/2012/07/09/enigma-1530-tom-daley/>
  >>> SubstitutedExpression('TOM * 13 = DALEY').run().n
  (TOM * 13 = DALEY)
  (796 * 13 = 10348) / A=0 D=1 E=4 L=3 M=6 O=9 T=7 Y=8
  [1 solution]
  1


  See SubstitutedExpression.run_command_line() for more examples.
  """

  @classmethod
  def set_default(cls, **kw):
    """
    set default values for instance initialisation.
    """
    for (k, v) in kw.items():
      cls.defaults[k] = v

  # standard default values
  # add new parameters here, and they will be set up automatically
  # but you still need to document them in __init__
  defaults = dict(
    # parameters that have a default value
    base=10, distinct=1, process=1, reorder=1, first=0, denest=0, sane=1, warn=0, opt='', verbose=1,
    # other parameters
    exprs=None, symbols=None, digits=None, s2d=None, d2i=None, answer=None, accumulate=None,
    literal=None, template=None, solution=None, header=None,
    check=None, macro=None, env=None, code=None, decl=None
  )

  def __init__(self,
    exprs, base=None, symbols=None, digits=None, s2d=None, l2d=None, d2i=None, answer=None,
    accumulate=None, literal=None, template=None, solution=None, header=None, distinct=None,
    check=None, macro=None, env=None, code=None, process=None, reorder=None, first=None,
    denest=None, decl=None, sane=None, warn=None, opt=None, verbose=None
  ):
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
    distinct - symbols which should have distinct values (1 = all, 0 = none) (default: 1)
    literal - symbols which stand for themselves (e.g. "012") (default: None)
    answer - an expression for the answer value
    accumulate - accumulate answers using the specified object
    check - a boolean function used to accept/reject solutions (default: None)
    env - additional environment for evaluation (default: None)
    code - additional lines of code evaluated before solving (default: None)
    macro - macro expansions applied to expressions containing @macro (default: None)
    denest - work around CPython statically nested block limit
    decl - additional declarations used in functions generated when denest is enabled
    sane - enable/disable sanity checks (default: 1)
    verbose - control informational output (default: 1)

    If you want to allow leading digits to be 0 pass an empty dictionary for d2i.
    """

    # return the first not-None value, or defaults[key]
    def get_default(key, *values):
      for v in values:
        if v is not None: return v
      return self.__class__.defaults.get(key, None)

    # set defaults from class, most are simple except:
    # s2d - use l2d if s2d is not specified (for backward compatability)
    scope = locals()
    for k in self.__class__.defaults.keys():
      if k == 's2d':
        v = get_default(k, s2d, l2d)
      else:
        v = get_default(k, scope[k])
      setattr(self, k, v)

    self._processed = 0 # set by process
    self._prepared = 0 # set by prepare()

    if self.process: self._process()


  # verbose flags:
  # generate output
  vH = 4  # output header template
  vT = 8  # output solutions (from template)
  #vS = 512  # output solution symbol -> digit mapping
  vA = 16  # output answer counts / accumulated measures
  # information / debugging
  vE = 32  # output elapsed time
  vP = 64  # output solver parameters
  vI = 128  # output solver info
  vC = 256  # output code (before compilation)
  vW = 1024  # output warnings
  # standard debug levels
  v0 = 0  # 0 = no output
  v1 = vH | vT | vA | vW  # 1 = header + solutions + count
  v2 = v1 | vI  # 2 = 1 + solver info
  v3 = v2 | vE | vC  # 3 = 2 + timing + code
  v9 = vH | vT | vA | vE | vP | vI | vC | vW # 9 = everything

  def _verbose(self, n):
    if not n: return 0
    # if it is a string, parse it into a number
    if isinstance(n, basestring):
      # sort out "new style" flags, e.g. "1+E" "1-T" "1-T+E" "9" "HTAEPIC"
      d = dict((k, getattr(self, 'v' + k)) for k in 'HTAEPICW01239')
      v = d['0']
      try:
        op = '+'
        for k in n:
          if k in '+-':
            op = k
            continue
          x = d[k]
          if op == '+':
            v |= x
          elif op == '-':
            v &= ~x
            op = '+'
        return v
      except Exception:
        pass

      # sort out "old-style" numeric arguments
      if re.match(r'[\d\-\|\+\,]+', n):
        n = sum(_digits(n))

    # old style verbose flags (1, 2, 3)
    if n < 4:
      return (0, self.v1, self.v2, self.v3)[n]
    # otherwise
    return n

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
    literal = self.literal
    denest = self.denest
    decl = self.decl
    macro = self.macro
    process = self.process
    sane = self.sane
    warn = self.warn
    opt = self.opt
    verbose = self.verbose

    # sort out verbose argument
    verbose = self._verbose(verbose)

    if verbose & self.vW:
      if opt:
        xs = list(v for (k, v) in {'i': 'iterators', 's': 'sets'}.items() if k in opt)
        if xs:
          printf("WARNING: enabling experimental options: {xs}", xs=join(xs, sep=", "))
      if sane == 0 :
        printf("WARNING: sanity checks disabled - good luck!")

    # the symbols to replace (for implicit expressions)
    if symbols is None: symbols = str_upper
    symbols = set(symbols)
    if s2d: symbols.update(s2d.keys())
    if literal: symbols.update(literal)
    symbols = join(sorted(symbols))

    # process expr to be a list of (<expr>, <value>) pairs, where:
    # <value> is:
    # None = look for a true value
    # word = look for a value equal to the substituted word
    # integer = look for the specific value
    if process:

      # allow expr to be a single string
      if isinstance(exprs, basestring): exprs = [exprs]

      # replace any macros within expressions (and answer, template)
      exprs = list(_expand_macros(s, macro) for s in exprs)
      answer = _expand_macros(answer, macro)
      template = _expand_macros(template, macro)

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
        if symbols and not (_is_explicit(expr) or (isinstance(v, basestring) and _is_explicit(v))):
          expr = _fix_implicit(expr, symbols)
          if isinstance(v, basestring): v = _fix_implicit(v, symbols)

        # value is either an alphametic or a numeric literal
        if isinstance(v, basestring) and not (v[0] == '{' and v[-1] == '}'):
          v = base2int(v, base=base)

        xs.append((expr, v))

      exprs = xs

      # fix up implicit (old style) parameters
      answer = _fix_implicit(answer, symbols)
      template = _fix_implicit(template, symbols)

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
    _template = join(ts, fn=(lambda t: encl(t, '()')), sep=' ')
    exprs = xs

    # initial mapping of symbols to digits
    if s2d is None: s2d = dict()

    # literal values are symbols which stand for themselves
    if literal is None: literal = ()
    for s in literal:
      s2d[s] = base2int(s, base=base)

    # allowable digits (and invalid digits)
    if digits is None:
      digits = set(xrange(base))
    else:
      digits = set(digits)
      if sane > 0:
        ds = set(xrange(base))
        if verbose & self.vW:
          # check for invalid digits
          for d in digits:
            if d not in ds:
              printf("WARNING: SubstitutedExpression: non-valid digit {d} for base {base} specified", d=repr(d))
        digits.intersection_update(ds)
    # TODO: I suspect this needs to work with more values of "distinct"
    # if all values are distinct (including literals), remove them from digits
    if distinct == 1: digits = digits.difference(s2d.values())
    idigits = set(xrange(base)).difference(digits)

    # find words in all exprs
    words = _find_words(_template)
    # and determine the symbols that are used
    symbols = join(sorted(set().union(*words)))

    # invalid (<symbol>, <digit>) assignments
    invalid = set()
    if d2i is not None:
      # it should provide a sequence of (<digit>, <symbol[s]>) pairs
      for (d, ss) in (d2i.items() if hasattr(d2i, 'items') else d2i):
        if sane > 0 and verbose & self.vW and d not in digits and d not in idigits:
          printf("WARNING: SubstitutedExpression: non-valid invalid digit {d} specified", d=repr(d))
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
    if isinstance(distinct, int): distinct = (symbols if distinct else '')
    # distinct should be a sequence (probably of strings)
    if isinstance(distinct, basestring): distinct = [distinct]

    # add the value of the symbols into the template
    self.template = (_template if template is None else template)
    if self.solution is None: self.solution = join(diff(symbols, literal))
    if self.header is None: self.header = _replace_words(self.template, identity)

    # sort out negative value for denest, will not be enabled if running under PyPy
    if denest < 0: denest = (0 if _pypy else -denest)

    # sort out denest=1
    if denest == 1: denest = 50

    # update the processed values
    self.exprs = exprs
    self.symbols = symbols
    self.digits = digits
    self.s2d = s2d
    self.answer = answer
    self.distinct = distinct
    self.literal = literal
    self.denest = denest
    self.decl = decl
    self.verbose = verbose
    self._words = words
    self._invalid = invalid
    self._idigits = idigits
    self._exprs = (exprs, xs, vs, ts, syms)
    self._fail = 0
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
    literal = self.literal
    env = self.env
    code = self.code
    reorder = self.reorder
    denest = self.denest
    decl = self.decl
    sane = self.sane
    warn = self.warn
    opt = self.opt
    verbose = self.verbose

    words = self._words
    invalid = self._invalid
    idigits = self._idigits
    (exprs, xs, vs, ts, syms) = self._exprs

    # output run parameters
    if self.verbose & self.vP:
      print("-- [code] --" + nl + join(self.save(quote=1), sep=nl) + nl + "-- [/code] --")

    # # remove assigned symbols from distinct groups [suggested by Frits]
    # if s2d:
    #   # update invalid pairs
    #   for (k, v) in s2d.items():
    #     for ds in distinct:
    #       if k in ds:
    #         invalid.update((d, v) for d in ds if d != k)
    #   # update distinct
    #   distinct = list(join(x) for x in (diff(x, s2d.keys()) for x in distinct) if len(x) > 1)

    # valid digits for each symbol
    valid = dict()
    for s in symbols:
      if s in s2d: continue
      valid[s] = list(digits.difference(d for (x, d) in invalid if x == s))
    #for k in sorted(valid.keys()): printf("{k} -> {v}", v=valid[k])

    # at this point we can apply some heuristic re-writing rules:
    # word = value  -> value = word, if value is free of alphametic symbols
    if sane > 0:
      for (i, (expr, val, k)) in enumerate(exprs):
        #printf("[{i}] ({expr!r}, {val!r}, {k}) xs={xs} vs={vs}", xs=xs[i], vs=vs[i])

        # "<word> == <expr>" and expr contains no alphametic symbols:
        # ("<word> == <expr>", None, 1)  -->  ("<expr>", "<word>", 3)
        if k == 1:
          word = xpr = None
          m = re.match(r'\s*\{([' + symbols + r']+)\}\s*==\s*(.+)\s*$', expr)
          if m:
            (word, xpr) = m.groups()
          else:
            # try: "<expr> == <word>"
            m = re.match(r'\s*(.+?)\s*==\s*\{([' + symbols + r']+)\}\s*$', expr)
            if m:
              (xpr, word) = m.groups()

          if word and expr:
            if re.search(r'\{[' + symbols + r']+\}', xpr) is None:
              if verbose > 0: printf("[SubstitutedExpression: replacing ({t}) -> ({xpr} = {word})]", t=ts[i])
              exprs[i] = (xpr, word, 3)
              (xs[i], vs[i]) = (vs[i], xs[i])

        # "<word> = <int>": (even more efficient to use --assign instead)
        # ("<word>", <int>, 2)  --> ("<int>", "<word>", 3)
        if k == 2:
          if expr[0] == '{' and expr[-1] == '}':
            word = expr[1:-1]
            if all(x in symbols for x in word):
              if verbose > 0: printf("[SubstitutedExpression: replacing ({t}) -> ({val} = {{{word}}})]", t=ts[i])
              exprs[i] = (int2base(val, base=10), word, 3)
              (xs[i], vs[i]) = (vs[i], xs[i])

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

    if verbose & self.vI:
      # output solver information
      printf("[base={base}, digits={digits}, symbols={symbols!r}, distinct={distinct}]")
      printf("[s2d={s2d}, d2i={d2i}]")
      # output the solving strategy
      (ss, d) = (list(), set(s2d.keys()))
      for (i, x) in enumerate(xs):
        ss.append(sprintf("({e}) [{n}+{m}]", e=ts[i], n=len(x.difference(d)), m=len(vs[i].difference(d, x))))
        d.update(x, vs[i])
      printf("[strategy: {ss}]", ss=join(ss, sep=' -> '))

    # turn distinct into a dict mapping <symbol> -> <excluded symbols>
    if not isinstance(distinct, dict):
      d = dict()
      for ss in distinct:
        if sane > 0:
          if len(set(ss).difference(s2d.keys())) > len(digits):
            if verbose & self.vW:
              printf("[SubstitutedExpression: WARNING: distinct=\"{ss}\" has more symbols than available digits]")
            self._fail = 1
        for s in ss:
          if s not in d: d[s] = set()
          d[s].update(x for x in ss if x != s)
      distinct = d

    # generate the program (line by line)
    (prog, _, indent) = ([], "", "  ")
    (vx, vy, vr) = ("_x_", "_y_", "_r_") # local variables (that don't clash with _sym(x))

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
      prog.append(sprintf("{_}{s} = {d}", s=_sym(s)))
      done.add(s)

    # [denest] workaround statically nested block limit
    if denest:
      #  set other initial values and words to None
      for s in symbols:
        if s not in s2d:
          prog.append(sprintf("{_}{s} = None", s=_sym(s)))
      for w in words:
        prog.append(sprintf("{_}{w} = None", w=_sym(w)))
      # keep track of nested functions
      blocks = [ gensym('_substituted_expression_block') ]
      block = None
      block_args = join(chain(symbols, words), fn=_sym, sep=", ")
      indent_reset = indent

    # look for words which can be made
    for w in words:
      if all(x in done for x in w):
        prog.append(sprintf("{_}{w} = {x}", w=_sym(w), x=_word(w, base)))

    in_loop = False
    use_sets = ('s' in opt)  # use sets in code generation? [currently disabled by default]

    # deal with each <expr>,<value> pair
    for ((expr, val, k), xsyms, vsyms) in zip(exprs, xs, vs):

      # [denest] work around statically nested block limit
      if denest and block is None:
        # start a new function block
        block = blocks[-1]
        _ = indent_reset
        # In Python3 we can use [[ nonlocal ]] instead of passing the symbols around
        prog.append(sprintf("{_}def {block}({block_args}):"))
        _ += indent
        if decl: prog.append(sprintf("{_}{decl}"))
        in_loop = False

      # EXPERIMENTAL: do something about: "<iterator>: = <word>
      if ('i' in opt) and k == 3 and expr.endswith(':'):
        prog.append(sprintf("{_}for {vx} in {expr}")) # expr already has a colon
        _ += indent
        #prog.append(sprintf("{_}{w} = {vx}", w=_sym(val)))
        done.update(xsyms)

      else:
        # deal with each symbol in <expr>
        # TODO: we could consider these in an order that makes words
        # in <words> as soon as possible
        for s in xsyms:
          if s in done: continue
          # allowable digits for s
          ds = valid[s]
          in_loop = True

          check = None
          if done and s in distinct:
            # TODO: we should exclude initial values (that are excluded from ds) here
            check = list(_sym(x) for x in done if x in distinct[s])

          if use_sets:
            check = (sprintf(".difference({check})", check=_set(check)) if check else '')
            prog.append(sprintf("{_}for {s} in {ds}{check}:", s=_sym(s), ds=_set(ds)))
            _ += indent
          else:
            prog.append(sprintf("{_}for {s} in {ds}:", s=_sym(s)))
            _ += indent
            if check:
              check = join(((_sym(s) + ' != ' + x) for x in check), sep=' and ')
              prog.append(sprintf("{_}if {check}:"))
              _ += indent

          done.add(s)

          # look for words which can now be made
          for w in words:
            if s in w and all(x in done for x in w):
              prog.append(sprintf("{_}{w} = {x}", w=_sym(w), x=_word(w, base)))

      # calculate the expression
      if k != 0 and (not expr.endswith(':')): # (but not for the answer expression)
        x = _replace_words(expr, (lambda w: encl(_sym(w), '()')))
        prog.append(sprintf("{_}try:"))
        prog.append(sprintf("{_}  {vx} = int({x})"))
        prog.append(sprintf("{_}except NameError:")) # catch undefined functions
        prog.append(sprintf("{_}  raise"))
        if warn and verbose & self.vW:
          prog.append(sprintf("{_}except Exception as e:"))
          msg = 'printf("[WARNING: [{x}] {e}]", x=type(e).__name__)'
          prog.append(sprintf("{_}  {msg}"))
          prog.append(sprintf("{_}  {skip}", skip=('continue' if in_loop else 'return')))
        else:
          prog.append(sprintf("{_}except Exception:")) # maybe "except (ArithmeticError, ValueError)"
          prog.append(sprintf("{_}  {skip}", skip=('continue' if in_loop else 'return')))

      # check the value
      if k == 3:
        # this is a literal (alphametic) word
        # so it must have a non-negative value
        prog.append(sprintf("{_}if {vx} >= 0:"))
        _ += indent
        for (j, y) in enumerate(val[::-1], start=-len(val)):
          if y in done:
            # this is a symbol with an assigned value
            prog.append(sprintf("{_}{vy} = {vx} % {base}"))
            # check the value
            prog.append(sprintf("{_}if {vy} == {y}:", y=_sym(y)))
            _ += indent
            prog.append(sprintf("{_}{vx} //= {base}"))
            # and check x == 0 for the final value
            if j == -1:
              prog.append(sprintf("{_}if {vx} == 0:"))
              _ += indent
          else:
            # this is a new symbol...
            prog.append(sprintf("{_}{y} = {vx} % {base}", y=_sym(y)))
            check = list()
            # check it is different from existing symbols
            if y in distinct:
              check.extend(_sym(x) for x in done if x in distinct[y])
            # check any invalid values for this symbol
            for v in idigits.union(v for (s, v) in invalid if y == s):
              check.append(str(v))
            if check:
              if use_sets:
                if len(check) == 1:
                  check = _sym(y) + " != " + check[0]
                else:
                  check = _sym(y) + " not in " + _set(check)
              else:
                check = join((_sym(y) + " != " + x for x in check), sep=" and ")
              prog.append(sprintf("{_}if {check}:"))
              _ += indent
            prog.append(sprintf("{_}{vx} //= {base}"))
            # and check x == 0 for the final value
            if j == -1:
              prog.append(sprintf("{_}if {vx} == 0:"))
              _ += indent
            done.add(y)
            # look for words which can now be made
            for w in words:
              if y in w and all(x in done for x in w):
                prog.append(sprintf("{_}{w} = {x}", w=_sym(w), x=_word(w, base)))

      elif k == 1:
        # look for a True value
        prog.append(sprintf("{_}if {vx}:"))
        _ += indent

      elif k == 2:
        # it's a comparable value
        prog.append(sprintf("{_}if {vx} == {val}:"))
        _ += indent

      # [denest] work around statically nested block limit
      if denest and len(_) > denest:
        # chain into the next block
        block = gensym('_substituted_expression_block')
        blocks.append(block)
        # return the current state of the symbols
        # Python3 can use [[ yield from ... ]]
        prog.append(sprintf("{_}for {vr} in {block}({block_args}): yield {vr}"))
        block = None

    # [denest] work around statically nested block limit
    if denest:
      if block is None:
        # we need a final trivial block
        block = blocks[-1]
        _ = indent_reset
        prog.append(sprintf("{_}def {block}({block_args}):"))
        _ += indent
        if decl: prog.append(sprintf("{_}{decl}"))
      # close final function block
      prog.append(sprintf("{_}yield [{block_args}]"))
      _ = indent_reset
      # now call the first block
      block = blocks[0]
      prog.append(sprintf("{_}for [{block_args}] in {block}({block_args}):"))
      _ += indent

    # yield solutions as dictionaries
    d = join(((encl(s, "'") + ": " + _sym(s)) for s in sorted(done)), sep=', ')
    if answer:
      # compute the answer
      r = _replace_words(answer, (lambda w: encl(_sym(w), '()')))
      prog.append(sprintf("{_}{vr} = {r}"))
      prog.append(sprintf("{_}yield ({{ {d} }}, {vr})"))
    else:
      prog.append(sprintf("{_}yield {{ {d} }}"))

    # turn the program lines into a string
    prog = join(prog, sep=nl)

    if verbose & self.vC:
      printf("-- [code language=\"python\"] --{nl}{prog}{nl}-- [/code] --")

    # compile the solver
    # a bit of jiggery pokery to make this work in several Python versions
    # older Python barfs on:
    #   ns = dict()
    #   eval(prog, None, ns)
    #   solve = ns[solver]
    if not env: env = dict()
    gs = update(globals(), env)
    try:
      code = compile(prog, '<string>', 'exec')
    except Exception:
      # the program failed to compile
      # this can be because the supplied expressions do not form valid Python
      # or due to an issue in the Python interpreter itself
      # (e.g. in standard Python you can't have more than 20 nested blocks,
      # or more than 100 indent levels - PyPy does not have these limitations)
      printf("SubstitutedExpression: compilation error from Python interpreter [{sys.executable}]")
      if not (verbose & self.vC):
        printf("(use verbose level 256 to output code before compilation)")
        printf("(or use the \"denest=1\" option (--denest, -X) to reduce program complexity)")
      raise
    eval(code, gs)

    self._solver = gs[solver]
    self._globals = gs
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
    if self._fail: return

    solver = self._solver
    answer = self.answer
    header = self.header
    if check is None: check = self.check
    if first is None: first = self.first
    verbose = (self.verbose if verbose is None else self._verbose(verbose))

    if verbose & self.vH and header: printf("{header}")

    n = 0
    for s in solver():
      if check and (not check(s)): continue
      if verbose & self.vT: self.output_solution((s[0] if answer else s))
      # return the result
      yield s
      n += 1
      if first and first == n: break

  # solve the puzzle, and just return the answers
  def answers(self, **kw):
    """
    like solve(), but just return the answer for each solution
    (assuming the 'answer' parameter has been specified).
    """
    assert self.answer, "'answer' parameter must be specified"
    for (_, ans) in self.solve(**kw):
      yield ans

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
      # or: (k, int2base(d[k], base=10))
      ss.append(map2str(((k, d[k]) for k in solution), sep=" ", enc=""))
    if ss: print(join(ss, sep=' / '))


  def run(self, check=None, first=None, verbose=None):
    """
    find solutions to the substituted expression problem and output them.

    check - a function to accept/reject solutions
    first - if set to True will stop after the first solution is output
    verbose - control output

    returns a Record object with the following attributes:
      n = the number of solutions found
      answer = a multiset() object counting the number of times each answer
        occurs (if the "answer" parameter was set in init())
      accumulate = result of accumulating the answers (if the "accumulate"
        parameter was also set)
    """
    verbose = (self.verbose if verbose is None else self._verbose(verbose))

    if not self._prepared: self._prepare()

    # return:
    # n = number of solutions
    # count = count of answers
    # accumulator = accumulation of answers
    (n, ans, acc) = (0, None, None)

    # collect answers (either total number or collected by "answer")
    answer = self.answer
    if answer: ans = multiset()

    # also an accumulator object can be specified
    accumulate = self.accumulate
    if answer and accumulate:
      (acc, acc_t) = (accumulate, "accumulator")
      # if accumulate is a string, we should evaluate it
      if isinstance(acc, basestring):
        (acc, acc_t) = (eval(acc, self._globals), acc)

    # measure internal time
    t = None
    if verbose & self.vE:
      t = Timer(name="internal")
      t.start()

    # solve the problem, counting the answers
    for s in self.solve(check=check, first=first, verbose=verbose):
      n += 1
      if answer: ans.add(s[1])

    if t: t.stop()

    # output solutions
    s = ('' if n == 1 else 's')
    if answer and ans:
      answer = _replace_words(answer, identity)

      if not accumulate:
        # report the answer counts
        if verbose & self.vA:
          for (k, v) in ans.most_common():
            printf("{answer} = {k} [{v} solution{s}]")

      else:
        # report accumulated values
        if isinstance(acc, Sequence):
          # if we've been passed a sequence (of callables)
          acc = tuple(f(ans) for f in acc)
          if verbose & self.vA:
            printf("{acc_t}({answer}) = {acc} [from {n} value{s}]", vs=join(acc, sep=", ", enc="()"))

        else:
          # single callable
          acc = acc(ans)
          if verbose & self.vA:
            printf("{acc_t}({answer}) = {acc} [from {n} value{s}]")

    elif verbose & self.vA:
      # report solution count
      printf("[{n} solution{s}]")

    if t: t.report()

    return Record(n=n, answer=ans, accumulate=acc)

  # backward compatability
  go = run

  def substitute(self, s, text, digits=None):
    """
    given a solution to the substituted expression sum and some text,
    return the text with the letters substituted for digits.
    """
    return substitute(s, text, digits=digits)

  # !!! EXPERIMENTAL !!!
  # it may be better to implement this as a subclass of SubstitutedExpression
  @classmethod
  def split_sum(cls,
    terms, result=None, k=1, carries=None, extra=None,
    base=None, symbols=None, s2d=None, d2i=None, distinct=None, literal=None,
    answer=None, accumulate=None, env=None, code=None,
    template=None, sane=None, warn=None, verbose=None
  ):
    """
    split the alphametic sum represented by [[ sum(<terms>) = <result> ]]
    into sums consisting of <k> columns of the original sum with carries
    between the chunks.

    alternatively, if just the <terms> parameter is passed (and the <result>
    parameter is None), then the <terms> parameter can be given as:
      - a string representing the sum: "<term> + <term> + ... = <result>"
      - a sequence of simultaneous sums, represented as strings or
        (<terms>, <result>) pairs.

    additional parameters:

      carries - symbols that can be used for carries between chunks
      extra - extra expressions (that don't get split)

    the following parameters are passed to the SubstitutedExpression solver:

      base - the number base to operate in (default: 10)
      s2d - initial symbol to digit mapping
      d2i - initial invalid digits
      distinct - symbols which should have distinct values
      literal - symbols which stand for themselves
      answer - expression for the answer value
      accumulate - accumulate answers using specified object
      env - additional environment for evaluation
      code - additional lines of code evaluated before solving
      template - solution template
      sane - enable/disable sanity checks
      warn - enable/disable exception warnings
      verbose - control informational output

    if <result> is None, then <terms> can contain the sum represented
    as a string (e.g. "ABC + DEF = GHI" or "{ABC} + {DEF} = {GHI}"),
    or a sequence of sums, each represented as a string or as a
    (<terms>, <result>) pair.

    return value is an object with the following attributes:

      exprs - the alphametic expressions corresponding to the chunks
      symbols - the symbols used in the original sum
      carries - the symbols used in the carries between chunks
      d2i - is augmented with additional restrictions for carry symbols
      distinct - symbols which should have distinct values
      literal - symbols which stand for themselves
      template - template for original sum
      answer - answer parameter
      accumulate - accumulate parameter
      env - env parameter
      code - code parameter
      verbose - verbose parameter
      extra - extra expressions
      solver - a function to return the solver (with "standard" arguments)
      solve - a function to generate solutions from the solver (ditto)
      run - a function to run the solver (ditto)
    """
    # defaults
    if base is None: base = cls.defaults.get('base', 10)
    if carries is None: carries = str_lower + rev(str_upper) + str_digit
    if extra is None: extra = ()
    if s2d is None: s2d = cls.defaults.get('s2d', None)
    if d2i is None: d2i = cls.defaults.get('d2i', None)
    if distinct is None: distinct = cls.defaults.get('distinct', None)
    if literal is None: literal = cls.defaults.get('literal', None)
    if answer is None: answer = cls.defaults.get('answer', None)
    if accumulate is None: accumulate = cls.defaults.get('accumulate', None)
    if env is None: env = cls.defaults.get('env', None)
    if code is None: code = cls.defaults.get('code', None)
    if sane is None: sane = cls.defaults.get('sane', 1)
    if warn is None: warn = cls.defaults.get('warn', 0)
    if verbose is None: verbose = cls.defaults.get('verbose', None)

    # the symbols to replace (for implicit expressions)
    if symbols is None: symbols = str_upper
    symbols = set(symbols)
    if s2d: symbols.update(s2d.keys())
    if literal: symbols.update(literal)
    symbols = join(sorted(symbols))

    # process the sums
    sums = list()

    # if result is given, there is one sum: sum(terms) = result
    if result is not None:
      sums.append((terms, result))

    else:
      # result is None, terms can be:
      # - the sum represented as a string
      # - a sequence of sums, represented as strings or (terms, result) pairs
      if isinstance(terms, basestring): terms = [terms]

      for v in terms:
        if isinstance(v, basestring):
          ts = re.split(r'[\s\+\=]+', v)
          r = ts.pop()
          sums.append((ts, r))
        else:
          sums.append(v)

    # convert implicit (without braces) into explicit (with braces)
    sums = list((_fix_implicit_seq(terms, symbols), _fix_implicit(result, symbols)) for (terms, result) in sums)
    extra = _fix_implicit_seq(extra, symbols)
    answer = _fix_implicit(answer, symbols)

    # check terms/result are fully alphametic, and strip braces
    for i in xrange(len(sums)):
      (terms, result) = sums[i]
      assert all(x[0] == '{' and x[-1] == '}' for x in terms + [result]), "terms/result must be fully alphametic"
      sums[i] = (list(x[1:-1] for x in terms), result[1:-1])

    # find words in: terms, result, extra, answer
    words = set()
    for (terms, result) in sums:
      words.update(terms)
      words.add(result)
    for x in extra:
      words.update(_find_words(x))
    if answer:
      words.update(_find_words(answer))

    # determine symbols that are used, and available carries
    symbols = join(sorted(union(words)))
    carries = diff(carries, symbols, fn=list)

    # no leading zeros by default
    if d2i is None:
      d2i = set((0, w[0]) for w in words if len(w) > 1)
    elif isinstance(d2i, dict):
      d2i = set((d, s) for (d, ss) in d2i.items() for s in ss)

    # prepare return values
    template_ = list()
    for (terms, result) in sums:
      template_.append(encl(join(terms, fn=encl, sep=' + ') + " = " + encl(result), b="()"))
    template_ = join(template_, sep=' ')

    # k=0 disables splitting
    if k == 0: k = len(result)

    # calculate max possible sum for terms <ts>
    def max_sum(ts, maxv, d2i, base):
      r = 0
      for t in ts:
        # just look at the leading symbol
        k = t[0]
        v = maxv.get(k)
        if v is None:
          # find the largest symbol not in d2i
          v = maxv[k] = peek(x for x in irange(base - 1, 0, step=-1) if (x, k) not in d2i)
        # assume all the other symbols have value (base - 1)
        n = len(t)
        r += (v if n == 1 else (v + 1) * base**(n - 1) - 1)
      return r

    # construct the sub-expressions for each chunk
    (exprs, cs, maxv) = (list(), list(), dict())
    for (terms, result) in sums:
      while terms:
        (carry, ck) = ('', 0)
        if len(terms) > 1:
          # chop k characters off the end of each term
          ts = list(t[-k:] for t in terms)
          ts_ = list(filter(None, (t[:-k] for t in terms)))
          # upper bound for carry out
          maxc = max_sum(ts, maxv, d2i, base) // (base**k)
          # number of carry symbols required
          maxc_ds = nsplit(maxc, base=base)
          ck = len(maxc_ds)
        else:
          # use the remaining term
          (ts, ts_) = (terms, None)
        if ts_:
          # chop k characters off the end of the result
          (rs, rs_) = (result[-k:], result[:-k])
        else:
          # otherwise, use the whole result
          (rs, rs_) = (result, None)
        # allocate a carry out
        if rs_:
          carry = join(carries[:ck])
          assert len(carry) == ck, "ran out of carry symbols"
          carries = carries[ck:]
          cs.append(carry)
        # add an expression for this set of columns
        exprs.append(join(ts, fn=encl, sep=" + ") + " = " + encl(carry + rs))
        if carry:
          # add the carry to the remaining terms
          ts_.append(carry)
          # determine d2i values
          d2i.update((d, carry[0]) for d in irange(maxc_ds[0] + 1, base - 1))
          maxv[carry[0]] = maxc_ds[0]
        if not rs_: break
        (terms, result) = (ts_, rs_)

    carries = join(cs)

    if extra:
      exprs.extend(extra)
      template_ += " " + join(extra, sep=") (", enc="()")
    if answer:
      template_ += " (" + answer + ")"
    if distinct == 1: distinct = symbols
    if template is None: template = template_
    # a solver with "standard" arguments
    solver = Delay(
      SubstitutedExpression,
      exprs,
      base=base, distinct=distinct, literal=literal, env=env, code=code,
      s2d=s2d, d2i=d2i, template=template, solution=symbols,
      answer=answer, accumulate=accumulate, sane=sane, warn=warn, verbose=verbose,
    )
    return Record(
      exprs=exprs,
      base=base,
      symbols=symbols,
      distinct=distinct,
      literal=literal,
      carries=cs,
      s2d=s2d,
      d2i=d2i,
      template=template,
      answer=answer,
      accumulate=accumulate,
      env=env,
      code=code,
      verbose=verbose,
      solver=(lambda: solver.value),
      solve=(lambda *args, **kw: solver.value.solve(*args, **kw)),
      answers=(lambda *args, **kw: solver.value.answers(*args, **kw)),
      run=(lambda *args, **kw: solver.value.run(*args, **kw)),
    )

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

    if self.literal:
      args.append(sprintf("--literal={q}{self.literal}{q}"))

    if self.digits:
      args.append(sprintf("--digits={q}{digits}{q}", digits=join(self.digits, sep=",")))

    if self.s2d:
      for (k, v) in sorted(self.s2d.items(), key=lambda t: t[1]):
        args.append(sprintf("--assign={q}{k},{v}{q}"))

    # we should probably make this from self.invalid
    if self.d2i:
      for (k, v) in sorted((self.d2i.items() if hasattr(self.d2i, 'items') else self.d2i), key=lambda t: t[0]):
        if v:
          args.append(sprintf("--invalid={q}{k},{v}{q}", v=join(sorted(v))))

    if self.answer:
      args.append(sprintf("--answer={q}{self.answer}{q}"))

    if self.accumulate:
      args.append(sprintf("--accumulate={q}{self.accumulate}{q}"))

    if self.template:
      args.append(sprintf("--template={q}{self.template}{q}"))

    if self.solution:
      args.append(sprintf("--solution={q}{self.solution}{q}"))

    if self.header:
      args.append(sprintf("--header={q}{self.header}{q}"))

    if self.env is not None:
      printf("WARNING: can't generate arg for \"env\" parameter (maybe use \"code\" instead)")

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

    if self.denest is not None:
      args.append(sprintf("--denest={self.denest}"))

    if self.decl is not None:
      args.append(sprintf("--decl={q}{self.decl}{q}"))

    if self.sane is not None:
      args.append(sprintf("--sane={self.sane}"))

    if self.warn is not None:
      args.append(sprintf("--warn={self.warn}"))

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
      "  --symbols=<str> (or -s<s>) = symbols to replace with digits",
      "  --base=<n> (or -b<n>) = set base to <n>",
      "  --assign=<symbol>,<decimal> (or -a<s>,<d>) = assign decimal value to symbol",
      "  --digits=<digit>,... or --digits=<digit>-<digit> (or -d...) = available digits",
      "  --invalid=<digits>,<symbols> (or -i<ds>,<ss>) = invalid digit to symbol assignments",
      "  --answer=<expr> (or -A<expr>) = count answers according to <expr>",
      "  --accumulate=<expr> (or -M<expr>) = accumulate answers according to <expr>",
      "  --template=<str> (or -T<s>) = solution template",
      "  --solution=<str> (or -S<s>) = solution symbols",
      "  --header=<str> (or -H<s>) = solution header",
      "  --distinct=<str> (or -D<s>) = symbols that stand for different digits (0 = off, 1 = on)",
      "  --literal=<str> (or -L<s>) = symbols that stand for themselves",
      "  --code=<str> (or -C<s>) = initialisation code (can be used multiple times)",
      "  --first (or -1) = stop after the first solution",
      "  --reorder=<n> (or -r<n>) = allow reordering of expressions (0 = off, 1 = on)",
      "  --denest=<n> (or -X<n>) = workaround statically nested block limit (0 = off, 1 = on, 2+ = depth)",
      "  --sane=<n> (or -Y<n>) = enable/disable sanity checks (0 = off, 1 = on)",
      "  --warn=<n> (or -W<n>) = enable/disable exception warnings (0 = off, 1 = on)",
      "  --verbose[=<s>] (or -v[<s>]) = verbosity (0 = off, 1 = on, HTAEPIC239 = more)",
      "  --help (or -h) = show command-line usage",
      "",
      "verbosity levels:",
      "  H = output header template [1,2,3,9]",
      "  T = output solutions (from template) [1,2,3,9]",
      "  A = output answer count / accumulated measures [1,2,3,9]",
      "  E = output elapsed timing info [3,9]",
      "  P = output solver parameters [9]",
      "  I = output solver info [2,3,9]",
      "  C = output generated Python code [3,9]",
      "",
    )

  # process option <k> = <v> into <opt>, returns:
  #   None = help
  #   True = option processed
  #   Exception = error
  @classmethod
  def _getopt(cls, k, v, opt, allow=()):
    # used single character options: abcdhikrsvACDEHLMSTWXY1@

    if k == 'h' or k == 'help':
      # --help (or -h)
      return None
    elif k == 's' or k == 'symbols':
      # --symbols=<str> (or -s<str>)
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
    elif k == 'L' or k == 'literal':
      opt['literal'] = v
    elif k == 'C' or k == 'code':
      if 'code' not in opt: opt['code'] = []
      opt['code'].append(v)
    elif k == '@' or k == 'macro':
      (m, t) = re.split(r'\s*=\s*', v)
      if m[0] == '@': m = m[1:]
      if opt.get('macro', None) is None: opt['macro'] = dict()
      opt['macro'][m] = t
    elif k == 'A' or k == 'answer':
      opt['answer'] = v
    elif k == 'M' or k == 'accumulate':
      opt['accumulate'] = v
    elif k == '1' or k == 'first':
      opt['first'] = (int(v) if v else 1)
    elif k == 'v' or k == 'verbose':
      opt['verbose'] = (v if v else 1)
    elif k == 'r' or k == 'reorder':
      opt['reorder'] = (int(v) if v else 0)
    elif k == 'X' or k == 'denest':
      opt['denest'] = (int(v) if v else 1)
    elif k == 'G' or k == 'decl':
      opt['decl'] = v
    elif k == 'Y' or k == 'sane':
      opt['sane'] = (int(v) if v else 0)
    elif k == 'W' or k == 'warn':
      opt['warn'] = (int(v) if v else 1)
    elif k == 'O' or k == 'opt':
      opt['opt'] = v
    elif 'extra' in allow and (k == 'E' or k == 'extra'):
      if opt.get('extra', None) is None: opt['extra'] = list()
      opt['extra'].append(v)
    elif 'split' in allow and (k == 'k' or k == 'split'):
      opt['split'] = (int(v) if v else 1)
    elif 'carries' in allow and (k == 'c' or k == 'carries'):
      opt['carries'] = v

    # unrecognised option
    else:
      raise ValueError()

    return True


  # class method to make options from arguments
  @classmethod
  def _opt_from_args(cls, args, **kw):
    #if not args: return

    # process options
    opt = dict(_argv=list(), s2d=dict(), d2i=None, verbose=1, first=0, reorder=1, denest=0, sane=1, warn=0)
    for arg in args:
      # deal with option args
      try:
        k = v = None
        if arg.startswith('--'):
          (k, _, v) = arg.lstrip('-').partition('=')
        elif arg.startswith('-'):
          (k, v) = (arg[1], arg[2:])

        if not k:
          # push non-option args onto _argv
          opt['_argv'].append(arg)
          continue

        if not cls._getopt(k, v, opt, **kw):
          return None

      except Exception:
        raise ValueError(str.format("[{name}] invalid option: {arg}", name=cls.__name__, arg=arg)) # from None

    return opt

  # class method to make object from a collection of arguments
  @classmethod
  def from_args(cls, args):
    # if args is a string
    if isinstance(args, basestring):
      # split into lines, and discard comments
      args = list(s for s in (x.partition('#')[0].strip() for x in args.splitlines()) if s)

    # parse the args
    opt = cls._opt_from_args(args)
    if opt is None: return
    # create the object
    argv = opt.pop('_argv')
    return cls(argv, **opt)

  # class method to call from the command line
  @classmethod
  def run_command_line(cls, args):
    """
    run the SubstitutedExpression solver with the specified command
    line arguments.

    we can solve substituted sum problems:

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
      self = cls.from_args(args)
      if self is not None:
        # call the solver
        if self.run():
          return 0

    # failure, output usage message
    print(join(cls._usage(), sep=nl))
    return -1

  @classmethod
  def run_split_sum(cls, args):
    if args:
      # parse the args, and sort out the supported ones
      opt = cls._opt_from_args(args, allow={'split', 'carries', 'extra'})
      if opt is not None:

        # "--" can be used to separate sums to split from extra expressions
        argv = opt.pop('_argv')
        i = find(argv, '--')
        if i != -1:
          sums = argv[:i]
          extra = argv[i + 1:]
        else:
          # attempt to split everything
          sums = argv
          extra = list()

        cols = opt.pop('split', 1)
        carries = opt.pop('carries', None)
        kw = dict()
        if opt.get('extra') is not None: extra = opt['extra'] + extra
        kw['extra'] = extra
        # check restricted arguments
        assert 'digits' not in opt, "split_sum: doesn't handle 'digits' specification (use 'invalid' instead)"
        if 'distinct' in opt:
          assert len(opt['distinct']) == 1, "split_sum: doesn't handle multiple 'distinct' values"
          opt['distinct'] = opt['distinct'][0]
        # copy accepted arguments
        for k in ['base', 'symbols', 's2d', 'd2i', 'answer', 'accumulate', 'env', 'code', 'template', 'distinct', 'literal', 'verbose']:
          if k in opt:
            kw[k] = opt.pop(k)
        #if opt: printf("SubstitutedExpression.run_split_sum: ignoring args: {opt}")
        self = cls.split_sum(sums, result=None, k=cols, carries=carries, **kw)
        if self.run():
          return 0

    # failure
    print(join(cls._usage(), sep=nl))
    printf("SubstitutedExpression.split_sum also accepts the following option:")
    printf("  --split=<n> (or -k<n>) = split the sum into groups of <n> columns")
    printf("  --carries=<str> (or -c<s>) = symbols available for carries")
    printf("  --extra=<str> (or -E<s>) = additional (non-sum) expressions")
    printf()
    return -1

  # class method to load a run file
  @classmethod
  def from_file(cls, path, args=None, env=None):
    (cmd, argv) = parsefile(path, args)
    if run.alias.get(cmd, cmd) != cls.__name__:
      printf("WARNING: loading '{cmd}' into '{cls.__name__}'")
    opt = cls._opt_from_args(argv)
    if opt is None: return
    argv = opt.pop('_argv')
    if env: opt['env'] = env
    return cls(argv, **opt)

  # parse a string as a run-file
  @classmethod
  def from_string(cls, string, args=None, env=None):
    (cmd, argv) = parsefile('<string>', args, string=string)
    if argv:
      if run.alias.get(cmd, cmd) != cls.__name__:
        printf("WARNING: loading '{cmd}' into '{cls.__name__}'")
      opt = cls._opt_from_args(argv)
      if opt is None: return
      argv = opt.pop('_argv')
      if env: opt['env'] = env
      return cls(argv, **opt)

  # class method to provide a read/eval/print loop
  @classmethod
  def repl(cls, args=(), timed=1):
    """
    Provide a read/eval/print loop for evaluating alphametics.

    Use the following command to invoke it:

      % python enigma.py SubstitutedExpression.repl

    timed=1 will time the evaluation.

    """
    catch(import_fn, "readline")

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
          cls.run_command_line(exprs)
        except Exception as e:
          print(e)
          print("[ERROR: try again]")
        print()

  @classmethod
  def run_repl(cls, args):
    cls.repl()


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

class Slots(object):

  def __init__(self, wildcard='?', symbols=str_upper + str_lower):

    # wildcard character in input strings
    self.wildcard = wildcard

    # pool of valid symbols
    self.symbols = symbols

    # slot ids
    self._id = 0

    # slot properties
    self._s2p = defaultdict(set) # <slot> -> <props>
    self._p2s = defaultdict(lambda: defaultdict(set)) # <type> -> <value> -> <slots>

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

    if s in str_digit:
      # integer literal, use the same slot for the same literal
      return self.slot_find(_EQ, int(s))

    if s in symbols:
      # a symbol, use the same slot for the same input symbol
      return self.slot_find(_IS, s)

    # unrecognised input symbol
    raise ValueError(str.format("_allocate: invalid input symbol <{s}>", s=s))

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
    (i, j) = sorted((i, j))
    # copy any properties from slot j to slot i
    self.slot_setprops(i, *(self._s2p[j]))
    # mark slot j as being unified with i
    self.slot_setprops(i, (_UN, j))

  # unify two sequence of slots <s> and <t>
  def unify(self, s, t):
    assert len(s) == len(t), "unification length mismatch"
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
# SubstitutedDivisionSolution = namedtuple('SubstitutedDivisionSolution', 'a b c r subs d s')

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

  >>> SubstitutedDivision('pkmkh / ?? = ???', ['pkm - pmd = xp', 'xpk - ?? = kh', 'khh - mbg = k']).run().n
  pkmkh / ?? = ??? (rem k) [pkm - pmd = xp, xpk - ?? = kh, khh - mbg = k]
  47670 / 77 = 619 (rem 7) [476 - 462 = 14, 147 - 77 = 70, 700 - 693 = 7] / b=9 d=2 g=3 h=0 k=7 m=6 p=4 x=1
  [1 solution]
  1


  See SubstitutedDivision.run_command_line() for more examples.
  """

  rtype = None

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
      return (none if v is None else (encl(v) if brace else v))

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
        if ((not subs[i][-1]) or subs[i][-1] == '0'): subs[i][-1] = None
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
    for s in chain([a, b, c], *subs):
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
    if kw.get('extra') is not None: expr.extend(kw['extra'])

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
    d2i = defaultdict(set)
    if kw.get('d2i', None):
      for (k, v) in kw['d2i'].items():
        d2i[k].update(self.input_symbols[s] for s in v)
    for (v, ss) in slots.prop_items(_NE):
      if 'digits' in kw and v not in kw['digits']: continue
      for s in ss:
        d2i[v].add(slots.symbol(s))
    opt['d2i'] = d2i

    # verbatim options
    for v in ('base', 'digits', 'answer', 'accumulate', 'env', 'code', 'verbose', 'denest', 'sane', 'warn'):
      if v in kw:
        opt[v] = kw[v]

    # initialise the substituted expression
    SubstitutedExpression.__init__(self, expr, **opt)

  def substitute_all(self, d, ss):
    base = self.base
    if ss is None: return None
    return tuple(base2int(self.substitute(d, s), base=base) for s in ss)

  def solve(self, check=None, first=None, verbose=None):
    """
    generate solutions for the substituted long division problem.

    solutions are returned as a SubstitutedDivisionSolution() object

    check - a boolean function called to reject unwanted solutions
    first - if set to True only the first solution is returned
    verbose - an integer controlling the output of solutions and additional information
    """
    verbose = (self.verbose if verbose is None else self._verbose(verbose))
    answer = self.answer
    # solution templates
    (ta, tb, tc, tsubs, tr) = self.args
    if tr is None: tr = '0'
    for s in reversed(tsubs):
      if s:
        if s[-1] is None: s[-1] = '0'
        break
    # find solutions (but disable solution output)
    for s in SubstitutedExpression.solve(self, verbose=(verbose & ~(self.vT))):
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
      if SubstitutedDivision.rtype is None: SubstitutedDivision.rtype = namedtuple('SubstitutedDivisionSolution', 'a b c r subs d s')
      ss = SubstitutedDivision.rtype(a, b, c, r, subs, d, s)
      if check and (not check(ss)): continue
      # output the solution
      if verbose & self.vT: self.output_solution(ss)
      # return the result
      yield ((ss, ans) if answer else ss)
      if first: break

  def output_solution(self, s, template=None, solution=None):
    # copy any input symbols that were eliminated
    SubstitutedExpression.output_solution(self, s.s, template=template, solution=solution)

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
  def _getopt(cls, k, v, opt, **kw):
    kw['allow'] = ({'extra'} if kw.get('allow') is None else append(kw['allow'], 'extra'))
    return SubstitutedExpression._getopt(k, v, opt, **kw)

  @classmethod
  def _opt_from_args(cls, args, **kw):
    opt = super(SubstitutedDivision, cls)._opt_from_args(args, **kw)
    if opt is None: return
    # "--" can be used to separate division sum from extra expressions
    argv = opt['_argv']
    i = find(argv, '--')
    if i != -1:
      opt['extra'] = opt.get('extra', []) + argv[i + 1:]
      opt['_argv'] = argv[:i]
    return opt

  @classmethod
  def run_command_line(cls, args):
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
    return super(SubstitutedDivision, cls).run_command_line(args)

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
    self.fn = true

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
    self._table = namedtuple('Table', ('played',) + self._games + ('points',))

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
      for r in cproduct(gs): yield r

  # points for a game
  def points(self, g, t=0):
    if t: g = self._swap.get(g, g)
    return self._points.get(g, 0)

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
  def scores(self, gs, ts, f, a, pss=None, pts=None, min_goals=0, valid=true, s=[]):
    """
    Generate possible score lines for a sequence of match outcomes <gs>,
    team assignments <ts>, and total goals for <f> and against <a>.

    A sequence of scores for matches already played <pss> and
    corresponding team assignments <pts> can be specified, in which case
    the goals scored in these matches will be subtracted from <f> and
    <a> before the score lines are calculated.

    <min_goals> is the minimum number of goals scored by each team.

    <valid> is a function that can be used to validate scores
    (which will be None, or (x, y)).

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
    return self._scores(gs, ts, f, a, min_goals, valid, [])

  def _scores(self, gs, ts, f, a, min_goals, valid, s):
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
        if valid(None):
          # [Python 3]: yield from ...
          for r in self._scores(gs, ts, f, a, min_goals, valid, s + [None]): yield r
      # is it a draw?
      elif g == 'd':
        for i in irange(min_goals, min(f, a)):
          s0 = (i, i)
          if valid(s0):
            # [Python 3]: yield from ...
            for r in self._scores(gs, ts, f - i, a - i, min_goals, valid, s + [s0]): yield r
      # is it a win?
      elif g == 'w':
        for j in irange(min_goals, a):
          for i in irange(j + 1, f):
            s0 = ((j, i) if t else (i, j))
            if valid(s0):
              # [Python 3]: yield from ...
              for r in self._scores(gs, ts, f - i, a - j, min_goals, valid, s + [s0]): yield r
      # is it a loss?
      elif g == 'l':
        for i in irange(min_goals, f):
          for j in irange(i + 1, a):
            s0 = ((j, i) if t else (i, j))
            if valid(s0):
              # [Python 3]: yield from
              for r in self._scores(gs, ts, f - i, a - j, min_goals, valid, s + [s0]): yield r

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
    """return a sequence of outcomes ('x', 'w', 'd', 'l') for a sequence of scores"""
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
  extract_table = lambda self, ms, t: call(self.table, self.extract(ms, t))
  extract_goals = lambda self, ss, t: call(self.goals, self.extract(ss, t))


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
    ms = list((x, t) for x in xrange(0, t)) + list((t, x) for x in xrange(t + 1, n))
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
      teams = sorted(xrange(0, n), key=lambda i: (rows[i].count('?'), len(set(rows[i]))))
    if matches is None: matches = dict()
    if d is None: d = dict()
    if vs is None: vs = list(irange(0, 9))
    for z in self._substituted_table(table, n, teams, matches, d, vs): yield z


  def _substituted_table_goals(self, gf, ga, matches, d, teams, scores, min_goals, valid):
    # are we done?
    if not teams:
      yield scores
      return

    # deal with the first team
    t = teams[0]

    # matches for team t
    ms = list(m for m in matches.keys() if t in m)
    # matches remaining to be scored
    rs = diff(ms, scores)
    if not rs:
      # check the values
      (f, a) = self.goals(list(scores[m] for m in ms), list(m.index(t) for m in ms))
      if f == d[gf[t]] and a == d[ga[t]]:
        for z in self._substituted_table_goals(gf, ga, matches, d, teams[1:], scores, min_goals, valid): yield z
    else:
      # matches we already have scores for
      sms = list(m for m in ms if m in scores)
      # find possible scores for each remaining match
      ss = self.scores(
        list(matches[m] for m in rs),
        list(m.index(t) for m in rs),
        d[gf[t]],
        d[ga[t]],
        list(scores[m] for m in sms),
        list(m.index(t) for m in sms),
        min_goals,
        valid
      )
      for s in ss:
        scores2 = update(scores, rs, s)
        for z in self._substituted_table_goals(gf, ga, matches, d, teams[1:], scores2, min_goals, valid): yield z

  # gf, ga - goals for, goals against (map: team -> symbol)
  # matches - match outcomes
  # teams - order teams are processed in
  # scores - score lines
  def substituted_table_goals(self, gf, ga, matches, d=None, teams=None, scores=None, min_goals=0, valid=true):
    """
    determine the scores in matches from a substituted table football problem.

    generates dicts <scores>, which give possible score lines for the
    matches in <matches> (if a match is specified as 'x' (unplayed) a
    score of None is returned).

    gf, ga - goals for, goals against columns in the table. specified
    as maps of teams to symbols that index into <d> to give the actual
    values.

    matches - the match outcomes. usually this will be the result of a
    call to substituted_table().

    teams - the order the teams are processed in.

    scores - known scores. usually this is empty.

    min_goals - minimum number of goals for each team in a match (usually 0).
    """
    if d is None: d = digit_map(0, 9)
    if teams is None: teams = (list(gf.keys()) if hasattr(gf, 'keys') else list(xrange(0, len(gf))))
    if scores is None: scores = dict()
    # fill out unplayed matches
    for (k, v) in matches.items():
      if v == 'x': scores[k] = None
    for z in self._substituted_table_goals(gf, ga, matches, d, teams, scores, min_goals, valid): yield z


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
      printf("{d}", d=map2str(d, sep=" ", enc=""))
    if end is not None:
      printf("{end}")

###############################################################################

# Domino Grid solver (see Enigma 179, Enigma 303, Enigma 342, ...)

class DominoGrid(object):

  def __init__(self, N, M, grid):
    """
    create a domino grid to solve.

    the grid is an NxM grid of dominoes, the values in the
    grid are specified as a linearised list.

    "holes" in the grid are indicated with the value: None.
    """
    # checks
    n = len(grid)
    assert n == N * M
    (D, r) = divmod(n - grid.count(None), 2)
    assert r == 0
    self.grid = grid
    self.N = N  # columns
    self.M = M  # rows
    self.D = D  # number of dominoes

  # solve the grid
  # fixed = pairs of indices of fixed dominoes
  # used = dominoes not available for use
  # returns: (<solved grid>, <dominoes used>)
  def solve(self, fixed=None, used=[]):
    """
    find placements of dominoes in the specified grid.

    fixed = pairs of indices of fixed dominoes
    used = dominoes that are not available for use

    any domino identified in 'fixed' is automatically in 'used'.

    returns: (<solved grid>, <used dominoes>)
    """
    (N, M, D, grid) = (self.N, self.M, self.D, self.grid)

    # is a square assigned?
    assigned = lambda v: v is None or v < 0

    # g = grid
    # n = label of next domino (1 to D)
    # ds = dominoes already placed
    def _solve(g, n, ds):
      # are we done?
      if n > D:
        # output the pairings
        yield (g, ds)
      else:
        # find the next unassigned square
        for (i, d) in enumerate(g):
          if assigned(d): continue
          (y, x) = divmod(i, N)
          # find placements for the domino
          js = list()
          # horizontally
          if x < N - 1 and not assigned(g[i + 1]): js.append(i + 1)
          # vertically
          if y < M - 1 and not assigned(g[i + N]): js.append(i + N)
          # try possible placements
          for j in js:
            d = ordered(g[i], g[j])
            if d not in ds:
              for s in _solve(update(g, [i, j], [-n, -n]), n + 1, ds.union([d])): yield s
          break

    # fixed can contain initial placements of dominoes
    (n, ds) = (1, set(used))
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
    """
    given a solution from solve() output the solved grid.
    """
    (N, M, grid) = (self.N, self.M, self.grid)
    (s, ds) = s

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
  def run(self, fixed=None, used=[], sep='', prefix=''):
    """solve a grid and output the solutions"""
    for s in self.solve(fixed, used):
      self.output_solution(s, prefix=prefix)
      print(sep)

  # backward compatability
  go = run

###############################################################################

# Timing

import atexit
import time

class Timer(object):

  """
  This class provides elapsed timing measures.

  There is a default timing object called 'timer' created. So you can
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


  When a Timer object is initialised the 'timer' parameter specifies what
  timing function to use. A value of 'E' use elapsed (real) time and a
  value of 'P' use process (CPU) time. 'E' should always be available,
  'P' may not be. You can specify 'PE', to try 'P' first, and then 'E'.

  If you know what timing function you want to use you can pass it
  directly. (Or pass the name of a function in the 'time' module).
"""

  # lookup for timers
  timers = None

  @classmethod
  def init(self):
    d = dict()
    if hasattr(time, 'thread_time'):
      d['T'] = time.thread_time
    if hasattr(time, 'process_time') and sys.platform != "win32":
      d['P'] = time.process_time # process time
    if hasattr(time, 'perf_counter'):
      d['E'] = time.perf_counter # elapsed time
    elif sys.platform == "win32" and hasattr(time, 'clock'):
      d['E'] = time.clock # elapsed time
    elif hasattr(time, 'time'):
      d['E'] = time.time # elapsed time
    Timer.timers = d

  def __init__(self, name='timing', timer="PE", file=sys.stderr, exit_report=1, auto_start=1, verbose=0):
    """
    Create (and start) a timer.

    name = the name used in the report
    timer = function used to measure time (should return a number of seconds)
    file = where the report is sent
    exit_report = should the report be generated at exit
    auto_start = should the timer be automatically started
    """
    if Timer.timers is None: Timer.init()
    self._t0 = None
    self._t1 = None
    self._name = name
    if not callable(timer):
      # timer can be the name of a function from time, e.g. 'time' or 'clock'
      fn = getattr(time, timer, None)
      if fn is not None:
        timer = fn
      else:
        # timer can be a sequence of timers to try: 'P' = process time, 'E' = elapsed time
        for k in timer:
          fn = Timer.timers.get(k, None)
          if fn is not None:
            timer = fn
            break
        else:
          raise ValueError("no such timer \"" + timer + "\"")
    self._timer = timer
    self._file = file
    self._exit_report = exit_report
    self._report = None
    self._verbose = verbose or ('v' in _PY_ENIGMA)
    if self._verbose: printf("[{name}] timer = {timer}")
    if auto_start: self.start()

  def start(self):
    """set the start time of a timer"""
    if self._exit_report:
      atexit.register(self.report, force=0)
      self._exit_report = False
    self._t1 = None
    self._t0 = self._timer()
    if self._verbose: printf("[{self._name}] start = {self._t0}")

  def stop(self, report=0):
    """set the stop time of a timer"""
    self._t1 = self._timer()
    if self._verbose: printf("[{self._name}] stop = {self._t1}")
    if report: self.report(force=1)

  def elapsed(self, disable_report=1):
    """
    return the elapsed time of a stopped timer

    disable_report = should the report be disabled
    """
    if disable_report: self._report = '<none>'
    return (self._t1 or self._timer()) - self._t0

  def format(self, t, fmt='{:.2f}'):
    """format a time for the report"""
    u = 's'
    if t < 1.0: (t, u) = (1000 * t, 'ms')
    if t < 1.0: (t, u) = (1000 * t, 'us')
    return (fmt + u).format(t)

  def report(self, force=1):
    """
    Stop the timer and generate the report (if required).

    The report will only be generated once (if it's not been disabled).
    """
    if self._report and (not force): return self._report
    if self._t1 is None: self.stop()
    e = self.elapsed()
    self._report = str.format("[{n}] total time: {e:.7f}s ({f})", n=self._name, e=e, f=self.format(e))
    print(self._report, file=self._file)
    return self._report

  def printf(self, fmt='', **kw):
    e = self.elapsed(disable_report=0)
    s = _sprintf(fmt, kw, sys._getframe(1))
    printf("[{n} {e}] {s}", n=self._name, e=self.format(e))

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.report()

# function wrapper
def timed(f):
  """return a timed version of function <f>"""
  @functools.wraps(f)
  def _inner(*args, **kw):
    n = f.__name__
    t = Timer(n)
    r = f(*args, **kw)
    #printf("[{n} {args} {kw} => {r}]")
    t.report()
    return r
  return _inner

# create a default timer
timer = Timer(auto_start=0)


_profiler = lazy_import('cProfile.Profile', name='_profiler')

class Profiler(object):
  """
  This class provides an interface to the Python cProfile module.

  There is a default profiler object called 'profiler' created, so you
  can profile code fragments using:

    from enigma import profiler

    profiler.start()
    some_code()
    profiler.stop()
  """
  def start(self):
    self.profiler = _profiler()
    self.profiler.enable()

  def stop(self):
    self.profiler.disable()
    self.profiler.print_stats(sort=0)

# create a default profiler
profiler = Profiler()

###############################################################################

# namespace

# an even simpler form of the 'Record' (or 'types.SimpleNamespace') class
# to make sub-namespaces within the module
#
# (I don't think this normal Python practise, but it works, except for __doc__)
# (although __doc__seems to work in Python 3.9)

class namespace(object):

  def __init__(self, name, vs):
    self.__name = name
    self.__dict__.update(**vs)

  def __repr__(self):
    return '<' + self.__class__.__name__ + ' ' + repr(self.__name) + '>'

def make_namespace(name, vs):
  return namespace(name, vs)

  ## or attempt to make a module:
  #r = type(enigma)("enigma." + name)
  #r.__dict__.update(**vs)
  #return r

###############################################################################

# NOTE: template_system is in testing, interface may change

# template system problems

def __template_system():
  __doc__ = """
  Solve a template system.

  Functions provided:

    template_system.solve()
  """

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


template_system = make_namespace('template_system', __template_system())

###############################################################################

# NOTE: grouping is in testing, interface may change

# grouping problems

def __grouping():
  __doc__ = """
  Grouping puzzles provide a collection of lists of values, and
  require the solver to form groups of values (usually disjoint groups
  with one element from each of the lists) that satisfy some condition
  (often that each pair of elements in the group shares exactly some
  given number of letters.

  Functions provided:

    Simple grouping problems:
      grouping.groups() - find groups
      grouping.output_groups() - output groups
      grouping.solve() - find and output groups

    Grouping problems with \"gangs\":
      grouping.gang() - construct a gang
      grouping.gangs() - find gangs
      grouping.output_gangs() - output gangs
      grouping.solve_gangs() - find and output gangs

    Useful selection functions:
      grouping.letters()
      grouping.share_letters()
  """

  def groups(vs, fn, s=[]):
    """
    group the lists of elements in <vs> into groups (one element from each list)
    such that the values in the groups satisfy the selection function <fn>

    returns a sequence of groups.
    """
    # are we done?
    if not vs[0]:
      yield tuple(s)
    else:
      # otherwise choose the next group to go with category 0
      for v in cproduct(enumerate(x) for x in vs[1:]):
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
    """
    output a sequence of groups <gs>
    """
    for g in gs:
      print(str.join(sep, g))
    print(end)

  def solve(vs, fn=true, sep=", ", end=""):
    """
    group the lists of elements in <vs> into groups (one element from each list)
    such that the values in the groups satisfy the selection function <fn>,
    and as each collection of groups is found output them.
    """
    for gs in groups(vs, fn):
      output_groups(gs, sep, end)


  # a k-gang has a leader x, and k followers chosen from a sequence ys
  # pairwise they satisfy the selection function <fn>
  # return a set of followers for leader x
  def gang(k, x, ys, fn):
    """
    select a <k>-gang for leader <x> by choosing <k> followers from <ys>.
    The selection function holds for each (<x>, <y>) pair and for the
    entire group of <k> followers.

    returns the group of <k> followers for leader <x>.
    """
    # select possible followers
    for vs in subsets((y for y in ys if fn(x, y)), size=k):
      if fn(*vs):
        yield vs

  # find multiple k-gangs for leaders in xs, followers in ys
  def gangs(k, xs, ys, fn, gs=[]):
    """
    form the elements of <xs> and <ys> into a collection of <k>-gangs, where
    each gang has a leader chosen from <xs> and <k> followers chosen from <ys>,
    such that the selection function <fn> is satisfied for each (<leader>, <follower>)
    pair, and also is satisfied pairwise for the entire group of <k> followers.

    returns the groups of <k> followers for each leader in <xs>.
    """
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
      print(x + ": " + str.join(sep, y))
    print(end)


  def solve_gangs(k, xs, ys, fn, sep=", ", end=""):
    for gs in gangs(k, xs, ys, fn):
      output_gangs(xs, gs)

  # useful selection functions

  # return the set of letters in a string
  @cached
  def letters(s):
    return set(x for x in s.lower() if x.isalpha())

  # return a check function that checks each pair of values shares exactly <k> letters
  def share_letters(k, cache=1):
    """
    return a function that checks each pair of values passed shares
    exactly <k> letters.

    use the <cache> parameter to control whether the function is
    cached or not.
    """
    fn = ((lambda x: x == k) if isinstance(k, int) else k)
    # check each pair of values shares exactly <k> different letters
    def check(*vs):
      return all(fn(len(letters(a).intersection(letters(b)))) for (a, b) in itertools.combinations(vs, 2))
    return (cached(check) if cache else check)

  # return the namespace
  return locals()


grouping = make_namespace('grouping', __grouping())

###############################################################################

# matrix routines (see Enigma 287)

# given two matrices A and B, returns (det(A), X) st A * X = B
# A must be square, and the elements must support __truediv__
def _matrix_gauss(A, B):
  n = len(A)
  p = len(B[0])
  det = 1

  for i in xrange(0, n - 1):
    k = i
    for j in xrange(i + 1, n):
      if abs(A[j][i]) > abs(A[k][i]):
        k = j

    if k != i:
      (A[i], A[k]) = (A[k], A[i])
      (B[i], B[k]) = (B[k], B[i])
      det = -det

    for j in xrange(i + 1, n):
      t = A[j][i] / A[i][i] # note use of /
      for k in xrange(i + 1, n):
        A[j][k] -= t * A[i][k]
      for k in xrange(p):
        B[j][k] -= t * B[i][k]

  for i in xrange(n - 1, -1, -1):
    for j in xrange(i + 1, n):
      t = A[i][j]
      for k in xrange(p):
        B[i][k] -= t * B[j][k]

    t = 1 / A[i][i] # note use of /
    det *= A[i][i]
    for j in xrange(p):
      B[i][j] *= t

  return (det, B)

# solve a system of linear equations
def _matrix_linear(A, B, n, m, valid):
  # for each column i
  for i in itertools.count(0):
    if not (i < m): break
    if n < m: raise ValueError("incomplete")

    # choose the row with the largest value in the column i
    j = max(xrange(i, n), key=(lambda j: abs(A[j][i])))

    # if necessary bring it to row i
    if j != i: (A[i], A[j], B[i], B[j]) = (A[j], A[i], B[j], B[i])

    # scale equation i so the co-efficient in column i is 1
    v = A[i][i]
    if v == 0: raise ValueError("incomplete")
    if v != 1:
      for k in xrange(i, m):
        A[i][k] /= v
      B[i] /= v

    # eliminate co-efficients in column i
    rs = list()
    for j in xrange(0, n):
      if j != i:
        t = A[j][i]
        if t != 0:
          for k in xrange(i, m):
            A[j][k] -= t * A[i][k]
          B[j] -= t * B[i]
          # if all coefficients in row j are 0
          if all(A[j][k] == 0 for k in xrange(0, m)):
            if B[j] == 0:
              # mark the row for deletion
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

  # apply solution validation?
  if valid:
    vs = list()
    for v in B:
      # valid can raise a ValueError, or return None
      x = valid(v)
      if x is None: raise ValueError("invalid value: " + repr(v))
      vs.append(x)
    return vs

  # otherwise just return B
  return B

# helper function for creating a matrix of linear equations
def _matrix_equation(sym, n, coeffs, k, z):
  row = [z] * n
  # if coeffs is a dictionary
  if isinstance(coeffs, dict):
    for (c, v) in coeffs.items():
      row[sym[c]] += v
  else:
    # if coeffs is a sequence of (symbol, value) pairs
    try:
      for (c, v) in coeffs:
        row[sym[c]] += v
    except ValueError:
      # otherwise, just a sequence of symbols
      for c in coeffs:
        row[sym[c]] += 1
  # return the (<coefficients>, <value>) pair
  return (tuple(row), k)

# the Matrix class replaces the functions in the matrix namespace
# (for a more powerful Matrix class, see sympy)
class Matrix(list):
  "A class for manipulating 2 dimensional matrices."

  def __init__(self, rows, field=None):
    """
    create a matrix with rows from iterator <rows>.

    each row should have the same number of elements.
    """
    ncols = None
    for row in rows:
      row = list(row)
      if ncols is None:
        ncols = len(row)
      elif len(row) != ncols:
        raise ValueError("Matrix: all rows should have the same length")
      self.append(row)
    self.field = field

  def rows(self):
    "an iterator that returns the rows of a matrix"
    return self

  def cols(self):
    "an iterator that returns the columns of a matrix"
    return zip(*self)

  def nrows(self):
    "return the number of rows in a matrix"
    return len(self)

  def ncols(self):
    "return the number of columns in a matrix"
    if self: return len(self[0])

  def get_field(self):
    """
    return the field of the elements, or if none was provided
    use a rational implementation provided by Rational()
    """
    if self.field is None: self.field = Rational()
    return self.field

  def map2d(self, f):
    "map the function <f> over the matrix"
    return Matrix(((f(x) for x in row) for row in self), field=self.field)

  def add(self, other):
    "return a new matrix that is the result of adding a matrix to this one"
    return Matrix((((a + b for (a, b) in zip(r1, r2)) for (r1, r2) in zip(self, other))), field=self.field)

  def __add__(self, other):
    if isinstance(other, Matrix):
      # matrix addition
      return self.add(other)
    else:
      # constant addition
      return self.map2d(lambda x: x + other)

  __radd__ = __add__

  def __neg__(self):
    return self.map2d(lambda x: -x)

  def sub(self, other):
    "return a new matrix that is the result of subtracting a matrix from this one"
    return Matrix((((a - b for (a, b) in zip(r1, r2)) for (r1, r2) in zip(self, other))), field=self.field)

  def __sub__(self, other):
    if isinstance(other, Matrix):
      return self.sub(other)
    else:
      return self.map2d(lambda x: x - other)

  __rsub__ = lambda self, other: -self + other

  def multiply(self, other):
    "return a new matrix that is the result of multiplying this matrix by another"
    tr_other = other.transpose()
    return Matrix((((sum(a * b for (a, b) in zip(row, col)) for col in tr_other) for row in self)), field=self.field)

  def __mul__(self, other):
    if isinstance(other, Matrix):
      # matrix multiplication
      return self.multiply(other)
    else:
      # constant multiplication
      return self.map2d(lambda x: x * other)

  __rmul__ = __mul__

  def transpose(self):
    "return the transposition of the matrix"
    return Matrix(zip(*self), field=self.field)

  def gauss(self, B=None):
    """
    return (det(A), X) where A * X = B.

    if B is not specified a suitably sized identity matrix is used and
    X is the inverse of A.
    """
    n = self.nrows()
    m = self.ncols()
    F = self.get_field()

    # check matrix is square
    assert n == m, "gauss: non-square matrix"

    # if B is None, use the identity matrix
    if B is None: B = Matrix.identity(n, m)

    # convert A and B (so that elements support __truediv__)
    A = self.map2d(F)
    B = Matrix(B).map2d(F)

    # solve it
    try:
      (d, X) = _matrix_gauss(A, B)
      return (d, Matrix(X, field=self.field))
    except ZeroDivisionError:
      return (0, None)

  def inv(self):
    "return the inverse of the matrix"
    (d, X) = self.gauss()
    return X

  def det(self):
    "return the determinant of the matrix"
    (d, X) = self.gauss()
    return d

  def linear_solve(self, B=0, valid=None):
    """
    solve a system of linear equations.

      A x = B

    <A> is the matrix of coefficients of the variables (n equations in m variables)
    <B> is the the vector of constants (a single value will be replicated)

    If the system is underspecified an "incomplete" error is raised.
    If the system is inconsistent an "inconsistent" error is raised.

    Otherwise a sequence of the solution values x is returned.
    """
    n = self.nrows()
    m = self.ncols()
    F = self.get_field()

    # if B is a scalar value, make an appropriate length vector
    if not isinstance(B, Sequence): B = [B] * n

    # construct the matrices into the specified field
    A = self.map2d(F)
    B = list(map(F, B))

    # run the solver
    return _matrix_linear(A, B, n, m, valid)

  # alternative names
  columns = cols
  determinant = det
  inverse = inv

  @classmethod
  def create(cls, nrows, ncols, k=0, field=None):
    """
    create a matrix with <nrows> rows and <ncols> columns.

    initially filled out with value <k>, which may be a constant
    or a function: k(r, c).
    """
    self = list()
    if callable(k):
      for r in xrange(nrows):
        self.append(list(k(r, c) for c in xrange(ncols)))
    else:
      for r in xrange(nrows):
        self.append([k] * ncols)
    return cls(self, field=field)

  @classmethod
  def identity(cls, nrows, ncols, field=None):
    "create an identity matrix"
    return cls.create(nrows, ncols, (lambda r, c: int(r == c)), field=field)

  # this works the same as matrix.linear used to ...
  @classmethod
  def linear(cls, A, B=None, field=None, valid=None, F=None):
    """
    solve a system of linear equations.

      A x = B

    <A> is the matrix of coefficients of the variables (n equations in m variables)
    <B> is the the vector of constants (a sequence of a single value that will be replicated)
    <field> is the field to operate over (which must support __truediv__)

    If <field> is not specified an implementation of the rational numbers
    will be used (by calling Rational()).

    If the system is underspecified an "incomplete" error is raised.
    If the system is inconsistent an "inconsistent" error is raised.

    Otherwise a sequence of the solution values x is returned (which will
    be in the specified field).

    The rows of the matrix of coeffecients and constants can be
    specified as:

      A = (row, row, row, ...) B = (const, const, const, ...)
      A = (row, row, row, ...) B = const # if all consts are equal
      A = ((row, const), (row, const), ...)
    """
    field = (field or F)

    # if B is not specified, then assume we've been supplied a list of
    # (row, constant) pairs, where A is made by collecting the rows,
    # and B the constants
    if B is None: (A, B) = zip(*A)

    # construct the matrix A
    A = cls(A, field=field)

    # and solve the set of equations
    return A.linear_solve(B, valid=valid)


  # create a function that creates (row, const) pairs suitable for construction
  # matrix A in a call to linear
  # symbols = sequence of symbols used in the system of equations
  # k = default constant (if none is specified)
  @classmethod
  def equation(cls, symbols, k=0, z=0):
    """
    create a function that can be used to create (row, const) values,
    suitable for constructing the matrix A used in Matrix.linear().

    symbols = sequence of symbols used in the system of equations
    k = default constant

    >>> eq = Matrix.equation("abcd", 42)
    >>> eq("acd")
    ((1, 0, 1, 1), 42)
    >>> eq(dict(a=1, b=2, c=3), 19)
    ((1, 2, 3, 0), 19)
    """
    # map symbols to indices
    sym = dict((s, i) for (i, s) in enumerate(symbols))
    return (lambda cs, k=k, z=z: _matrix_equation(sym, len(sym.keys()), cs, k, z))

# for backward compatibility allow matrix.{linear, create, equation} to work
matrix = Matrix

###############################################################################

# some handy development routines

# compare version numbers (numeric components separated by non-numeric components)
def compare_versions(x, y):
  fn = lambda s: tuple(map(int, re.split(r'[^\d]+', s.split(None, 1)[0])))
  return compare(fn(x), fn(y))

# require version to be at least specified value
# e.g.:
#   require("enigma.version", "2022-12-05")
#
# this works for versions with numeric components separated by non-numeric components
# so: require("sys.version", "3.10") will raise an exception on Python 3.9
def require(key, value=None, cmp=compare_versions):
  (mod, k) = key.split('.')
  try:
    v = getattr(sys.modules[mod], k)
    if callable(v): v = v()
  except (KeyError, AttributeError, ValueError):
    raise ValueError(str.format("unable to extract {key!r}", key=key))
  if value is not None and cmp(v, value) < 0:
    raise ValueError(str.format("version mismatch {key!r} = {v!r}; require >= {value!r}", key=key, v=v, value=value))
  return v

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
def status(fn, at_exit=0):
  status.fn = fn
  if at_exit: atexit.register(fn)

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

    code = import_fn("code")
    readline = import_fn("readline")
    import_fn("rlcompleter")
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

def parsefile(path, args=None, interleave=None, string=None):

  # parse from a file or string
  def parse(f):
    import shlex
    # parse it removing whitespace, comments, quotes
    lexer = shlex.shlex(f, posix=1)
    lexer.whitespace_split = True
    return list(lexer)

  if path == '<string>':
    words = parse(string)
  else:
    # it should be a file
    with open(path, 'r') as f:
      words = parse(f)

  cmd = words.pop(0)

  if interleave is None:
    interleave = (cmd not in { 'SubstitutedExpression', 'SubstitutedDivision' })

  def divide(s, fn=(lambda s: s.startswith('-'))):
    for (i, x) in enumerate(s):
      if not fn(x):
        return (s[:i], s[i:])
    return (s, ())

  # insert any extra args
  if args:
    if interleave:
      ((s1, s3), (s2, s4)) = (divide(words), divide(args))
      words = flatten([s1, s2, s3, s4])
    else:
      words = flatten([words, args])

  return (cmd, words)


_run_exit = None

# run command line arguments
# always returns None, but sets _run_exit
@static(alias={ 'Alphametic': 'SubstitutedExpression' })
def run(cmd, *args, **kw):
  """
  run with command line arguments

  <cmd> can be a class in enigma.py that accepts a command line,
  or it can be a run file, Python program or other script

  <args> are the command line arguments to be provided

  additional options are:

    timed - if set, time the execution of <cmd>
    repeat - for repeated runs (usually for timing purposes)
    flags - 'p' = enable prompts, 'v' = enable verbose
    interpreter - interpreter to use
    verbose - enable informational output
  """
  global _run_exit, _PY_ENIGMA
  _run_exit = None

  # cmd could be a list of [cmd, args]
  if not args and not isinstance(cmd, basestring) and isinstance(cmd, Sequence):
    args = list(cmd)
    cmd = args.pop(0)
  # make sure arguments are strings
  args = list(str(x) for x in args)

  verbose = kw.get('verbose') or ('v' in _PY_ENIGMA)
  timed = kw.get('timed')
  flags = kw.get('flags', '')
  interp = kw.get('interpreter')
  #interact = kw.get('interact')

  if 'repeat' in kw:
    n = int(kw.pop('repeat', 1))
    for _ in irange(1, n):
      run(cmd, *args, **kw)
    return

  # enabling 'prompt' disables timing
  if 'p' in _PY_ENIGMA or 'p' in flags: timed = 0
  saved = None

  # an alternative way to run a solver is to use "-r[t] / --run[:timed] <file> <additional-args>"
  if cmd.startswith('-r') or cmd.startswith('--run'):
    if cmd == "-rt" or cmd == "--run:timed": timed = 1
    if not args: raise ValueError("missing argument for -r / --run")
    (cmd, args) = (args[0], args[1:])
  elif cmd.startswith('-'):
    if verbose: printf("run: unrecognised command \"{cmd}\"")
    return

  # if cmd names a file
  if os.path.isfile(cmd):
    if verbose: printf("run: attempting to run file \"{cmd}\"")
    if timed and not isinstance(timed, basestring): timed = os.path.basename(cmd)
    if (not interp) and cmd.endswith(".run"):
      # *.run => treat it as a run file
      (cmd, args) = parsefile(cmd, args)
    else:
      if (not interp) and any(cmd.endswith(x) for x in (".py", ".py2", ".py3")):
        # use runpy for *.py
        run_path = import_fn('runpy.run_path')
        get_argv(force=1, args=args)
        sys.argv = [cmd] + list(args)
        if flags:
          saved = [_PY_ENIGMA]
          _PY_ENIGMA = join(sorted(uniq(_PY_ENIGMA + flags)))
        try:
          if timed: timed = Timer(name=timed)
          r = run_path(cmd, run_name=kw.get('run_name', '__main__'))
          if timed: timed.report()
        finally:
          if saved: [_PY_ENIGMA] = saved
        _run_exit = (0 if r else -1)
      else:
        import shlex
        import subprocess
        # attempt to use a shebang line (see: run.py)
        path = os.path.abspath(cmd)
        if interp:
          cmd = interp.strip()
          # if it is a run file...
          if path.endswith(".run"):
            args = ['--run', path] + args
            path = enigma.__file__
        else:
          with open(path, 'r') as fh:
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
        _run_exit = -1
        try:
          if verbose: printf("run: calling command {cmd}")
          # use elapsed time for subprocesses, rather than process time
          if timed: timed = Timer(name=timed, timer="E")
          _run_exit = subprocess.call(cmd)
          if timed: timed.report()
        finally:
          if saved: [PY_ENIGMA] = saved
      if verbose: printf("run: _run_exit = {_run_exit}")
      return
  else:
    if verbose: printf("run: not a file \"{cmd}\"")

  # if cmd names a class[.method]
  (obj, _, fn_name) = cmd.partition('.')
  if not fn_name: fn_name = 'command_line'
  fn_name = "run_" + fn_name
  fn = globals().get(run.alias.get(obj, obj))
  if fn:
    fn = getattr(fn, fn_name, None)
    if fn:
      if timed and not isinstance(timed, basestring): timed = 'timing'
      if flags:
        saved = [_PY_ENIGMA]
        _PY_ENIGMA = join(sorted(uniq(_PY_ENIGMA + flags)))
      try:
        if verbose: printf("run: calling function {fn.__name__} ...")
        if timed: timed = Timer(name=timed)
        _run_exit = (fn(list(args)) or 0)
        if timed: timed.report()
      except Exception as e:
        printf("run: FAILURE in {cmd} ...\n\nException details:\n{e}\n")
        import traceback
        traceback.print_exc(limit=0)
        _run_exit = -1
        if timed: timed.report()
      finally:
        if saved: [_PY_ENIGMA] = saved
      if verbose: printf("run: _run_exit = {_run_exit}")
      return
    else:
      printf("enigma.py: {obj}.{fn_name}() not implemented")
  else:
    if verbose: printf("run: not a class[.method] \"{cmd}\"")

  # if we get this far we can't find the solver
  printf("enigma.py: unable to run \"{cmd}\"")
  _run_exit = -1
  if verbose: printf("run: _run_exit = {_run_exit}")
  return

def timed_run(*args):
  run(*args, timed=1)

###############################################################################

# implementation of command line options

# help (-h)
def _enigma_help():
  print('command line arguments:')
  print('  <class> <args> = run run_command_line(<args>) method on class')
  print('  [-r[t] | --run[:timed]] <file> [<additional-args>] = run the solver and args specified in <file>')
  print('  -t[v] = run tests [v = verbose]')
  print('  -u[cdr[v]] = check for updates [c = only check, d = always download, r = rename after download, v = verbose]')
  print('  -p[ru[v]] = use pip for updates [r = show requirements, u = install/update, v = verbose]')
  print('  -h = this help')


# run doctests (-t)
def _enigma_test(verbose=0):
  doctest = import_fn("doctest")
  print("[testing, testing ...]")
  r = doctest.testmod(enigma, verbose=verbose)
  printf("[testing complete]")
  return r

# check for updates to enigma.py (-u)
# check = only check the current version
# download = always download the latest version
# verbose = 0 (quiet), 1 (normal), 2 (verbose)
def __enigma_update(url, check=1, download=0, rename=0, verbose=1):

  if verbose > 1: printf("update: url = {url}")

  # import urlopen from the appropriate module
  urlopen = import_fn(("urllib2" if _python == 2 else "urllib.request") + '.urlopen')

  # py-enigma-version.txt = "<version>[ <md5sum>]"
  if verbose > 1: printf("update: downloading [ {url}py-enigma-version.txt ] ...")
  u = urlopen(url + 'py-enigma-version.txt')
  readline = lambda f: f.readline(64).decode().strip()
  # line 1 = "<version>"
  v = readline(u)
  if verbose > 1: printf("update: version = {v}")
  cksum = None
  # line 2 = "md5=<md5sum>"
  s = readline(u)
  if s:
    if s.startswith("md5="): cksum = s[4:]
    if verbose > 1: printf("update: cksum = {cksum}")
  if verbose > 0: printf("latest version is {v}")

  if (__version__ < v and not check) or download:
    hashlib = import_fn("hashlib")
    h = hashlib.md5()
    name = v + '-enigma.py'
    if verbose > 0: printf("downloading latest version to \"{name}\"")
    with open(name, 'wb') as f:
      if verbose > 1: printf("update: downloading [ {url}enigma.py ] ...")
      u = urlopen(url + 'enigma.py')
      while True:
        if verbose > 0: print('.', end='')
        data = u.read(8192)
        if not data: break
        f.write(data)
        h.update(data)
    if verbose > 0: printf("{nl}download complete")
    if verbose > 1: printf("update: download checksum = {h}", h=h.hexdigest())
    if cksum:
      if cksum == h.hexdigest():
        if verbose > 0: printf("checksum verified")
      else:
        raise IOError("checksum failure")
    if rename:
      dst = __file__
      if verbose > 0: printf("renaming \"{name}\" to \"{dst}\"")
      os.rename(name, dst)
  elif __version__ < v:
    if verbose > 0: print("enigma.py is NOT up to date")
  else:
    if verbose > 0: print("enigma.py is up to date")
    if cksum:
      # verify checksum of local file matches
      hashlib = import_fn("hashlib")
      h = hashlib.md5()
      with open(__file__, 'rb') as f:
        if verbose > 1: printf("update: verifying checksum for \"{__file__}\"")
        while True:
          data = f.read(8192)
          if not data: break
          h.update(data)
      if cksum == h.hexdigest():
        if verbose > 0: printf("checksum verified")
      else:
        printf("WARNING: checksum mismatch for \"{__file__}\"")

@static(url='https://raw.githubusercontent.com/enigmatic-code/py-enigma/master/')
# was: @static(url='http://www.magwag.plus.com/jim/')
def _enigma_update(url=None, check=1, download=0, rename=0, quiet=0, verbose=0):
  """
  check enigma.py version, and download the latest version if
  necessary.

  this function is called by the -u command line option.

    % python enigma.py -u
    [enigma.py version 2019-07-06 (Python 3.7.5)]
    checking for updates...
    latest version is 2019-07-06
    enigma.py is up to date

  check - set to check current version against latest
  download - set to always download latest version
  rename - set to rename downloaded file to enigma.py
  """
  if not quiet: print('checking for updates...')

  if url is None: url = _enigma_update.url

  try:
    v = (2 if verbose else 0 if quiet else 1)
    __enigma_update(url, check=check, download=download, rename=rename, verbose=v)
  except IOError as e:
    print(e)
    printf("ERROR: failed to download update from {_enigma_update.url}")

# interaction with pip (if installed)
@static(
  req="enigma@https://github.com/enigmatic-code/py-enigma/tarball/master",
  ver='2.7.' + join(x for x in __version__ if x.isdigit()),
)
def _enigma_pip(requirements=0, update=0, verbose=0):
  req = _enigma_pip.req

  # output requirements for enigma.py
  if requirements:
    printf()
    if verbose: printf("# install enigma.py from GitHub")
    printf("{req}")
    printf()
    return

  # update enigma.py using pip
  if update:
    import subprocess
    cmd = [sys.executable, '-m', 'pip', 'install', '-U', req]
    if verbose: printf(">>> {cmd}", cmd=join(cmd, sep=' '))
    subprocess.call(cmd)
    return

__doc__ += """

COMMAND LINE USAGE:

enigma.py has the following command-line usage:

  % python3 enigma.py

    The reports the current version of the enigma.py module, and the
    current python version:

      % python enigma.py
      [enigma.py version {version} (Python {python})]

      % python3 enigma.py
      [enigma.py version {version} (Python {python3})]


  % python3 enigma.py -t[v]

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


  % python3 enigma.py -u[cdr[v]]

    The enigma.py module can be used to check for updates. Running
    with the -u flag will check if there is a new version of the
    module available (requires a functioning internet connection), and
    if there is it will download it.

    If the module can be updated you will see something like this:

      % python3 enigma.py -ur
      [enigma.py version 2013-09-10 (Python {python3})]
      checking for updates...
      latest version is {version}
      downloading latest version to "{version}-enigma.py"
      ........
      download complete
      checksum verified
      renaming "{version}-enigma.py" to "enigma.py"

    Note that the updated version is downloaded to a file named
    "<version>-enigma.py" in the current directory. You can then
    upgrade by renaming this file to "enigma.py" (this will happen
    automatically if the 'r' flag is passed).

    If you are running the latest version you will see something like
    this:

      % python3 enigma.py -u
      [enigma.py version {version} (Python {python3})]
      checking for updates...
      latest version is {version}
      enigma.py is up to date

    If -uc is specified then the module will only check if an update
    is available, it won't download it.

    If -ud is specified then the latest version will always be
    downloaded.


  % python3 -m enigma -p[ru[v]]

    This provides integration with Python's pip package manager, to
    allow installing/updating enigma.py via pip directly from GitHub
    (so you will also need git installed), and you will probably want
    to use "-m enigma" on the command line (once it is installed)
    rather than the path of the enigma.py file.

    The -r command will output an entry suitable for incorporation
    into a requirements.txt file:

      % python3 -m enigma -pr
      [enigma.py version {version} (Python {python3})]

      {pip_req}

    The -u flag will use pip to install/upgrade enigma.py:

      % python3 -m enigma -pu
      [enigma.py version {version} (Python {python3})]
      Collecting {pip_req}
      ...
      Successfully installed enigma-{pip_version}


  % python3 enigma.py -h

    Provides a quick summary of the command line usage:

      % python3 enigma.py -h
      [enigma.py version {version} (Python {python3})]
      command line arguments:
        <class> <args> = run run_command_line(<args>) method on class
        [-r | --run] <file> [<additional-args>] = run the solver and args specified in <file>
        -t[v] = run tests [v = verbose]
        -u[cdr[v]] = check for updates [c = only check, d = always download, r = rename, v = verbose]
        -p[ru[v]] = use pip for updates [r = show requirements, u = install/update, v = verbose]
        -h = this help

  Solvers that support the run_command_line() class method can be invoked
  directly from the command line like this:

    python3 enigma.py <class> <args> ...

    Supported solvers are:
      SubstitutedSum
      SubstitutedDivision
      SubstitutedExpression
      SubstitutedExpression.split_sum

    For example, Enigma 327 can be solved using:

    % python3 enigma.py SubstitutedSum "KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE"
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5


    Enigma 440 can be solved using:

    % python3 enigma.py SubstitutedDivision "????? / ?x = ??x" "??? - ?? = ?" "" "??? - ??x = 0"
    ????? / ?x = ??x (rem 0) [??? - ?? = ?, None, ??? - ??x = 0]
    10176 / 96 = 106 (rem 0) [101 - 96 = 5, None, 576 - 576 = 0] / x=6
    [1 solution]


    Enigma 1530 can be solved using:

    % python3 enigma.py SubstitutedExpression "TOM * 13 = DALEY"
    (TOM * 13 == DALEY)
    (796 * 13 == 10348) / A=0 D=1 E=4 L=3 M=6 O=9 T=7 Y=8
    [1 solution]


    Alternatively the arguments to enigma.py can be placed in a text file
    and then executed with the --run / -r command, for example:

    % python3 enigma.py --run enigma327.run
    (KBKGEQD + GAGEEYQ + ADKGEDY = EXYAAEE)
    (1912803 + 2428850 + 4312835 = 8654488) / A=4 B=9 D=3 E=8 G=2 K=1 Q=0 X=6 Y=5

""".format(
  version=__version__, python='2.7.18', python3='3.11.5',
  pip_version=_enigma_pip.ver, pip_req=_enigma_pip.req,
)

def _enigma_main(args=None):
  if args is None: args=get_argv()

  # allow solvers to run from the command line:
  #   % python enigma.py <class> <args> ...
  # or put all the arguments into a file and use:
  #   % python enigma.py -r <file> <additional-args>
  #   % python enigma.py --run <file> <additional-args>
  #   % python enigma.py <file> <additional-args>
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
  if 'h' in args: _enigma_help()

  # -t => run tests
  # -tv => in verbose mode
  if 't' in args: _enigma_test(verbose=('v' in args['t']))

  # -u => check for updates, and download newer version
  # -uc => just check for updates (don't download)
  # -ud => always download latest version
  # -u[d]r => replace enigma.py with downloaded file
  if 'u' in args:
    kw = dict((w, w[0] in args['u']) for w in ['check', 'download', 'rename', 'quiet', 'verbose'])
    _enigma_update(**kw)

  # -p = install enigma.py via pip
  # -pr => output requirements.txt entry
  if 'p' in args:
    kw = dict((w, w[0] in args['p']) for w in ['requirements', 'update', 'verbose'])
    _enigma_pip(**kw)

def _namecheck(name, verbose=0):
  if verbose or ('v' in _PY_ENIGMA): printf("[_namecheck] checking \"{name}\"")
  return name == "__main__" or name == "<run_path>"

if _namecheck(__name__): _enigma_main()
