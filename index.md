### Python

Zanteresowanych zgłębianiem tajemnic Pythona zapraszam do przeszukiwania internetu, tutaj skupimy się na Numpy. 

### Numpy - łagodne wprowadzenie

Aby rozpocząć zabawę z numpy wpisujemy w konsolę
```markdown
$ python
```
oczywiście $ to tylko znak konsoli Linuxowej(dla normalnego użytkownika), a następnie, już w konsoli Pythona
```markdown
>>> import numpy as np
```
Dobra, czas na pierwsze tablice NumPy
```markdown
>>> a = np.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])
>>> b = np.array([3,5,7,3,2,1])
>>> b
array([3, 5, 7, 3, 2, 1])
```
Do tworzenia specjalnych tablic(wypełnionych wartościami 1 lub 0) można też użyć komend
```markdown
>>> c = np.zeros((2,3,2))
>>> c
array([[[ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.]],

       [[ 0.,  0.],
        [ 0.,  0.],
        [ 0.,  0.]]])
>>> d = np.ones((2, 2, 3))
>>> d
array([[[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]],

       [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]])
```
gdzie tuple (2,3,2) oraz (2,2,3) definiują wymiary naszych nowych tablic.
Jak zapewnie widzisz, elementy stworzonych tablic w kroku poprzednim i wcześniejszym różnią się!
```markdown
>>> type(a[0])
<type 'numpy.int64'>
>>> type(c[0][0][0])
<type 'numpy.float64'>
```
Jeśli chcielibyśmy by elementy naszych tablic otrzymały typ inny niż omyślny wystarczy użyć argumentu `dtype`
```markdown
>>> a = np.arange(6, dtype=float)
>>> a
array([ 0.,  1.,  2.,  3.,  4.,  5.])
>>> type(a[0])
<type 'numpy.float64'>
```
Okej, mamy różne różnowymiarowe tablice. Może jednak warto się upewnić jaki mają kształt, do tego służy pole `shape`
```markdown
>>> a.shape
(6,)
>>> b.shape
(6,)
>>> c.shape
(2, 3, 2)
>>> d.shape
(2, 2, 3)
```
```markdown
```
