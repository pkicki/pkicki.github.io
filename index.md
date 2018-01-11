### Python

Zanteresowanych zgłębianiem tajemnic Pythona zapraszam do przeszukiwania internetu, tutaj skupimy się na Numpy. 

### Numpy - łagodne wprowadzenie

NumPy jest przekozacką biblioteką do obliczeń naukowych. Szczególnie sprawdza się przy operowaniu na wielowymiarowych tablicach `ndarray` (N dimenional array). W swoim pakiecie zawiera też przydatne przy przetwarzaniu sygnałów funkcje, ale my skupimy się bardiej na tym co niezbędnei, by móc pisać właśne sieci neuronowe. Oczywiście moglibyśmy to osiągnąć korzystając z gołego Pyhona, a jakże, ale NumPy zapewnia, jakże pożądaną, skalowalność.

Aby rozpocząć zabawę z numpy wpisujemy w konsolę
```markdown
$ python
```
oczywiście $ to tylko znak konsoli Linuxowej (dla normalnego użytkownika). 
Następnie, już w konsoli Pythona dokonujemy importu naszej ulubonej biblioteki
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
Jak się później okaże `shape` będzie podstawą naszego debugu i jedynym srzymierzeńcem w próbie ogarnięcia co się z tymi macierzami dzieje.

Skoro znam kształty naszych macierzy, to nauczmy się je zmieniać! Do tego skorzystamy z funkcji `reshape(shape)`
```markdown
>>> b
array([3, 5, 7, 3, 2, 1])
>>> b = b.reshape((2,3))
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> a
array([ 0.,  1.,  2.,  3.,  4.,  5.])
>>> a = a.reshape(3,2)
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
```
Warto zwrócić uwagę jak pięknie nam NumPy organizuje macierz do wyśwetlenia, co dobitnie pokazuje różnicę pomiędzy `(3,2)`, a `(2,3)`.
Generalnie, zerowym wymiarem jest wysokość, a pierwszym szerokość.
UWAGA! W tej funkcji nie ma potrzeby zamykania pożądanego kształtu w tuple.

Okej, czas na trochę matematyki! Zobaczmy jakie działania są wspierane dla naszych ukochanych macierzy.
No to dodajmy te macierze
```markdown
>>> a + b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,2) (2,3)
```
Ups, no kto by się spodziewał... NumPy pilnuje czy ktoś tutaj ne chce wystrychnąć algebry na dudka! Dodajmy, więc coś co ma szansę być dodane
```markdown
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> a + b.T
array([[  3.,   4.],
       [  7.,   5.],
       [ 11.,   6.]])
```
Algebra nam mówi, że to już sens ma, a przy okazji dowiedzieliśmy się, że macierze można prosto transponnować dzięki `.T`.

No ale przyznasz drogi naukowcu (w końcu używasz biblioteki doobliczeń naukowych), że gdyby NumPy pozwalał tylko a to, na co pozwala algebra, to nie był by tak fajny, a jednak jest fajny. Spróbujmy więc czegoś z pozoru dziwnego
```markdown
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> f
array([1, 2])
>>> a.shape
(3, 2)
>>> f.shape
(2,)
>>> a + f
```
No nie, to nie może się udać! Jak dwuwymiarową macierz chcielibyśmy dodać do jedno wymi...
```markdown
array([[ 1.,  3.],
       [ 3.,  5.],
       [ 5.,  7.]])
```
To działa! Przyznasz nawet, mam nadzieję że dość intuicyjnie. Zgodził nam się wymiar `2` i dzięki temu każdy wiersz został powiększony o wektor `f`.
Algera oszukana! Ale czy zawsze nam się to uda?
```markdown
>>> a + np.array([1,2,3])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```
Otóż nie. Gdy wymiary nie pasują, nawet NumPy nic nie poradzi.
Okej, spróbujmy inaczej
```markdown
>>> t = np.array([[3], [4], [10]])
>>> t.shape
(3, 1)
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> t
array([[ 3],
       [ 4],
       [10]])
>>> a + t
array([[  3.,   4.],
       [  6.,   7.],
       [ 14.,  15.]])

```
Jak można się było spodziewać, działa!

Dodawanie, dodawaniem, no ale jest też drugie bardzo istotne działanie... Tak, chodzi o mnożenie macierzy!
No to do dzieła.
```markdown
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> a * b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,2) (2,3) 
```
Choć algebra powie nam "Ale jak to?!", NumPy powie "Nie tędy droga!".
Nasze ulubione mnożenie `*` nie zadziała! Otóź NumPy przyjmuje znak `*` jako operator mnożenia element po elemencie, z resztą zobacz sam.
```markdown
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> a * a
array([[  0.,   1.],
       [  4.,   9.],
       [ 16.,  25.]])
```
Jasne? No ja myślę!
Dobra, ale jak pomnożyć te macierze jak Pan Bóg przykzał?!
Otóż tak
```markdown
>>> a
array([[ 0.,  1.],
       [ 2.,  3.],
       [ 4.,  5.]])
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> np.dot(a, b)
array([[  3.,   2.,   1.],
       [ 15.,  16.,  17.],
       [ 27.,  30.,  33.]])
```
Ten wynik nie powinien już nikogo dziwić.
```markdown
>>> np.dot(b, a)
array([[ 38.,  53.],
       [  8.,  14.]])
```
Mówili, że mnożenie macierzy nie jest przemienne... Na szczęście czasami jest :D 
Oczywiście mnożenie przzez skalar działa równie dobrze
```markdown
>>> b * 7
array([[21, 35, 49],
       [21, 14,  7]])
```
No, pomnożyliśmy sobie, wracamy do roboty. Jakie jeszcze działania wspiera nasza biblioteka? Odpowiedź brzmi: logiczne. Popatrz tylko na te wspaniałe rezultaty
```markdown
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> b > 4
array([[False,  True,  True],
       [False, False, False]], dtype=bool)
>>> b == 3
array([[ True, False, False],
       [ True, False, False]], dtype=bool)
```
```markdown
```
```markdown
```
```markdown
```
```markdown
```
```markdown
```
