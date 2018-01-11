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
Takie ekstrapolowanie działania nazwya się w NumPy boradcasting.

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

Okej, okej, ale czy ktoś na początku nie wspominał, że NumPy implementuje takie fajne obiekty `ndarray`, gdzie występuje magiczna literka `n`...?
Fakt! Co to bylo by za narzedzie gdyby maks na co by je było stać to tablice trójwymiarowe? NumPy mnoży dowolnej wielkości tablice
```markdown
>>> c = np.arange(18).reshape(2,3,3)
>>> c
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> d = np.arange(24).reshape(2,2,3,2)
>>> d
array([[[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]]],


       [[[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]]])
>>> np.dot(c, d)
array([[[[[  10,   13],
          [  28,   31]],

         [[  46,   49],
          [  64,   67]]],


        [[[  28,   40],
          [ 100,  112]],

         [[ 172,  184],
          [ 244,  256]]],


        [[[  46,   67],
          [ 172,  193]],

         [[ 298,  319],
          [ 424,  445]]]],



       [[[[  64,   94],
          [ 244,  274]],

         [[ 424,  454],
          [ 604,  634]]],


        [[[  82,  121],
          [ 316,  355]],

         [[ 550,  589],
          [ 784,  823]]],


        [[[ 100,  148],
          [ 388,  436]],

         [[ 676,  724],
          [ 964, 1012]]]]])
```
Można zapytać co tu się najlepszego odwaliło?! Odpowiedź jest prosta, a z pomocą przyjdzie nam `shape`
```markdown
>>> c.shape
(2, 3, 3)
>>> d.shape
(2, 2, 3, 2)
>>> np.dot(c, d).shape
(2, 3, 2, 2, 2)
```
Widzisz wzór? Jeśli nie, to pozwól że go wyartukułuję dla Twej wygody
```markdown
>>> c.shape
(X..., A)
>>> d.shape
(Y..., A, B)
>>> np.dot(c, d).shape
(X..., Y..., B)
```
Teraz jasne? W mnożeniu macierzy kasują się wymiary: ostatni pierwszej oraz przedostatni drugiej, a resztę zapsujemy kolejno i tak tworzy się wynik. Prawda, że prosta zasada? Jak cofniesz się do przykładów dwuwymiarowych to też zauważysz, że zależnośc jest spełniona.

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
