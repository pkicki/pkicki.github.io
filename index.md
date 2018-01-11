### Python

Zainteresowanych zgłębianiem tajemnic Pythona zapraszam do przeszukiwania internetu, tutaj skupimy się na Numpy. 

### Numpy - łagodne wprowadzenie

NumPy jest przekozacką biblioteką do obliczeń naukowych. Szczególnie sprawdza się przy operowaniu na wielowymiarowych tablicach `ndarray` (N dimenional array). W swoim pakiecie zawiera też przydatne przy przetwarzaniu sygnałów funkcje, ale my skupimy się bardziej na tym co niezbędne, by móc pisać właśne sieci neuronowe. Oczywiście moglibyśmy to osiągnąć korzystając z gołego Pyhona, a jakże, ale NumPy zapewnia, jakże pożądaną, skalowalność.

Aby rozpocząć zabawę z numpy wpisujemy w konsolę
```markdown
$ python
```
oczywiście $ to tylko znak konsoli Linuxowej (dla normalnego użytkownika). <br />
Następnie, już w konsoli Pythona dokonujemy importu naszej ulubionej biblioteki
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
Do tworzenia specjalnych tablic (wypełnionych wartościami 1, 0 lub losowymi) można też użyć funkcji `np.zeros`, `np.ones`, `np.random.randn`
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
>>> e = np.random.randn(2,2)
>>> e
array([[ 0.05113948,  0.27751734],
       [-0.56604264,  0.19052047]])
```
gdzie tuple (2,3,2) oraz (2,2,3) definiują wymiary naszych nowych tablic.<br />
UWAGA! W funkcji `np.random.randn` nie można użyć tupli do definicji kształtu - wymagane są kolejne wymiary jako osobne argumenty.<br />
Jak zapewnie widzisz, elementy stworzonych tablic w kroku poprzednim i wcześniejszym różnią się!
```markdown
>>> type(a[0])
<type 'numpy.int64'>
>>> type(c[0][0][0])
<type 'numpy.float64'>
```
Jeśli chcielibyśmy by elementy naszych tablic otrzymały typ inny niż domyślny wystarczy użyć argumentu `dtype`
```markdown
>>> a = np.arange(6, dtype=float)
>>> a
array([ 0.,  1.,  2.,  3.,  4.,  5.])
>>> type(a[0])
<type 'numpy.float64'>
```

W Pythonie rządzą wycinki tablic, każdy to wie, a NumPy tylko to potwierdza
```markdown
>>> c
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> c[:,1]
array([[ 3,  4,  5],
       [12, 13, 14]])
>>> c[:,:,2]
array([[ 2,  5,  8],
       [11, 14, 17]])
```	
Tzw. slajsy(slice) są bardzo powszechne i niezwykle użyteczne
```markdown
>>> c[:,:,2] = 123
>>> c
array([[[  0,   1, 123],
        [  3,   4, 123],
        [  6,   7, 123]],

       [[  9,  10, 123],
        [ 12,  13, 123],
        [ 15,  16, 123]]])
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
Jak się później okaże `shape` będzie podstawą naszego debugu i jedynym sprzymierzeńcem w próbie ogarnięcia co się z tymi macierzami dzieje.

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
Warto zwrócić uwagę jak pięknie nam NumPy organizuje macierz do wyświetlenia, co dobitnie pokazuje różnicę pomiędzy `(3,2)`, a `(2,3)`.
Generalnie, zerowym wymiarem jest wysokość, a pierwszym szerokość.<br />
UWAGA! W tej funkcji nie ma potrzeby zamykania pożądanego kształtu w tuple.

Ponadto możemy dodawać do naszych tablic nowy wymiar, dzięki `np.newaxis`
```markdown
>>> c
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> c[:, np.newaxis]
array([[[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8]]],


       [[[ 9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]]]])
>>> c[:, np.newaxis].shape
(2, 1, 3, 3)
>>> c[:,:,:, np.newaxis]
array([[[[ 0],
         [ 1],
         [ 2]],

        [[ 3],
         [ 4],
         [ 5]],

        [[ 6],
         [ 7],
         [ 8]]],


       [[[ 9],
         [10],
         [11]],

        [[12],
         [13],
         [14]],

        [[15],
         [16],
         [17]]]])
>>> c[:,:,:, np.newaxis].shape
(2, 3, 3, 1)
```

Okej, czas na trochę matematyki! Zobaczmy jakie działania są wspierane dla naszych ukochanych macierzy.<br />
No to dodajmy te macierze
```markdown
>>> a + b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (3,2) (2,3)
```
Ups, no kto by się spodziewał... NumPy pilnuje czy ktoś tutaj nie chce wystrychnąć algebry na dudka! Dodajmy, więc coś, co ma szansę być dodane
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
Algebra nam mówi, że to już sens ma, a przy okazji dowiedzieliśmy się, że macierze można prosto transponować dzięki `.T`.

No ale przyznasz drogi naukowcu (w końcu używasz biblioteki do obliczeń naukowych), że gdyby NumPy pozwalał tylko na to, na co pozwala algebra, to nie byłby tak fajny, a jednak jest fajny. Spróbujmy więc czegoś z pozoru dziwnego
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
To działa! Przyznasz nawet, mam nadzieję, że dość intuicyjnie. Zgodził nam się wymiar `2` i dzięki temu każdy wiersz został powiększony o wektor `f`.<br />
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
Jak można się było spodziewać, działa!<br />
Takie ekstrapolowanie działania nazywa się w NumPy boradcasting.

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
Choć algebra powie nam "Ale jak to?!", NumPy powie "Nie tędy droga!".<br />
Nasze ulubione mnożenie `*` nie zadziała! Otóż NumPy przyjmuje znak `*` jako operator mnożenia element po elemencie, z resztą zobacz sam.
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
Jasne? No ja myślę!<br />
Dobra, ale jak pomnożyć te macierze jak Pan Bóg przykazał?!<br />
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
Mówili, że mnożenie macierzy nie jest przemienne... Na szczęście czasami jest :D <br />
Oczywiście mnożenie przez skalar działa równie dobrze
```markdown
>>> b * 7
array([[21, 35, 49],
       [21, 14,  7]])
```

Okej, okej, ale czy ktoś na początku nie wspominał, że NumPy implementuje takie fajne obiekty `ndarray`, gdzie występuje magiczna literka `n`...?<br />
Fakt! Co to byłoby za narzędzie, gdyby maksimum, na jakie by je było stać to tablice trójwymiarowe? NumPy mnoży dowolnej wielkości tablice
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
Można zapytać: Co tu się najlepszego odwaliło?! Odpowiedź jest prosta, a z pomocą przyjdzie nam `shape`
```markdown
>>> c.shape
(2, 3, 3)
>>> d.shape
(2, 2, 3, 2)
>>> np.dot(c, d).shape
(2, 3, 2, 2, 2)
```
Widzisz wzór? Jeśli nie, to pozwól, że go wyartykułuję dla Twej wygody
```markdown
>>> c.shape
(X..., A)
>>> d.shape
(Y..., A, B)
>>> np.dot(c, d).shape
(X..., Y..., B)
```
Teraz jasne? W mnożeniu macierzy kasują się wymiary: ostatni pierwszej oraz przedostatni drugiej, a resztę zapisujemy kolejno i tak tworzy się wynik. Prawda, że prosta zasada? Jak cofniesz się do przykładów dwuwymiarowych, to też zauważysz, że zależność jest spełniona.

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
Ktoś mógłby zapytać "Po co to?", już śpieszę z odpowiedzią
```markdown
>>> b
array([[3, 5, 7],
       [3, 2, 1]])
>>> b[b>4]
array([5, 7])
>>> np.where(b > 4, b, -10)
array([[-10,   5,   7],
       [-10, -10, -10]])
>>> b[b>4] = 10
>>> b
array([[ 3, 10, 10],
       [ 3,  2,  1]])
```
Jak widać dzięki operatorom logicznym możemy wybierać elementy macierzy spełniające warunek, oraz modyfikować je w zależności od spełnienia warunku, dzięki funkcji `where` lub indeksowaniu `b[b>4]`.

NumPy daje nam do dyspozycji kilka innych bardzo przydatnych funkcji `sum`, `min`, `max`, `mean`, `var`, `std`. Oczywiście przy ich pomocy możemy obliczyć kolejno: sumę, minimum, maksimum, średnią, wariancję i odchylenie standardowe.<br />
Jednak, co w nich najciekawsze, oprócz przydanego działania, to prezentują one znakomicie użycie parametru `axis` i to postaram się Ci na przykładzie `np.sum` pokazać
```markdown
>>> c
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> c.shape
(2, 3, 3)
>>> np.sum(c)
153
>>> np.sum(c, axis=0)
array([[ 9, 11, 13],
       [15, 17, 19],
       [21, 23, 25]])
>>> np.sum(c, axis=1)
array([[ 9, 12, 15],
       [36, 39, 42]])
>>> np.sum(c, axis=2)
array([[ 3, 12, 21],
       [30, 39, 48]])
>>> np.sum(c, axis=(0,1))
array([45, 51, 57])
>>> np.sum(c, axis=(0,2))
array([33, 51, 69])
>>> np.sum(c, axis=(1,2))
array([ 36, 117])
```
Dużo obliczeń, ale postaramy się sobie z nimi poradzić po kolei. Samo `sum` daje nam sumę wszystkich elementów. Jednakże, kiedy wyspecyfikujemy parametr `axis=0` otrzymujemy dwuwymiarową macierz! Dzieje się tak, dlatego, że sumowanie obywa się wzdłuż osi `0`, czyli dodajemy górną macierz 3x3 do dolnej -> `0+9=9`, `1+10=11`, ..., `8+17=25`.<br />
Dla osi 1 dodajemy poszczególne kolumny -> `0+3+6=9`, `1+4+7=12`, ..., `11+14+17=42`.<br />
Dla osi 2 dodajemy poszczególne wiersze -> `0+1+2=3`, `3+4+5=12`, ..., `15+16+17=48`.<br />
Możemy też na raz wykonać działanie względem kilku osi, podając `axis` jako tuple, i tak:<br />
(0,1) -> weźmy wynik sumy dla `axis=0` i zsumujmy kolumny,<br />
(0,2) -> weźmy wynik sumy dla `axis=0` i zsumujmy wiersze,<br />
(1,2) -> weźmy wynik sumy dla `axis=1` i zsumujmy wiersze.<br />

Dobra wystarczy jeszcze tylko się nauczyć składać kilka macierzy w jedną i będziemy mistrzami!
Do takich czarów służą `hstack` i `vstack`. Powiedzieć, że `h` jest od horizontal (poziomo), a `v` od vertical (pionowo), to jak nie powiedzieć wszystko. Zresztą popatrz
```markdown
>>> a = np.arange(12).reshape(2, 2, 3)
>>> b = np.arange(12).reshape(2, 2, 3)
>>> a
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> b
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])

>>> np.hstack([a, b])
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11],
        [ 6,  7,  8],
        [ 9, 10, 11]]])
>>> np.hstack([a, b]).shape
(2, 4, 3)
>>> np.vstack([a, b])
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]],

       [[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> np.vstack([a, b]).shape
(4, 2, 3)
```

Czy to już wszystko co warto wiedzieć o NumPy?<br />
Na pewno nie! Internet i znakomita dokumentacja NumPy pewnie pomogą Ci o wiele więcej razy niż to krótkie wprowadzenie.<br />
"Szukajcie, a znajdziecie" ~ [Łk, 11, 9]
