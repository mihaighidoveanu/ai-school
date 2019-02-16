# Similaritatea între doua texte

## Motivație

Deseori în problemele care implică textul, avem nevoie de o metodă de a cuantifica similaritatea a două documente. 
De exemplu, putem calcula similaritatea între două texte pentru a da un semnal de alarmă vizavi de plagierea unei lucrări. 

Algoritmul folosit aici calculează similaritatea între două texte bazată pe numărul de apariții ale fiecărui cuvânt în text. Este o metodă oarecum rudimentară, pentru că nu ține cont de contextul și înțelesul cuvintelor, ci doar de ce cuvinte și cât de des au fost folosite.

Din această cauză, algoritmul ar fi foarte puțin util la proiecte precum recomandarea unor cărți în stilul cărților citite de către un utilizator. 
Dar ar putea fi foarte util la detectarea articolelor de știri care vorbesc despre același subiect, pentru a elimina redundanța. 

În continuare, vom prezenta pe scurt algoritmul folosit și apoi o documentație a funcțiilor implementate în Python.

## Algoritmul folosit

Algoritmul calculează o măsură de similaritate între două texte, care mai departe vor fi referite ca TA și TB. 

1. Eliminăm semnele de punctuație și numerele din text și transformăm fiecare text într-o listă de cuvinte.

**BONUS** : Putem îmbunătăți rezultatele prin a face *stemming* sau *lemmatization* pe cuvintele textului. Pe corpusul SICK am obținut rezultate mai bune cu *stemming*.

2. Pentru că ne interesează cuvintele cât mai relevante pentru textul dat și nu cele generale, vom elimina *stopwords* din fiecare listă.
3. Toate cuvintele se aduc în formă *lowercase* pentru a trata la fel același cuvânt, indiferent dacă apare la începutul propoziției sau în interiorul ei. 
4. Calculăm frecvența cuvintelor în fiecare text și obținem câte un vector de frecvență pentru fiecare text : VA și VB.
5. Pentru a calcula distanța între cei doi vectori, ei trebuie să aibă aceeași dimensiune. Astfel, adăugăm la vectorul VA intrări cu valoarea 0 pentru toate cuvintele care nu apar în TA dar apar în TB. Procedăm la fel și pentru vectorul VB.
Avem grijă ca poziția i din fiecare vector să se refere la același cuvânt.
6. Dați cei doi vectori VA și VB, se calculează distanța între ei astfel. Înmulțirea din formulă se referă la produsul scalar dintre cei doi vectori. 
```python
 distanta = arccos((VA * VB) / sqrt((VA * VA) * (VB * VB)))
```

## Rezultatul obținut

Distanța astfel calculată este o măsură pentru cât de diferite sunt cele două texte, bazată pe frecvența apariției cuvintelor în fiecare text. 
Astfel, două texte sunt cu atât mai similare cu cât distanța între ele este mai mică. 

## Rularea programului 

Programul se rulează din linia de comandă și primește doi parametri, și anume două fișiere cu text pentru care vrem să calculăm similaritatea. 
Exemplu :
```bash
python similarity.py input1.txt input2.txt
```

## Documentație

1. remove_chars(s, removals)
	- șterge anumite caractere dintr-un text
	- folosită pentru a elimina punctuația dintr-un text
	@param s : șir de caractere din care eliminăm caracterele 
	@param removals : șir de caractere pe care le vom elimina

2. tokenize(text)
	@param text : un sir de caractere
	- transformă textul într-o listă de cuvinte, din care elimină *stopwords*, numerele și semnele de punctuație
	

3. count_vectorize(words, vocabulary)
	@param words : listă de cuvinte
	@param vocabulary : o listă de cuvinte care pot apărea sau nu în lista de la primul parametru
	- întoarce un vector de frecvență pentru o listă de cuvinte, cu pozițiile în ordinea cuvintelor din vocabular

4. compute_distance(v1, v2)
	@params v1,v2 : vectori de frecvență ai cuvintelor unor texte
	- calculează distanța între cei doi vectori după următoarea formulă 
	```D = arccos((V1*V2)/sqrt((V1*V1)*(V2*V2)))```

5. compute_similarity(text1, text2)
	@params text1, text2 : șiruri de caractere reprezentând două texte
	- calculează similaritatea între două texte conform algoritmului de mai sus și folosind funcțiile de mai sus


## Îmbunătățiri viitoare

La pasul de preformatare al textului, putem să îmbunătățim prin înlocuirea cuvintelor din text cu hiperonimele sale. Astfel, sinonimele din cele două texte ar putea ajunge să fie reprezentate de același cuvânt, deci să nu mai adauge nimic la distanță. Asta este de dorit, dat fiind că două texte cu multe sinonime sunt similare. 










