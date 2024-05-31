Preguntes setmana 1 (10/05):
 1) Com creem el vocabulari? creem dos sets un per els diferents caracters de l'idioma d'entrada i un per els diferents caracters de l'idioma de sortida. Crearem dos diccionaris, un per cada idioma per realitzar el mapeig dels dos vocabularis
 2) Com funciona la codificació HOT-encoding? cada caracter del vocabulari es representa amb un vector binari on serán tot 0 i un 1 a la posició del caràcter al mapeig del vocabulari. Es crea una matriu per la codificació del llenguatge d'entrada i dues matrius per la decodificació del llenguatge d'entrada i per la del llenguatge de sortida. Les matrius son MxN on M es el numero de tokens a codificar/descodificar dels llenguatges d'entrada i N es la longitud màxima del conjunt d'oracions del idioma.
 3) Quina serà la longitud dels vectors codificats? Serà el nombre de caràcters que hi han al vocabulari del llengüatge.
 4) En que es diferencíen el model GRU de LSTM? Principalmet el model GRU es més senzill que el LSTM, el GRU té 2 gates i el LSTM en té 3.
 5) Com funciona LSTM? funciona amb 3 gates: Input gate que representa la quantitat d'informació entrant volem guardar Forget gate: Quantitat d'informació que olvidem de la cel·la anterior i Output gate: Quantitat d'informació que volem traslladar al següent pas. Arriba una seqüència codificada i procesa cada caràcter codificat, utilitza la informació del caràcter actual i la imformació que ha rebut del caràcter anterior de l'estat ocult per calcular el nou estat ocult i la sortida actual. El nou estat ocult s'utiltzarà en el següent pas.
 6) Que es el estat ocult? Es un element que s'utilitza per capturar i retindre informació important durant el processament d'una seqüència de dades.
 7) Com funciona el GRU? Té dues gates. Reset gate: diu la quantitat d'informació a oblidar i la update gate: Fa les funcions de les input i output gates del LSTM
 8) Quantes capes té la xarxa LSTM i GRU? 5 capes, la capa d'entrada al codificador, la capa de codificador GRU, la capa d'entrada al decodificador, la capa de decodificador GRU i la capa densa que genera la sortida final.
 9) Els models son bidireccionals? ELs models que estem utilitzant no son bidireccionals.
 10) Quines métriques utilitzarem per mesurar el rendiment dels models? Bleu (Bilingual Evaluation Understudy), ROGUE (Recall-Oriented Understudy for Gisting Evaluation), METEOR (Metric for Evaluation of Translation with Explicit ORdering) i WER (Word Error Rate).
 11) Perquè l'accuracy no s'utilitza? L'accuracy avalua si la traducció completa es exactament la mateixa que la referència de la traducció el que pot ser molt restrictiu, a vegades poden haver vàries traduccions vàlides.

Preguntes setmana 2 (17/05):
  1) Perquè dona error al entrenar el model? (CUDRNN error dins la màquina azure)
  2) canvi funció entrenament? dona error de codi o perquè falta alguna paquet?
  3) Embedding o one-hot encoding (starting point)
  4) com afecta el tamany de les dades entrades? (El dataset de català es més petit i el podem executar al local, a mesura que augmenten les dades no podem executar al local)
  5) Com millorar el rendiment dels models?
  6) Quin dels models ens dona més rendiment al principi? Amb quin aconseguim millors resultats? (GRU vs LSTM)

Preguntes setmana 3 (31/05):
  1) Realitzar gràfica amb loss i validation per veure si hi ha overfitting.
  2) Modificar el tamany del hidden layer per veure com afecta al rendiment.
  3) Que fan Blue i Rouge? Per què els hem escollit?
  4) Realitzar gràfiques cada epoch (no mini-batches).
  5) GRU o LSTM, quin dels dos ens funciona millor?
  6) Realitzar comparacions entre word2vec i one hot-encoding.







