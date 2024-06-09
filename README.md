[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/L30CyvB9)
# XNAP-LANGUAGE TRANSLATION USING SEQ2SEQ LEARNING
Aquest projecte consisteix en un model RNN que utilitza una arquitectura d'aprenentatge Seq2Seq per realitzar traduccions. A aquest document hi consta el procés realitzat a partir del starting pont: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html amb les diferents modificacions aplicades per millorar el rendiment del model inicial. El codi està ja preparat per generar gràfiques de l'execució tant de la loss de l'entrenament com la de la validació i el bleu score del model. Per veure les gràfiques generades s'ha de tindre un usuari a [Weights & Biases](https://wandb.ai/site) i posar la key de l'usuari abans d'executar la resta de codi.

El principal objectiu del projecte és entendre com funciona el model i la seva estructura interna per així poder aconseguir una millor versió que realitzi bones traduccions. Això ve marcat per buscar reduir el overfitting i augmentar el bleu score del model amb l'ajustament dels hiperparàmetres i aplicant tècniques per reduir el overfitting.
## Code structure
Aquest projecte conté tot el necessari per executar el codi. A la carpeta idiomes tenim els dos datasets que hem utilitzat, un que conté l'arxiu amb les traduccions del Dutch (Holandès) a l'anglès i l'altre que conté les traduccions de l'alemany a l'anglès. Els arxius .pth són el codificador i decodificador ja entrenats amb també la configuració dels hiperparàmetres a l'execució (model_config.pth).

Després tenim diferents notebooks (.ipynb) amb els codis a executar per posar el model en funcionament, hi ha diferents arxius que hem anat actualitzant per arribar al notebook definitiu que conté el codi de l'execució final model_pytorch_bleu_rouge.ipynb. Aquest arxiu conté tot el codi on primer consten totes les funcions que realitzen el preprocessament de les dades. Després hi tenim les classes que defineixen el encoder, el decoder i el decoder amb atenció. Posteriorment, hi consta el codi que crea els dataloaders i finalment el que realitza l'entrenament i validació del model on al final es veuen exemples de diferents traduccions i la visualització del heatmap de l'atenció.

## How to use the code?
__1__ - Descarregar els dos datasets de traduccions que conté la carpeta idiomes.

__2__ - A l'anar al notebook posar la key d'usuari a la cel·la de wandb.

__3__ - Anar executant el codi cel·la per cel·la.

## Dataset
Per aquest projecte hem utilitzat el dataset Anki que conté una recopilació de traduccions d'oracions de diverses longituds per diferents idiomes. És un dataset ja especialitzat per l'entrenament i validació de models de traducció automàtica. Per al projecte hem agafat el dataset amb traduccions de l'anglès a l'alemany (270.000 traduccions) i de l'anglès a l'holandès (80.000 traduccions).

L'estructura dels datasets és la mateixa per qualsevol idioma, s'organitza per parelles d'oracions. Per cada fila de l'arxiu .txt conté una parella d'oracions amb el text en l'idioma d'origen (majoritàriament anglès) i la seva traducció en l'idioma que hem escollit. Les oracions estan ordenades per la longitud, comença amb oracions d'una paraula i va augmentant la longitud de les oracions.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bcaf3713-d17a-45ea-813a-747438132497" width="700" height="300">

## Arquitectura
Utilitzem un model Seq2Seq basat en RNN. Aquest model consta de dos components principals: l'encodificador (encoder) i el decodificador (decoder). L'encodificador processa la seqüència d'entrada (la frase en l'idioma d'origen) i la converteix en un vector de context, que encapsula la informació essencial de tota la seqüència. Aquest vector es passa al decodificador, que genera la seqüència de sortida (la frase en l'idioma de destí) de manera seqüencial. A cada pas de temps, el decodificador utilitza l'estat ocult anterior i la paraula generada anteriorment per predir la següent paraula. Aquest procés continua fins que es produeix un símbol de final de seqüència.

El nostre decodificador utilitza el mecanisme d'atenció (Attention) que permet al decodificador accedir a tots els estats ocults de l'encodificador. A cada pas del decodificador, es calcula un pes d'atenció per a cada estat ocult de l'encodificador, que determina quanta importància s'ha de donar a cada part de la seqüència d'entrada en predir la següent paraula. La combinació ponderada dels estats ocults d'entrada es converteix en el vector de context dinàmic per al pas de temps actual del decodificador. Els dos tipus de RNN que es solen utilitzar per aquests projectes son LSTM i GRU.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bf73d475-a1c5-4715-b8b7-b81268195d21" width="350" height="225">

Per a crear un model seq2seq eficient i robust, hem dut a terme una anàlisi exhaustiva de diversos factors clau que poden afectar el seu rendiment. Cadascun d'aquests factors ha estat estudiat meticulosament, i hem elaborat gràfics per a entendre-ho millor i permetre’ns prendre decisions informades per optimitzar cada aspecte del disseny i entrenament del model. A continuació, descrivim els principals aspectes considerats:



## DATALOADER
En el context d'entrenament de models de NLP, la forma en què les dades són carregades i presentades al model pot tenir un impacte significatiu en el rendiment i la generalització. 

Un DataLoader normal càrrega les dades en l'ordre en què són presents en el dataset. En el nostre cas, el dataset de frases està ordenat segons la longitud de les frases, començant amb frases d'una sola paraula, després de dues paraules, i així successivament. Això pot portar a diversos problemes: 
El model pot aprendre a reconèixer patrons específics de longitud en lloc de generalitzar sobre l'estructura de les frases. Per exemple, si durant diverses iteracions el model només veu frases de longitud similar, pot adaptar-se a aquestes longituds i no aprendre a manejar adequadament frases de diferent longitud.

Si l'entrenament es realitza en un ordre fix, les frases més simples (curtes) poden influir desproporcionadament en les primeres etapes de l'entrenament, mentre que les frases més complexes (llargues) poden no rebre suficient atenció.

En canvi, un Random DataLoader barreja aleatòriament les frases abans de carregar-les, el que fa que el model s'exposi constantment a la variabilitat en la longitud de les frases, la qual cosa pot ajudar a millorar la seva capacitat per a manejar seqüències de diferents longituds durant la inferència. Aquesta simple acció fa que ens trobem millores en el model:

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/3c933ce4-bb6f-4273-a863-6e3f6fe8a3ea" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/6691d191-54fb-4cd0-8848-ffbdecac51f6" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/bbbce72f-dc68-4718-ab5a-8c419a506b5d" width="350" height="225">



## GRU VS LSTM
Dos de les arquitectures seq2seq més conegudes són les basades en GRU (Gated Recurrent Units) i LSTM (Long Short-Term Memory). 

L'arquitectura GRU és una variant de les RNN, que introdueix mecanismes de comportes per a manejar millor la dependència temporal i la memòria a curt termini en les seqüències de dades. Les GRU són més simples que les LSTM perquè utilitzen menys paràmetres, la qual cosa les fa més eficients en termes de càlcul i memòria. A causa d'aquesta simplicitat, les GRU poden entrenar-se més ràpidament i amb menys recursos, la qual cosa les fa adequades per a tasques on la longitud de les seqüències no és excessivament llarga, com és el nostre cas.

L'arquitectura LSTM és una altra variant de les RNN, dissenyada per a superar els problemes d'esvaïment i explosió del gradient que dificulten l'aprenentatge de dependències a llarg termini en seqüències llargues. Les LSTM incorporen una estructura de memòria més complexa, amb comportes d'entrada, sortida i oblit que permeten retenir informació rellevant durant períodes més llargs de temps. Aquesta capacitat de manejar dependències a llarg termini fa que les LSTM siguin especialment efectives en tasques on les seqüències d'entrada són extenses i contenen dependències de llarga durada.

En els nostres experiments, hem observat que les GRU superen a les LSTM en rendiment. Això es deu al fet que les frases en el nostre conjunt de dades no són prou llargues perquè les capacitats de memòria estesa de les LSTM tinguin un impacte significatiu. Les GRU, sent més simples i eficients, dominen les seqüències de longitud curta a mitjana de manera més efectiva en el nostre cas particular.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/e9c59d82-5ae4-4320-8763-2b50d6371038" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/5cf2fb86-7100-44d3-b3f0-1e4e5c9eb914" width="350" height="225">



## Hiperparàmetres
Partint d'uns hiperparàmetres base, volem optimitzar el nostre model per a que funcioni millor amb les mètriques que determinaran el rendiment del nostre model. En el nostre cas, seran:

__Bleu (Bilingual Evaluation Understudy) Score__: És una mètrica utilitzada en el processament del llenguatge natural (NLP) i la traducció automàtica per avaluar la qualitat del text generat envers una o varies traduccions de referencia d'alta qualitat.
BLEU funciona comparant n-grames (seqüències de n paraules consecutives) entre el text generat i els textes de referència. 

Calcula la precisió tenint en compte quants n-grams del text generat coincideixen amb els del text o textos de referència. A continuació, la puntuació de precisió es modifica amb una penalització per brevetat per evitar que es afavoreixin les traduccions més curtes.

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/8b670908-75a3-4c95-b21c-c82d1f67236b)

__Train Loss__: El train loss indica la pèrdua durant l'entrenament del model, reflectint l'error sobre les dades d'entrenament


__Valid Loss__: El valid loss indica la pèrdua durant la validació, reflectint l'error sobre les dades de validació no vistes durant l'entrenament

El cas base que tenim és el següent:

__Learning Rate__ = 0.01

__Dropout__ = 0.1

__Cell type__ = GRU

__Epochs__ = 25

__Optimizer__ = Adam

__Hidden Size__ = 128

__Batch Size__ = 32

### Tamany de les dades 
El primer hiperparàmetre que hem començat a analitzar és el tamany de les dades, en el nostre codi es modifica a través de la variable max_length. Comencem fent diferents proves des de 40.000 parells d'oracions, on el valid loss és gairebé 2, fins a proves amb 130.000 parells d'oracions on arribem a reduir el valid loss per sota d'1. 

Per tant, podem dir que aquest hiperparàmetre és el que més redueix l'overfitting en el nostre model, però, per altra banda, també augmenta molt el temps d'entrenament. En conseqüència, busquem un punt intermig on el nostre model no pateixi overfitting i el temps d'entrenament no sigui molt elevat. 

Finalment, hem decidit que el millor valor pel max_length és 10 on agafem 130.000 parells d'oracions, ja que encara que el temps d'entrenament és una mica elevat, la reducció de l'overfitting que ens proporciona aquest tamany és molt important. També cal dir que si tinguéssim més temps i poguéssim entrenar el model amb el dataset complet (270.000 parells d'oracions) podríem reduir encara molt més el valid loss.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/65753d0d-0453-4c6b-9a86-d552b671d3f8" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/bb202466-447d-41c6-8260-802584260f65" width="350" height="225">

### Batch_size
La mida del batch_size indica quantes mostres d'entrenament s'utilitzen en cada iteració del procés d'entrenament. Aquest hiperparàmetre es pot ajustar i determina el nombre de mostres processades simultàniament abans d'actualitzar els pesos del model. Una mida de batch_size més gran pot accelerar l'entrenament, mentre que una mida més petita pot proporcionar estimacions de gradient més precises. Per això vam decidir experimentar amb diferents mides per trobar la més adequada.

Fem 4 proves amb batch_size diferents, on provem amb 32, 64, 128 i 256. I com podem veure a les gràfiques de valid loss i train loss la millor mida de batch és 64, ja que és on obtenim un loss més petit. També comparem els diferents batch_size pel que fa al bleu i també obtenim que 64 és el millor resultat. Per tant, agafem un batch_size de 64 com a resultat final, ja que tant com en overfitting com en precisió de traducció és el millor resultat obtingut. 

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/df7a0472-37ee-4006-9fa9-f1a5770e4ddb" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/390228f1-1ffe-4e6d-9a4f-a356029f7331" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/639f2f7b-0d8c-4633-b2c8-337d33901470" width="350" height="225">


### Dropout
Un altre paràmetre a tenir en compte és el dropout, un métode per reduïr l’overfitting que veurem a continuació els seus efectes:

Amb un dropout de 0.2, la train loss és significativament menor que la valid loss i veiem que mentre la train baixa, la valid comença a augmentar. Això indica que el model se està sobreajustant a les dades d'entrenament, capturant patrons i detalls específics que faran que no generalitzi bé a dades noves de test. Podem interpretar que el model segueix sent massa complex, malgrat que es descativin el 20% de les neurones per cada epoch, per la complexitat de les dades que alimenten el model tot i haver l’overfitting, aquest valor de dropout resulta en la millor avaluació dels tres.

Amb un dropout de 0.7, passa més bé el contrari, la train i valid loss són gairebé paral·leles i pròximes entre si. Això mostra que el model no se està sobreajustant al les dades de train i té una millor capacitat de generalització, ja que l'alta taxa de dropout obliga el model a aprendre representacions més robustes i oblidarse de les més detallades del conjunt de train. No obstant això, aquest model obté el pitjor acompliment en termes de Bleu, ja que malgrat capturar els patrons més importants, no ha entrat suficientment en detall per realitzar bones traduccions.

Amb un dropout equilibrat (0.5) obtenim unes métriques intermitges entre els dos models anteriors. Aquests resultats ens serviran per execucions posteriors, tenint en compte la quantitat de traduccions que alimenten el model i la seva complexitat, és clar.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/493c1be9-47c6-4328-802f-810b46561f5f" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/0cfc1a59-65c7-4a6d-835f-1e96fadcf886" width="350" height="225">


### Learning Rate
El learning rate és un altre factor a tindre en compte, que determina la rapidesa en la que el model aprendrà els patrons del train set.

Un valor de 0.0001 per al "learning rate" és bastant petit. Això significa que els ajustos als pesos seran molt petits en cada pas de l'entrenament. Quan hi ha poques èpoques o poques dades (com es el nostre cas), un "learning rate" tan petit pot fer que el model aprengui molt lentament, ja que els ajustos són massa petits per a tenir un impacte significatiu en els pesos de la xarxa, com podem obervar és el model que seguiex decrementant la loss però a molt baixa velocitat .

En canvi, un lr més alta (0.01) pot fer que el model aprengui massa ràpid. A la gràfica podem observar que decrementa la loss a les primeres epoch i gairebé el 80% del entrenament es manté gairebé constant, el que significa que tot el cómput invertit no ha valgut de res

Per a aquest model i conjunt de dades, la taxa d'aprenentatge de 0.001 ofereix el millor rendiment, proporcionant una disminució constant i efectiva de la pèrdua de validació.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/747123c1-d5d2-4f76-aa3f-145840513a22" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/b0553ee1-2199-4b83-bb5b-a338269cfede" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/0310110a-7453-42f2-9207-3a90d8eb3054" width="350" height="225">


### Hidden_size
Hidden_size és un paràmetre que defineix el nombre d'unitats en les capes ocultes de la xarxa neuronal. En el codificador del model , determina la dimensió del vector d'estat ocult que representa la informació acumulada mentre es processa la seqüència d'entrada. I en el descodificador, defineix la mida del vector d'estat ocult que s'utilitza per generar la seqüència de sortida paraula per paraula, basant-se en el vector de context i l'estat ocult anterior.

Un major hidden_size permet a la xarxa capturar més informació i patrons complexos en les seqüències d'entrada, però també incrementa el nombre de paràmetres del model, cosa que pot augmentar el temps d'entrenament i la demanda de memòria. En canvi, un hidden_size massa petit pot resultar en una capacitat insuficient per modelar adequadament les dependències en les dades, mentre que una mida massa gran pot portar a sobreajustament.

Seguidament, fem proves amb 3 tamanys diferents de hidden_size (64, 128 i 256) i com podem veure a la gràfica de train i valid loss quan obtenim menys overfitting és amb una mida de 64, però això és perquè el tamany de les dades és molt petit. Per tant, com a conclusió podem dir que per tamanys de dades petits és millor utilitzar un hidden_size de 64, però per un tamany de dades grans és millor utilitzar el de 256. En conseqüència, per la nostra execució final utilitzarem un hidden_size de 256 perquè el tamany de dades que utilitzarem en aquesta execució serà molt gran. 

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/e3a9a73d-a129-4d84-8e20-2802e2f7d809" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91468482/2ec921eb-fe21-49ee-9bcf-4c92bd3d7db7" width="350" height="225">

### Regularització 
Utilitzem la regularització L2 amb un weight decay de 1e-5 per al nostre projecte de traducció d'idiomes amb seq2seq RNN per diverses raons. Primer, ens ha  ajudat a prevenir l'overfitting en reduir la complexitat del model. També gràcies a que millora la generalització a noves dades no vistes i estabilitza el procés d'entrenament en evitar actualitzacions de pesos extremadament grans.
S'ha comprovat que pot conduir a una millor convergència durant l'entrenament i per tant ens ha ajudat per mantenir una bona relació entre bias i variança.

## Execucions finals

Finalment vam fer una execució llarga amb els hiperparàmetres que millor funcionen per a tamanys de dades més grans. Al augmentar las epochs a 80.
Els nostres paràmetres finals que utilitzem per a l'última execució són els següents:

__Cell_type__ = GRU

__130K sentences pairs__

__LR__ = 0,001 

__Random dataloader__

__Optimizer__ = Adam 

__Dropout__ = 0,3

__Hidden_size__ = 256   

__Batch_size__ = 64 

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/15da6c9f-a367-4d64-8354-775b9b45a888" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/76a9a95f-6c8f-401f-9a3f-e0676892d681" width="350" height="225">


Podem observar que el Bleu arriba fins a uns valors de 0.45 (per a una traducció utilitzant la mètrica Bleu estem parlant d'una traducció molt bona).
També es pot observar que comparat amb les anteriors execucions,hem aconseguit reduir notablement l'overfitting que presentava.

### Exemples de predicció
Com es pot veure, les traduccions que fa encara no ser perfectes per a traduccions més complexes (ja sigui per estructura de la oració i llargària d'aquesta) si que podem dir que arriba a una traducció prou acurada.
Per a traduccions més curtes té un grau de similitut exacte i per a traduccions més complexes fa una traducció prou bona, encara que no perfecte.


També hem elaborat unes matrius d'atenció que ens mostren com es relacionen les paraules d'un idioma amb les de l'altre durant una traducció.En aquest cas, parlem d'una traducció de l'alemany a l'anglès. 

En aquesta matriu, les files representen les paraules en anglès (el text de destinació) i les columnes representen les paraules en alemany (el text d'origen). Cada cel·la de la matriu conté un valor que indica la importància de cada paraula alemanya quan es genera una paraula anglesa. L'hem utilitzada ja que pensem que la matriu d'atenció, representada com un mapa de calor (heatmap), proporciona una manera intuïtiva de veure quines parts de la frase d'origen influeixen més en la generació de cada paraula de la frase de destinació.

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/c7046100-5e76-47fc-b9db-7ec2ac367031" width="300" height="200">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/749d1c61-610a-421e-9113-ba6954ab6854" width="300" height="200">


<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/14a2731c-434f-4294-96ea-018219ec6719" width="400" height="250">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/6287c86e-8fdd-4d8a-806b-f047386d9b76" width="400" height="250">




### Comparació del model inicial vs model final
Com podem observar en les gràfiques que utilitzem per avaluar el rendiment i qualitat del nostre model, el model inicial presentava bastant overfitting i un bleu que arribava a 0.3. 

Amb el nostre model final conseguim reduir molt l'overfitting que presentava i augmentar el valor del bleu fins a 0.48(fent així que les traduccion siguin molt més precises).

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/4792e45c-7e06-4302-aae4-4ab27a7d79d1" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/8941839c-7eaf-431f-8534-59bdf50949dd" width="350" height="225">

### Comparativa amb diferents idiomes
Per a fer aquesta prova hem utilitzat tots 2 models entrenats amb 70k frases, el nou idioma que hem provat ha sigut el holandès (dutch) per tant farem la traducció de (holandès a anglès).

<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/7057c371-58a7-43c2-b5d8-95a076fd7432" width="350" height="225">
<img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/51f018f1-e0d3-4690-8d6d-6291d6b33724" width="350" height="225">
<p align="center">
  <img src="https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/c956a718-a861-4e7b-b527-936fe0cdebb1" width="300" height="150">
</p>




## Conclusions
Com a conclusions del projecte, pensem que ha sigut un projecte profitós per a tots en el que no només hem aprofundit en el funcionament i optimització del model amb estructura seq2seq RNN sinó que també hem après noves eines i formes de treballar. Ja sigui amb les màquines Azure que ens ha donat la capacitat de realitzar execucions més potents (encara que estiguessim limitats per les hores) i la interfaç de Weights & Biases (que encara que al principi s'ha d'aprendre a com gestionar) ens ha permet realitzar gràfiques per epoch d'una manera més eficient i automàtica (una vegada ja configurada).

Pensem que hem aconseguit millorar notablement el nostre starting_point (encara que les mètriques avaluadores no siguin excel·lents) i aprofundir més en com funcionen la traducció de llenguatges emprant models del tipus Seq2Seq RNN.

Tot i així, també ens hem trobat amb varies dificultats que no ens van permetre obtenir encara millors resultats.
Un d'ells és que no vam poder utilitzar els Starting Points fet amb Keras per problemes d'incompatibilitat i vam haver de canviar-ho per models de Pytorch (alentint-nos en el nostre procés de treball).
Altre punt,és que conforme més temps tingui el model per poder entrenar-se millors resultats podrà donar, i tenint en compte les restriccions de complexitat i temps que teniem amb les màquines Azure (30h) no vam poder arribar a aconseguir uns nivells excel·lents en les nostres mètriques.




## Contributors
Enric Canudas 1631674@uab.cat

Lluc Vicente 1631658@uab.cat

Ramón Álvaro 1635833@uab.cat

Bruno León 1633333@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades
UAB, 2024
