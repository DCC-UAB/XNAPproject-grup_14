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

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bcaf3713-d17a-45ea-813a-747438132497)

## Arquitectura
Utilitzem un model Seq2Seq basat en RNN. Aquest model consta de dos components principals: l'encodificador (encoder) i el decodificador (decoder). L'encodificador processa la seqüència d'entrada (la frase en l'idioma d'origen) i la converteix en un vector de context, que encapsula la informació essencial de tota la seqüència. Aquest vector es passa al decodificador, que genera la seqüència de sortida (la frase en l'idioma de destí) de manera seqüencial. A cada pas de temps, el decodificador utilitza l'estat ocult anterior i la paraula generada anteriorment per predir la següent paraula. Aquest procés continua fins que es produeix un símbol de final de seqüència.

El nostre decodificador utilitza el mecanisme d'atenció (Attention) que permet al decodificador accedir a tots els estats ocults de l'encodificador. A cada pas del decodificador, es calcula un pes d'atenció per a cada estat ocult de l'encodificador, que determina quanta importància s'ha de donar a cada part de la seqüència d'entrada en predir la següent paraula. La combinació ponderada dels estats ocults d'entrada es converteix en el vector de context dinàmic per al pas de temps actual del decodificador. Els dos tipus de RNN que es solen utilitzar per aquests projectes son LSTM i GRU.

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bf73d475-a1c5-4715-b8b7-b81268195d21)

Per a crear un model seq2seq eficient i robust, hem dut a terme una anàlisi exhaustiva de diversos factors clau que poden afectar el seu rendiment. Cadascun d'aquests factors ha estat estudiat meticulosament, i hem elaborat gràfics per a entendre-ho millor i permetre’ns prendre decisions informades per optimitzar cada aspecte del disseny i entrenament del model. A continuació, descrivim els principals aspectes considerats:



## DATALOADER
En el context d'entrenament de models de NLP, la forma en què les dades són carregades i presentades al model pot tenir un impacte significatiu en el rendiment i la generalització. 

Un DataLoader normal càrrega les dades en l'ordre en què són presents en el dataset. En el nostre cas, el dataset de frases està ordenat segons la longitud de les frases, començant amb frases d'una sola paraula, després de dues paraules, i així successivament. Això pot portar a diversos problemes: 
El model pot aprendre a reconèixer patrons específics de longitud en lloc de generalitzar sobre l'estructura de les frases. Per exemple, si durant diverses iteracions el model només veu frases de longitud similar, pot adaptar-se a aquestes longituds i no aprendre a manejar adequadament frases de diferent longitud.

Si l'entrenament es realitza en un ordre fix, les frases més simples (curtes) poden influir desproporcionadament en les primeres etapes de l'entrenament, mentre que les frases més complexes (llargues) poden no rebre suficient atenció.

En canvi, un Random DataLoader barreja aleatòriament les frases abans de carregar-les, el que fa que el model s'exposi constantment a la variabilitat en la longitud de les frases, la qual cosa pot ajudar a millorar la seva capacitat per a manejar seqüències de diferents longituds durant la inferència. Aquesta simple acció fa que ens trobem millores en el model:

FOTO
FOTO2
## GRU VS LSTM
Dos de les arquitectures seq2seq més conegudes són les basades en GRU (Gated Recurrent Units) i LSTM (Long Short-Term Memory). 

L'arquitectura GRU és una variant de les RNN, que introdueix mecanismes de comportes per a manejar millor la dependència temporal i la memòria a curt termini en les seqüències de dades. Les GRU són més simples que les LSTM perquè utilitzen menys paràmetres, la qual cosa les fa més eficients en termes de càlcul i memòria. A causa d'aquesta simplicitat, les GRU poden entrenar-se més ràpidament i amb menys recursos, la qual cosa les fa adequades per a tasques on la longitud de les seqüències no és excessivament llarga, com és el nostre cas.

L'arquitectura LSTM és una altra variant de les RNN, dissenyada per a superar els problemes d'esvaïment i explosió del gradient que dificulten l'aprenentatge de dependències a llarg termini en seqüències llargues. Les LSTM incorporen una estructura de memòria més complexa, amb comportes d'entrada, sortida i oblit que permeten retenir informació rellevant durant períodes més llargs de temps. Aquesta capacitat de manejar dependències a llarg termini fa que les LSTM siguin especialment efectives en tasques on les seqüències d'entrada són extenses i contenen dependències de llarga durada.

En els nostres experiments, hem observat que les GRU superen a les LSTM en rendiment. Això es deu al fet que les frases en el nostre conjunt de dades no són prou llargues perquè les capacitats de memòria estesa de les LSTM tinguin un impacte significatiu. Les GRU, sent més simples i eficients, dominen les seqüències de longitud curta a mitjana de manera més efectiva en el nostre cas particular.
FOTO


## Hiperparàmetres
Partint d'uns hiperparàmetres base, volem optimitzar el nostre model per a que funcioni millor amb les mètriques que determinaran el rendiment del nostre model. En el nostre cas, seran:

__Bleu (Bilingual Evaluation Understudy) Score__: És una mètrica utilitzada en el processament del llenguatge natural (NLP) i la traducció automàtica per avaluar la qualitat del text generat envers una o varies traduccions de referencia d'alta qualitat.
BLEU funciona comparant n-grames (seqüències de n paraules consecutives) entre el text generat i els textes de referència. 

Calcula la precisió tenint en compte quants n-grams del text generat coincideixen amb els del text o textos de referència. A continuació, la puntuació de precisió es modifica amb una penalització per brevetat per evitar que es afavoreixin les traduccions més curtes.

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/8b670908-75a3-4c95-b21c-c82d1f67236b)

__Train Loss__: El train loss indica la pèrdua durant l'entrenament del model, reflectint l'error sobre les dades d'entrenament


__Valid Loss__: El valid loss indica la pèrdua durant la validació, reflectint l'error sobre les dades de validació no vistes durant l'entrenament

### Dropout
Un altre paràmetre a tenir en compte és el dropout, un métode per reduïr l’overfitting que veurem a continuació els seus efectes:

Amb un dropout de 0.2, la train loss és significativament menor que la valid loss i veiem que mentre la train baixa, la valid comença a augmentar. Això indica que el model se està sobreajustant a les dades d'entrenament, capturant patrons i detalls específics que faran que no generalitzi bé a dades noves de test. Podem interpretar que el model segueix sent massa complex, malgrat que es descativin el 20% de les neurones per cada epoch, per la complexitat de les dades que alimenten el model tot i haver l’overfitting, aquest valor de dropout resulta en la millor avaluació dels tres.

Amb un dropout de 0.7, passa més bé el contrari, la train i valid loss són gairebé paral·leles i pròximes entre si. Això mostra que el model no se està sobreajustant al les dades de train i té una millor capacitat de generalització, ja que l'alta taxa de dropout obliga el model a aprendre representacions més robustes i oblidarse de les més detallades del conjunt de train. No obstant això, aquest model obté el pitjor acompliment en termes de Bleu, ja que malgrat capturar els patrons més importants, no ha entrat suficientment en detall per realitzar bones traduccions.

Amb un dropout equilibrat (0.5) obtenim unes métriques intermitges entre els dos models anteriors. Aquests resultats ens serviran per execucions posteriors, tenint en compte la quantitat de traduccions que alimenten el model i la seva complexitat, és clar.



### Learning Rate
El learning rate és un altre factor a tindre en compte, que determina la rapidesa en la que el model aprendrà els patrons del train set.

Un valor de 0.0001 per al "learning rate" és bastant petit. Això significa que els ajustos als pesos seran molt petits en cada pas de l'entrenament. Quan hi ha poques èpoques o poques dades (com es el nostre cas), un "learning rate" tan petit pot fer que el model aprengui molt lentament, ja que els ajustos són massa petits per a tenir un impacte significatiu en els pesos de la xarxa, com podem obervar és el model que seguiex decrementant la loss però a molt baixa velocitat .

En canvi, un lr més alta (0.01) pot fer que el model aprengui massa ràpid. A la gràfica podem observar que decrementa la loss a les primeres epoch i gairebé el 80% del entrenament es manté gairebé constant, el que significa que tot el cómput invertit no ha valgut de res

Per a aquest model i conjunt de dades, la taxa d'aprenentatge de 0.001 ofereix el millor rendiment, proporcionant una disminució constant i efectiva de la pèrdua de validació.

![lr1](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/3a757405-fd9e-421e-9d82-628857c407f9)
![lr2](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/715c7f87-4e5a-4f81-8cc9-1de4941b516f)


![lr3](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/130971223/b1a200d5-8431-423f-bee0-d72278a76ff1)
## Contributors
Enric Canudas 1631674@uab.cat

Lluc Vicente 1631658@uab.cat

Ramón Álvaro 1635833@uab.cat

Bruno León 1633333@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades
UAB, 2024
