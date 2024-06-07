[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/L30CyvB9)
# XNAP-LANGUAGE TRANSLATION USING SEQ2SEQ LEARNING
Aquest projecte consisteix en un model RNN que utilitza una arquitectura d'aprenentatge Seq2Seq per realitzar traduccions. A aquest document hi consta el procés realitzat a partir del starting pont: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html amb les diferents modificacions aplicades per millorar el rendiment del model inicial. El codi està ja preparat per generar gràfiques de l'execució tant de la loss de l'entrenament com la de la validació i el bleu score del model. Per veure les gràfiques generades s'ha de tindre un usuari a [Weights & Biases](https://wandb.ai/site) i posar la key de l'usuari abans d'executar la resta de codi.

El principal objectiu del projecte és entendre com funciona el model i la seva estructura interna per així poder aconseguir una millor versió que realitzi bones traduccions. Això ve marcat per buscar reduir el overfitting i augmentar el bleu score del model amb l'ajustament dels hiperparàmetres i aplicant tècniques per reduir el overfitting.
## Code structure
Aquest projecte conté tot el necessari per executar el codi. A la carpeta idiomes tenim els dos datasets que hem utilitzat, un que conté l'arxiu amb les traduccions del Dutch (Holandès) a l'anglès i l'altre que conté les traduccions de l'alemany a l'anglès. Els arxius .pth són el codificador i decodificador ja entrenats amb també la configuració dels hiperparàmetres a l'execució (model_config.pth).

Després tenim diferents notebooks (.ipynb) amb els codis a executar per posar el model en funcionament, hi ha diferents arxius que hem anat actualitzant per arribar al notebook definitiu que conté el codi de l'execució final model_pytorch_bleu_rouge.ipynb. Aquest arxiu conté tot el codi on primer consten totes les funcions que realitzen el preprocessament de les dades. Després hi tenim les classes que defineixen el encoder, el decoder i el decoder amb atenció. Posteriorment, hi consta el codi que crea els dataloaders i finalment el que realitza l'entrenament i validació del model on al final es veuen exemples de diferents traduccions i la visualització del heatmap de l'atenció.

## How to use the code?
1 - Descarregar els dos datasets de traduccions que conté la carpeta idiomes.

2 - A l'anar al notebook posar la key d'usuari a la cel·la de wandb.

3 - Anar executant el codi cel·la per cel·la.

## Dataset
Per aquest projecte hem utilitzat el dataset Anki que conté una recopilació de traduccions d'oracions de diverses longituds per diferents idiomes. És un dataset ja especialitzat per l'entrenament i validació de models de traducció automàtica. Per al projecte hem agafat el dataset amb traduccions de l'anglès a l'alemany (270.000 traduccions) i de l'anglès a l'holandès (80.000 traduccions).

L'estructura dels datasets és la mateixa per qualsevol idioma, s'organitza per parelles d'oracions. Per cada fila de l'arxiu .txt conté una parella d'oracions amb el text en l'idioma d'origen (majoritàriament anglès) i la seva traducció en l'idioma que hem escollit. Les oracions estan ordenades per la longitud, comença amb oracions d'una paraula i va augmentant la longitud de les oracions.

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bcaf3713-d17a-45ea-813a-747438132497)

## Arquitectura
Utilitzem un model Seq2Seq basat en RNN. Aquest model consta de dos components principals: l'encodificador (encoder) i el decodificador (decoder). L'encodificador processa la seqüència d'entrada (la frase en l'idioma d'origen) i la converteix en un vector de context, que encapsula la informació essencial de tota la seqüència. Aquest vector es passa al decodificador, que genera la seqüència de sortida (la frase en l'idioma de destí) de manera seqüencial. A cada pas de temps, el decodificador utilitza l'estat ocult anterior i la paraula generada anteriorment per predir la següent paraula. Aquest procés continua fins que es produeix un símbol de final de seqüència.

El nostre decodificador utilitza el mecanisme d'atenció (Attention) que permet al decodificador accedir a tots els estats ocults de l'encodificador. A cada pas del decodificador, es calcula un pes d'atenció per a cada estat ocult de l'encodificador, que determina quanta importància s'ha de donar a cada part de la seqüència d'entrada en predir la següent paraula. La combinació ponderada dels estats ocults d'entrada es converteix en el vector de context dinàmic per al pas de temps actual del decodificador. Els dos tipus de RNN que es solen utilitzar per aquests projectes son LSTM i GRU.

![image](https://github.com/DCC-UAB/XNAPproject-grup_14/assets/91469023/bf73d475-a1c5-4715-b8b7-b81268195d21)


## LSTM



## Contributors
Enric Canudas 1631674@uab.cat

Lluc Vicente 1631658@uab.cat

Ramón Álvaro 1635833@uab.cat

Bruno León 1633333@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades
UAB, 2024
