# Stochastic differential equations


## Forward diffutsion

## Bacward diffusion


si parte dalla SDE, ovvero semplicemente una ODE perturbata. Tipicamente avrà un centro attrattivo, un punto (per semplicità), sia nel caso della ODE che nel caso della SDE (è da intendersi in senso stocastico -> come distribuzione di probabilità finale). Vogliamo avere la possibilità di simulare una SDE, eventualmente con un campo come input, così come anche la matrice di diffusione, con anche un parametro di rumore epsilon. Noi vogliamo creare un solver per queste SDEs. 
Il processo inizia dal tempo 0 e continua per un tempo T, scelto abbastanza grande di modo che, evolvendo, la distribuzione dei punti iniziale abbia raggiunto un equilibrio. La distribuzione iniziale è scelta a piacere ma tipicamente, per la Manifold hypothesis, sarà una sotto-varietà dello spazio iniziale. La distribuzione arbitraria iniziale deve però comunque contenere il punto attrattivo ma, una volta avuta questa distribuzione, si procede a fare un sampling di n punti che si faranno evolvere con la SDE. Ad ogni istante di tempo dt si calcola la distribuzione che si è ottenuta (probabilitò inidcizzata dal tempo). In seguito, dopo aver calcolato questa distribuzione di probabilità, per ottenere la distribuzione inversa si fa il gradiente di questa cosa.

Se puoi, crea un file di configurazione con una distribuzione uniforme di 2000 punti in un disco, facciamo evolvere il sistema con la SDE utilizzando il solver e salvandoci, per ogni dt, i punti evoluti (quindi la configurazione). Di seguito si dovrebbe costruire la distribuzione di probabilità associata al sistema discreto di punti (ma ci pensiamo dopo).

Per l'equazione backward: noi abbiamo, per ogni istante di tempo, una configurazione di punti che, se è sufficientemente fitta, ci permette di stimare la p al tempo t, per ogni t. Avendo ciò, possiamo stimare anche il gradiente del logaritmo di p al tempo t. Questo è lo score, e una volta noto possiamo definire al SDE backward (equazione stocastica che va indietro nel tempo, ricreando i dati togliendo rumore). Facendo un sampling random di punti (stesso numero) dalla distribuzione finale, si evolvono con l'equazione backward che a questo punto è nota (perchè per ogni tempo abbiamo il gradiente del logaritmo di p(t), ovvero lo score). Ciò ci permette, integrando da T a 0, di ricreare, da questi sample, i nuovi dati.



## Quasipotential computation
quasipotenziale sempre tenendo in conisderazione la def originaria (in caso particolare, se esiste la decomposizione orfogonale, è due volte il gradiente)




### To do
- esiste un metodo certo per quantificare C(x), da p_T(x) = C(x)e^(-(1/epsilon))U(x)?
- simulazione evoluzione forward verso un ventro attrattivo di punti distribuiti uniformemente su una curva in R^2 (rispettante Mainfold hypothesis)
- disco con distribuzione di punti uniforme