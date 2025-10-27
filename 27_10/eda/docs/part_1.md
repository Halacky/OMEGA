We have studied the data structure and these are the conclusions we have reached at this stage:
1) In one session, the gestures follow sequentially 
2) Each gesture is repeated 6 times and only then proceeds to the next one. 
3) Rest should be taken between each repetition.
4) The visualization shows that the marking is not entirely accurate, rest begins and ends a little earlier (it looks like a delay in human reaction).

Further plans:
Take data for one patient and split the dataset into training and test samples in 2 ways:
1) **We take random 4 repetitions** 
2) **We take 4 repetitions sequentially** 
Each of the 4 fragments is divided into consecutive windows with a gesture. We do this for every gesture.
Next, we teach the model to determine which gesture a particular window belongs to.