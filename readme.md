###Pong AI ###
The code below is created to make a Neural Network to play Pong game.

The file **game.py** contains main loop of the game. Using this file it is possible
to collect the data (*main_collect_data* function), use already built model (*main_ai*)
or just to play the game (*main* function).

There are two files that could be used to generate Neural Networks. **neural_network.py**
would generate NN with three possible moves: up, down and no movement. **nn_no_stops.py** has
only two options: up and down.

Data is collected based on the players actions and stored in a csv file.
Labels: 0 - idle, 1 - up, 2 - down.