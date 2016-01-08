# Attentive Reader
This repo contains a simple variation of the attentive reader model from Reading Comprehension
paper from Moritz et al, NIPS 2015. That code works both on bAbI and reading comprehension tasks.

# Usage
-------
To be able to run an example experiment, given that you have access to the top4 and top8 datasets,
   you can launch the experiments as below:

```
    ./codes/att_reader/launch_exp_ex.sh
```

The "launch_exp_ex.sh" script just calls the "train_attentive_reader.py" script with the necessary
parameters.

    * Under the scripts/ directory you can find simple scripts to preprocess the data.
    * Under the codes/core, you can find simple functions and classes about the core utilities and
functionalities.
    * Under the codes/attentive_reader/ folder you can access to the main codes for the
attentive_reader
    * Experiment scripts will be added under the experiments/ directory.

The main code functionalities are provided under, ./codes/att_reader folder. In that folder, you
can access to the codes of the attentive reader. In the model.py, you can find the functions that
can be used to build the computational graph of the model. layers.py includes building blocks and
layers in order to build the model.

The main script that most of the model is being built and the computational graph is being
constructed is in the attentive_reader.py script.
