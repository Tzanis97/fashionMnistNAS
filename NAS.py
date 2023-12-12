import nni
search_space = {
    "filter_size_c1" : {'_type' : 'choice', '_value': [32,64,128]},
    "filter_size_c2" : {'_type' : 'choice', '_value': [32,64,128]},
    
    "kernel_size_c1" : {'_type' : 'choice', '_value': [3,5]},
    "kernel_size_c2" : {'_type' : 'choice', '_value': [3,5]},

    "nb_units" : {'_type' : 'choice', '_value': [80, 100, 120]},

    "learning_rate" : {'_type' : 'uniform', '_value': [0.001,0.01]}
}

# ΔΙΑΜΟΡΦΩΣΗ ΤΟΥ ΠΕΙΡΑΜΑΤΟΣ

from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.trial_command = 'python fashionmnistmodel.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space


experiment.config.tuner.name = 'TPE'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10

#give as experiment.run(one of your local server's listening portal)
experiment.run()





