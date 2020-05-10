# Trust and Social Control: Sources of cooperation, performance, and stability in informal value transfer systems
[Claudius Gräbner](https://claudius-graebner.com), Wolfram Elsner and Alex Lascaus

This repository contains the source code for the model used in the abovementioned paper.
The paper is published in the journal *Computational Economics* and is available 
[here](https://claudius-graebner.com/files/papers/AM-Trust-Social-Control-Hawala-CE2020.pdf) (open access).

## Folder structure of the repository

`bash`: contains bash scripts to run the simulation with particularly relevant parameter combinations (not necessary any more)

`python`: contains all Python scripts and, in `python/parameters/` the json files containing relevant parameter specifications.

`notebooks`: contains Python notebooks, including the one used to create the figures for the final paper

`tex`: contains TeX code used to create some of the figures in the final paper

`output`: contains all the output produced by the source code

## Main elements of the model

The general functioning of the model is explained in the main paper. 
Moreover, the code is well documented and should be largely self-explanatory. 
The general structure and the main elements of the model are as follows:

* The parameter files saved in `python/parameters/`. These are json files specifying all relevant model parameters. If a parameter value is given as a `list` and the model is called via `Main`, then the model is run for all possible parameter constellations for those specified in the lists.
    * For example, if the parameter `C_trust` is specified as `[0, 1]` and `C_control_1` as `[0, 1]`, then calling this parameter file means to runs one model instance with `C_trust=1` and `C_control_1=1`, one with `C_trust=0` and `C_control_1=1`, one with `C_trust=0` and `C_control_1=1`, and one with `C_trust=0` and `C_control_1=0`.
* The main model is implemented via the class `Model`, which is defined in `model.py`. Some parts were sourced out in separate files:
    * During the initialization of a `Model`, an initial agent population must be set up. This is done by the class `PopulationGenerator`, which is defined in `population_generator.py`.
    * The class `Agent` is defined in `agent.py`. 
    * The strategies of the agents are defined via their own class `Strategy`. This class is defined in `strategies.py`.
* The class `Main`, which is defined in `main.py` reads a parameter file, creates many instances of `Model`, runs these models and summarizes their results in a single data frame.
* The function `conduct_meta_mcs()` as defined in `MCS.py` facilitates Monte Carlo simulations. When it is called one has to supply the skeleton of the names of parameter files. Then `main.py` is executed for all parameter files matching this name and the results are aggregated into a single data frame that can then be further processed. 

Thus, the recommended way to use the model is via `MCS.py`. To run the model for all parameter constellations that are specified in the parameter files starting with 'hawala_shocks' we can use the following call:

```
python python/MCS.py python/parameters/hawala_shocks 10
```

This runs the model 10 times for each of the following parameter files: `hawala_shocks_c.json`, `hawala_shocks_t.json` and `hawala_shocks_tc.json`, and aggregates the results into a single data frame.


## Using the model

There are three ways to run the model:

1. Conduct a single model run using `model.py` directly. This only works for a single parameter constellation and is not recommended.

2. Run the same model for many times using `main.py`. This allows to sweep through different parameter constellations if the different parameter values are specified as a `list` in the json file.

3. Run the model for several parameter constellation using `MCS.py`. This is the recommended way to use the model and effectively calls `main.py` several times and can process more than one parameter file. 


## Replicating the figures of the main paper

To replicate the figures in the main paper you must first create the data by runing the following simulations.

For figure 5:

```
python python/MCS.py python/parameters/hawala_baseline 30
```

For figures 6 and 7:

```
python python/MCS.py python/parameters/hawala_shocks 30
```

For figure 8:

```
python python/MCS.py python/parameters/hawala_framework_nbhawaladars 30
```
and
```
python python/MCS.py python/parameters/hawala_framework_intdensity 30
```

For figure 9:

```
python python/MCS.py python/parameters/hawala_framework_forgiveness 30
```

For figure 10:

```
python python/MCS.py python/parameters/hawala_framework_pop_growth 30
```

For figure 11:

```
python python/MCS.py python/parameters/hawala_framework_mistakes 30
```

Then you can use the Python Notebook in `notebooks/Create-Figures.ipynb` to create all the figures. They are automatically saved to `output/figures`.

## Comments

In case you have questions, feedback or other comments feel free to contact Claudius [here](https://claudius-graebner.com/contact-1.html).