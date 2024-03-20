# Two-Stage Machine Learning-Based Approach for Predicting Points of Departure

This repository contains the source code and results associated with the manuscript titled "Two-Stage Machine Learning-Based Approach to Predict Points of Departure for Human Non-cancer and Developmental/Reproductive Effects," by Kvasnicka et al.

## Compatibility
The procedure has been tested on PC computers running Windows 10 and Windows 11.

## Getting Started

### Downloading the Repository and Assets
1. The latest version of the repository can be found under the "Releases" tab. Download the source code and the corresponding assets.
2. The assets include an `Input` directory and a `Results` directory, both containing large files.
3. Unzip the source code directory to a preferred location on your computer.
4. Unzip the asset directories into the source code directory. Your directory now contains all models, results, and figures corresponding to the manuscript.

### Setting Up the Environment
1. **Anaconda/Miniconda Installation**: If you don't have Anaconda or Miniconda installed, please download and install from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **Create a New Virtual Environment**: Open an Anaconda terminal and navigate to the source code directory.
    ```sh
    cd path/to/source-code-directory
    ```
   Then, create a new environment using the `environment.yml` file included in the directory:
    ```sh
    conda env create -f environment.yml -n your-environment-name
    ```
3. **Activate the Environment**:
    ```sh
    conda activate your-environment-name
    ```

### Usage

#### Analyzing Modeling Results
- The `results_analysis` module provides a `ResultsAnalyzer` class, which is central to analyzing modeling results.
- Use of `ResultsAnalyzer` is demonstrated in the Jupyter notebook `Analysis_for_ES&T_Manuscript`, also available as an HTML file in the `Analyses` directory.

#### Reproducing Manuscript Results and Figures
1. **Preprocess Raw Input Files**: Generate the `Processed` sub-directory within `Inputs` containing features and target variables.
    ```sh
    python preprocess.py
    ```
   This step may take several minutes. "Preprocessing completed" will be displayed in the console when this step is finished.

2. **Execute Modeling Workflows**: This step is computationally intensive and may take around 24 hours on a standard desktop to process all models according to `Input/Configuration/model-configuration.json`.
    ```sh
    python workflow_management.py
    ```
   This creates a `Results` directory with machine learning estimators, performance scores, and feature importance scores. "Run completed" will be displayed in the console when this step is finished.

3. **Plot Results**: Generate figures based on the modeling results.
    ```sh
    python plot.py
    ```
   This creates a new directory `Figures` with image files corresponding to the results. "Plotting completed" will be displayed in the console when this step is finished.

### Caution
Executing all modeling workflows is computationally intensive and utilizes parallel processing. Execution time may vary based on computer specifications.
