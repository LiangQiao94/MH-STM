# MH-STM
-----------------------------------------------------------------------

This is the readme file to provide information about the Python codes used in the case study and numerical study of the paper

   "A Multifacet Hierarchical Sentiment-Topic Model with Application to Multi-Brand Online Review Analysis"

We modified the Python package hlda (see https://github.com/joewandy/hlda) for implementation of the proposed method.

-----------------------------------------------------------------------

The Python source codes and data example for applying the proposed method are shown as follows:

 - The folder “simulation” contains the source codes as well as relying files for simulation study of the method:
      
      - source codes:

            - generate_demo_corpus.py: generate demo data for qualitative analysis in Section 4.1
            - generate_synthetic_corpus.py: generate simulated data for quantitative analysis in Section 4.2
            - MHSTM_model.py: the proposed model class
            - simulation.py: the main file for excecuting simulation procedure

      - relying files:
            
            - data: including the demo data and simulated data used in simulation
            - models: saving intermediate model results for model demonstration and evaluation

 - The folder “case study” contains the source codes as well as relying files for case study of the method:

      - source codes:

            - MHSTM_model.py: the proposed model class and model estimation on experimental data
            - aspect_brand_ranking: the main file for excecuting multi-aspect brand ranking 

      - relying files:
            
            - data: including the experimental data used in case study
            - models: saving intermediate model results for model demonstration and evaluation

-----------------------------------------------------------------------
