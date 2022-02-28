# graph-genetics
<div id="top"></div>

## Description

This is the code repository for the paper entitled "Graph classification for phenotype prediction in neurodegenerative diseases". The repository follows the methodology and results presented in the abovementioned work. 

![Image](figure1.png)

* [genes_of_interest](genes_of_interest) obtain Gene-Disease Associations from DisGeNET using an R script.
* [networks](networks) contains several Python scripts for building different networks (PPIs from different sources, random networks).
* [data_preprocessing](data_preprocessing) several scripts for obtaining genetic data from the different cohorts employed.
* [create_datasets](create_datasets) Python scripts for building different datasets for supervised classification models.
  * [create_nx_datasets.py] is the one for building graph-datasets for Graph Neural Networks (GNNs)
* [ml_models](ml_models) Machine learning models for comparing with GNNs.
* [data](data) contains several data files used in this work. Please note genetic data coming from the cohorts employed is not available due to privacy reasons.
* [results](results) contains several files with the results presented in this work

### Implementation

The code in this repo was built using:

* [Next.js](https://nextjs.org/)
* [React.js](https://reactjs.org/)
* [Vue.js](https://vuejs.org/)
* [Angular](https://angular.io/)
* [Svelte](https://svelte.dev/)
* [Laravel](https://laravel.com)
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact
Please refer any questions to:
Laura Hernández Lorenzo [laurahdezlorenzo](https://github.com/laurahdezlorenzo) - [laurahl@ucm.es](laurahl@ucm.es)

<p align="right">(<a href="#top">back to top</a>)</p>
