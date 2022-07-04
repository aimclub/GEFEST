.. image:: /docs/img/gefest_logo.png
   :alt: Logo of GEFEST framework

.. start-badges
.. list-table::
   :stub-columns: 1

   * - docs
     - |docs|
   * - license
     - | |license|
   * - support
     - | |tg|

.. end-badges

**GEFEST** (**G**\enerative **E**\volution **F**\or **E**\ncoded **ST**\ructures) is a toolbox for the generative design of
physical objects.

It uses: (1) numerical modelling to simulate the interaction between object and environment;
(2) evolutionary optimization to produce new variants of geometrically-encoded structures.

The basic abstractions in GEFEST are Point, Polygon, Structure and Domain.

The workflow of the generative design is the following:

.. image:: /docs/img/workflow.png
   :alt: workflow


Acknowledgments
==============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences
and workshops for their valuable advice and suggestions.

Contacts
========
- `Telegram channel for solving problems and answering questions on GEFEST <https://t.me/gefest_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Newsfeed <https://t.me/NSS_group>`_
- `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Citation
========

@inproceedings{nikitin2021generative, title={Generative design of microfluidic channel geometry using evolutionary
approach}, author={Nikitin, Nikolay O and Hvatov, Alexander and Polonskaia, Iana S and Kalyuzhnaya, Anna V and Grigorev,
Georgii V and Wang, Xiaohao and Qian, Xiang}, booktitle={Proceedings of the Genetic and Evolutionary Computation
Conference Companion}, pages={59--60}, year={2021} }

@article{nikitin2020multi, title={The multi-objective optimisation of breakwaters using evolutionary approach},
author={Nikitin, Nikolay O and Polonskaia, Iana S and Kalyuzhnaya, Anna V and Boukhanovsky, Alexander V}, journal={arXiv
preprint arXiv:2004.03010}, year={2020} }






.. |docs| image:: https://readthedocs.org/projects/gefest/badge/?version=latest
   :target: https://gefest.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |license| image:: https://img.shields.io/github/license/nccr-itmo/FEDOT
   :alt: Supported Python Versions
   :target: ./LICENSE.md

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :target: https://t.me/gefest_helpdesk
   :alt: Telegram Chat