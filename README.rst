.. image:: /docs/img/gefest_logo.png
   :alt: Logo of GEFEST framework

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |python|
   * - tests
     - | |build|
   * - docs
     - |docs|
   * - license
     - | |license|
   * - support
     - | |tg|
   * - gitlab
     - | |gitlab|
   * - funding
     - | |ITMO| |NCCR|

.. end-badges

**GEFEST** (**G**\enerative **E**\volution **F**\or **E**\ncoded **ST**\ructures) is a toolbox for the generative design of
physical objects.

In core it uses:
1. Numerical modelling to simulate the interaction between object and environment
2. Evolutionary optimization to produce new variants of geometrically-encoded structures

The basic abstractions in GEFEST are Point, Polygon, Structure and Domain. Architecture of the GEFEST can be described as:

.. figure:: /docs/img/workflow.png
   :figwidth: image
   :align: center


The evolutionary workflow of the generative design is the following:

.. figure:: /docs/img/evo.png
   :figwidth: image
   :align: center

The dynamics of the optimisation can be visualized as (breakwaters optimisation case):

.. image:: /docs/img/breakwaters.gif

How to use
==========

All details about first steps with GEFEST might be found in the `quick start guide <https://gefest.readthedocs.io/en/latest/gefest/quickstart.html>`__.

Tutorals for more spicific use cases can be found `tutorial section of docs <https://gefest.readthedocs.io/en/latest/tutorials/index.html>`__.

Project Structure
=================

The latest stable release of GEFEST is on the `main branch <https://github.com/ITMO-NSS-team/GEFEST/tree/main>`__.

The repository includes the following directories:

* Package `core <https://github.com/ITMO-NSS-team/GEFEST/tree/main/gefest/core>`__  contains the main classes and scripts. It is the *core* of GEFEST framework;
* Package `cases <https://github.com/ITMO-NSS-team/GEFEST/tree/main/cases>`__ includes several *how-to-use-cases* where you can start to discover how GEFEST works;
* All *unit and integration tests* can be observed in the `test <https://github.com/ITMO-NSS-team/GEFEST/tree/main/test>`__ directory;
* The sources of the documentation are in the `docs <https://github.com/ITMO-NSS-team/GEFEST/tree/main/docs>`__.
* Weights of pretrained DL models can be downloaded from `this repository <https://gitlab.actcognitive.org/itmo-nss-team/gefest-models>`__.

Cases and examples
==================
**Note**: To run the examples below, the old kernel gefest version, which can be installed on python 3.7 with:

.. code-block:: bash

   pip install git+https://github.com/aimclub/GEFEST.git@4f9c34c449c0eb65d264476e5145f09b4839cd70

- `Experiments <https://github.com/ITMO-NSS-team/GEFEST-paper-experiments>`__ with various real and synthetic cases
- `Case <https://github.com/ITMO-NSS-team/rbc-traps-generative-design>`__ devoted to the red blood cell traps design.

Migrated examples can be found in cases folder of the main branch.

Current R&D and future plans
============================

Currently, we are working on integration of new types of physical objects with consideration of their internal structure.

The major ongoing tasks:

* to integrate three dimensional physical objects
* to implement gradient based approaches for optimization of physical objects

Documentation
=============

Detailed information and description of GEFEST framework is available in the `Read the Docs <https://gefest.readthedocs.io/en/latest/>`__

Contribution guide
==================

The contribution guide is available in the `page <https://gefest.readthedocs.io/en/latest/contribution.html>`__

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences
and workshops for their valuable advice and suggestions.

Contacts
========

* `Telegram channel for solving problems and answering questions on GEFEST <https://t.me/gefest_helpdesk>`_
* `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
* `Newsfeed <https://t.me/NSS_group>`_
* `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Funding
=======

This research is financially supported by the Foundation for
National Technology Initiative's Projects Support as a part of the roadmap
implementation for the development of the high-tech field of
Artificial Intelligence for the period up to 2030 (agreement 70-2021-00187).

Citation
========

@article{starodubcev2023generative,
  title={Generative design of physical objects using modular framework},
  author={Starodubcev, Nikita O and Nikitin, Nikolay O and Andronova, Elizaveta A and Gavaza, Konstantin G and Sidorenko, Denis O and Kalyuzhnaya, Anna V},
  journal={Engineering Applications of Artificial Intelligence},
  volume={119},
  pages={105715},
  year={2023},
  publisher={Elsevier}}

@inproceedings{solovev2023ai,
  title={AI Framework for Generative Design of Computational Experiments with Structures in Physical Environment},
  author={Solovev, Gleb Vitalevich and Kalyuzhnaya, Anna and Hvatov, Alexander and Starodubcev, Nikita and Petrov, Oleg and Nikitin, Nikolay},
  booktitle={NeurIPS 2023 AI for Science Workshop},
  year={2023}}

.. |build| image:: https://github.com/aimclub/GEFEST/workflows/unit%20tests/badge.svg?branch=main
   :alt: Build Status
   :target: https://github.com/aimclub/GEFEST/actions

.. |docs| image:: https://readthedocs.org/projects/gefest/badge/?version=latest
   :target: https://gefest.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |license| image:: https://img.shields.io/github/license/ITMO-NSS-team/GEFEST
   :alt: Supported Python Versions
   :target: ./LICENSE.md

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :target: https://t.me/gefest_helpdesk
   :alt: Telegram Chat

.. |ITMO| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/add_badge/badges/ITMO_badge_rus.svg
   :alt: Acknowledgement to ITMO
   :target: https://itmo.ru

.. |NCCR| image:: https://github.com/ITMO-NSS-team/open-source-ops/blob/add_badge/badges/NCCR_badge.svg
   :alt: Acknowledgement to NCCR
   :target: https://actcognitive.org/

.. |gitlab| image:: https://img.shields.io/badge/mirror-GitLab-orange
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/GEFEST

.. |pypi| image:: https://img.shields.io/pypi/v/gefest.svg
   :alt: PyPI Package Version
   :target: https://img.shields.io/pypi/v/gefest

.. |python| image:: https://img.shields.io/pypi/pyversions/gefest.svg
   :alt: Supported Python Versions
   :target: https://img.shields.io/pypi/pyversions/gefest
