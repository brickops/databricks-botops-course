# databricks-botops-course

Course for doing BotOps (e.g. ML/LLM/AgentOps) on Databricks dataops, based on a data mesh monorepo structure.

## What will you learn in this course?

* How to build LLM Bots on Databricks 
* Move LLM Bots (chat bots, agents etc) from dev to prod 
* Using git branches and commits 
* We will not do Github Actions here, but the processes and code needed are shown
* Evaluation-driven development
* Testing Bots with chat interfaces 
* Structure your environments to allow for dev deployments of Bots 

## What is a Bot?

A Bot is a an app which uses LLM components, e.g. an agent (if it does autonomous routing/decisions), a chatbot,
or an LLM-based endpoint which replies to requests using LLM (Large Language Models=).

## Focus

How can we deploy Databricks bots in a way that is:

- (git-)versioned
- usable
- ordered and sustainable
- enabling each developer / data scientist to do deploy
- way of working for exploration, development, staging and production of pipelines

## Repository structure

We will do our tasks in the context of the folder representing the `tripbot` ml flow:

`orgs/acme/domains/transport/projects/taxinyc/flows/ml/tripbot/`

The structure is a proposal, which might have to be adapted in a real world organization.

The structure is:

- org: `acme`
    - domain: `transport`
        - project: `taxinyc`
            - flowtype: `ml` (meaning Machine Learning / AI, the alternative is `prep`, for ETL/data engineering)
                - flow: `revenue` 

The structure will be applied to:

- Agent / model naming
- Data *code*, i.e. the pyspark code herein git
- The database *tables* produced by that code
- The data pipelines being deployed

The purpose of this structure is to have sufficient granularity to enable each department/org, team/domain, project and pipeline, to be kept apart.

You can explore the structure here in Databricks, or more easily [in the repo with a browser](https://github.com/paalvibe/databricks-botops-course).

## Longer explanation of the repo structure

A longer explanation of the ideas behind the repo structure can be found in the article [Data Platform Urbanism - Sustainable Plans for your Data Work](https://www.linkedin.com/pulse/data-platform-urbanism-sustainable-plans-your-work-p%25C3%25A5l-de-vibe/).

## Dataops libs

For the dataops code, we use the [brickops](https://github.com/brickops/brickops) package from Pypi, to enable a versioned pipeline deployment and way of working. The main logic is under [dataops/deploy](https://github.com/brickops/brickops/blob/main/brickops/dataops/deploy/autojob.py).

## Reusing the structure and libs

The structure and brickops libs can be used in your own projects, by installing the pypi package `brickops`, 
forking the repo and copying the content and adapting it.

## Course

1. Go to course/
   - DO NOT run anything under course/00-Workshop-Admin-Prep
2. Got to course/01-Student-Prep/01-General
   - Go through the instructions under that folder
3. Start with the tasks under 02-DeployTasks
   - Some sections are just for reading or running, others you need to solve.