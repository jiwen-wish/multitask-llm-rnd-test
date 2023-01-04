# multitask-llm-rnd

- [modelling](./modelling/) - RnD modelling for multitask large language model (llm), primarily for query understanding but applies to content understanding tasks in general, migrated from [older modelling link](https://github.com/junwang-wish/query_understanding_model)
- [datasets](./datasets/) - Dataset creation for training and evaluating models, primarily for query understanding but applies to content understanding tasks in general, migrated from [older dataset link](https://github.com/junwang-wish/query_understanding_data)
- [labelling](./labelling/) - Labelling tools for collaborating with Wish offshore team to produce high-quality data in a secure fashion


## Debug
### If there is permission error when adding certain files to Git
1. Check what usergroup you are in
```
groups $USER
```
2. Let's say you are in `developers` group from command above
```
cd .git
sudo chmod -R g+ws *
sudo chgrp -R developers *
git config core.sharedRepository true
```





