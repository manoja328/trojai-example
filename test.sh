#python entrypoint.py configure \
#--automatic_configuration \
#--scratch_dirpath=./scratch/ \
#--metaparameters_filepath=./metaparameters.json \
#--schema_filepath=./metaparameters_schema.json \
#--learned_parameters_dirpath=./learned_parameters/ \
#--configure_models_dirpath=/path/to/new-train-dataset \
#--tokenizers_dirpath=./tokenizers


python entrypoint.py infer \
--model_filepath=./model/id-00000003/model.pt \
--tokenizer_filepath=./tokenizers/csarron-mobilebert-uncased-squad-v2.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch \
--examples_dirpath=./model/id-00000003/example_data \
--round_training_dataset_dirpath=/workspace/manoj/trojai-datasets/nlp-question-answering-aug2023 \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/