python run_text_classification.py \
--model_name_or_path distilbert-base-cased \
--train_file training_data.json \
--validation_file validation_data.json \
--output_dir output/ \
--test_file data_to_predict.json \
--max_seq_length 512 \
--overwrite_output_dir
