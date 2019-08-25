echo 'feature'
python /features_code/base_10feat.py
python /features_code/fuzzy.py
python /features_code/len.py
python /features_code/emb.py
python /features_code/match.py
python /features_code/simi.py

echo 'nn model'
python /nn_model/load_data.py
python /nn_model/run_generator.py

echi 'lgb model'
python /lgb_model/run_lgb.py
