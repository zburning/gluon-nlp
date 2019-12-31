from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
dev_data_path = '/home/ubuntu/.mxnet/datasets/squad/dev-v2.0.json '
output_prediction_file = './output_dir/predictions_p.json'
output_null_log_odds_file = './output_dir/null_odds_p.json'

evaluate_options = EVAL_OPTS(data_file=dev_data_path, pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)

results = evaluate_on_squad(evaluate_options)