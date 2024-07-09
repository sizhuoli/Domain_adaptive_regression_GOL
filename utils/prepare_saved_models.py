# dict of model name and corresponding model full path
import ipdb

model_dict = {
    'DK-height-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-chm-fixed/GOL_addlinear_layer_more_aug2024-02-07_20:52:19/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-height-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-chm-fixed/GOL_vitB16_addlinear_layer_more_aug_2024-02-11_16:19:12/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-height-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-chm-fixed/GOLvanilla_addlinear_layer_more_aug2024-02-08_01:08:30/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-height-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/square_nonzero_mean_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_15:22:20/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-height-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/square_nonzero_mean_regressor_vit_b16_reduce_more_aug_2024-02-14_19:16:16/regressor_vit_b16_reduce/val_best_r2.pth',
    'DK-height-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/square_nonzero_mean_regressor_vgg16v2norm_reduce_more_aug_2024-02-15_21:55:15/regressor_vgg16v2norm_reduce/val_best_r2.pth',
    'DK-count-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-count-fixed/GOL_addlinear_layer_more_aug_2024-02-07_19:16:13/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-count-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-count-fixed/GOL_vitB16_addlinear_layer_more_aug_2024-02-12_00:41:23/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-count-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-count-fixed/GOLvanilla_addlinear_layer_more_aug_2024-02-08_01:10:59/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-count-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/count_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_23:41:59/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-count-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/count_regressor_vit_b16_reduce_more_aug_2024-02-14_23:33:37/regressor_vit_b16_reduce/val_best_r2.pth',
    'DK-count-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/count_regressor_vgg16v2norm_reduce_more_aug_2024-02-15_23:11:19/regressor_vgg16v2norm_reduce/val_best_r2.pth',
    'DK-treecover5m-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-treecover-fixed/GOL_addlinear_layer_more_aug_2024-02-06_23:30:16/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-treecover5m-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-treecover-fixed/tree_cover_thre5m_GOL_vitB16_addlinear_layer_more_aug2024-02-11_19:02:51/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-treecover5m-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK-treecover-fixed/GOLvanilla_addlinear_layer_more_aug2024-02-08_14:01:09/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'DK-treecover5m-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/tree_cover_thre5m_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-14_07:46:19/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-treecover5m-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/tree_cover_thre5m_regressor_vit_b16_reduce_more_aug_2024-02-15_11:26:41/regressor_vit_b16_reduce/val_best_r2.pth',
    'DK-treecover5m-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/tree_cover_thre5m_regressor_vgg16v2norm_reduce_more_aug_2024-02-16_00:27:21/regressor_vgg16v2norm_reduce/val_best_r2.pth',
     # dim reduced to 3
    'DK-chm-vit-GOL-reduced': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/3_layer_feature_square_nonzero_mean_GOL_vit_b16_reduce_more_aug_2024-02-23_11:51:58/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-count-vit-GOL-reduced': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/DK/3_layer_feature_count_GOL_vit_b16_reduce_more_aug_2024-02-23_13:13:43/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'DK-treecover5m-vit-GOL-reduced': None,
    #
    'france-height-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_GOL_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-12_19:53:00/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-height-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_GOL_vitB16_addlinear_layer_more_aug_2024-02-12_19:10:55/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-height-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_GOLvanilla_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-14_12:45:52/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-height-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_13:12:28/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-height-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_regressor_vit_b16_reduce_more_aug_2024-02-15_00:08:52/regressor_vit_b16_reduce/val_best_r2.pth',
    'france-height-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/square_nonzero_mean_regressor_vgg16v2norm_reduce_more_aug_2024-02-15_13:47:49/regressor_vgg16v2norm_reduce/val_best_r2.pth',
    'france-count-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_GOL_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-13_00:44:02/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-count-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_GOL_vit_b16_reduce_addlinear_layer_more_aug_2024-02-12_22:32:12/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-count-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_GOLvanilla_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-14_16:02:23/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-count-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_16:34:11/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-count-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_regressor_vit_b16_reduce_more_aug_2024-02-15_01:49:16/regressor_vit_b16_reduce/val_best_r2.pth',
    'france-count-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/count_regressor_vgg16v2norm_reduce_more_aug_2024-02-15_15:25:39/regressor_vgg16v2norm_reduce/val_best_r2.pth',
    'france-treecover5m-vgg-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_GOL_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-13_10:20:24/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-treecover5m-vit-GOL': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_GOL_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_05:17:29/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-treecover5m-vgg-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_GOLvanilla_vgg16v2norm_reduce_addlinear_layer_more_aug_2024-02-14_19:28:54/tau1_metricL2_GOL_vgg16v2norm_reduce/val_best_rank_f1.pth',
    'france-treecover5m-vit-GOLvanilla': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_GOLvanilla_vit_b16_reduce_addlinear_layer_more_aug_2024-02-13_23:18:41/tau1_metricL2_GOL_vit_b16_reduce/val_best_rank_f1.pth',
    'france-treecover5m-vit-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_regressor_vit_b16_reduce_more_aug_2024-02-15_03:31:44/regressor_vit_b16_reduce/val_best_r2.pth',
    'france-treecover5m-vgg-regressor': '/home/sizhuo/Desktop/code_repository/GOL/saved_models/france/tree_cover_thre5m_regressor_vgg16v2norm_reduce_more_aug_2024-02-15_17:03:12/regressor_vgg16v2norm_reduce/val_best_r2.pth',
    }



def prepare_saved_model(model_name, models_dict = model_dict):
    model = models_dict[model_name]
    return model






if __name__ == '__main__':
    model = prepare_saved_model(model_dict, 'DK-height-vgg-GOL')
    print(model)
    # pass


