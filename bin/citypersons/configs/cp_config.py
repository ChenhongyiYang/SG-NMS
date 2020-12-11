from nets import net_class



cp_height, cp_width = 1024, 2048

crop_height, crop_width = cp_height // 2, cp_width // 2


net_height_train, net_width_train = 672 + 1, 1332 + 1
#net_height_test, net_width_test = 1344 + 1, 2688 + 1

r = 1.3
net_height_test = int(cp_height * r)
net_width_test = int(cp_width * r)


max_bbox_num = 100


cp_train_param = net_class.nets_params(
        net_shape = (net_height_train, net_width_train),
        img_shape = (cp_height, cp_width),
        crop_shape = (crop_height, crop_width),
        num_classes =2,
        layer_num = 3,
        feat_shape = ((1+(net_height_train-1)//32, 1+(net_width_train-1)//32),
                    (1+(net_height_train-1)//16, 1+(net_width_train-1)//16),
                    (1+(net_height_train-1)//8, 1+(net_width_train-1)//8)),
        anchor_sizes=[[25 * 1.14**i for i in range(24,16,-1)], [25 * 1.14**i for i in range(16,8,-1)],[25 * 1.14**i for i in range(8,0,-1)]],
        anchor_ratios=[[2.45], [2.45], [2.45]],
        anchor_offset=0.5,
        proposal_nums=(750, 1250, 1500),
        Ks=[(6, 3), (6, 3), (4, 2)],
        max_bbox_num=max_bbox_num,
        anchor_nums=(8, 8, 8),
        rpn_pos_threshold=0.6,  # does not matter when inference
        rpn_neg_threshold=0.3,  # does not matter when inference
        rpn_neg_threshold_low=0.0,  # does not matter when inference
        net_loc_threshold=0.5,  # does not matter when inference
        net_pos_threshold=0.6,  # does not matter when inference
        net_neg_threshold=0.5,  # does not matter when inference
        net_neg_threshold_low=0.0,  # does not matter when inference
        ae_up_threshold=0.6,
        ae_low_threshold=0.3,
        noise_xy = 0.,
        noise_wh = 0.,
        select_nms_th=0.7,
        rpn_mini_batch=256,  # does not matter when inference
        rfcn_mini_batch=256,  # does not matter when inference
        back_OHEM=True,
        output_box_ratio_high=3.0, # does not matter when train
        output_box_ratio_low=2.0, # does not matter when train
        out_box_num=300,
        net='resnet_50_v1'
)


cp_test_param = net_class.nets_params(
        net_shape=(net_height_test, net_width_test),
        img_shape=(cp_height, cp_width),
        crop_shape=(crop_height, crop_width),
        num_classes=2,
        layer_num=3,
        feat_shape=((1+(net_height_test-1)//32, 1+(net_width_test-1)//32),
                    (1+(net_height_test-1)//16, 1+(net_width_test-1)//16),
                    (1+(net_height_test-1)//8, 1+(net_width_test-1)//8)),
        anchor_sizes=[[25 * 1.14**i for i in range(24,16,-1)], [25 * 1.14**i for i in range(16,8,-1)],[25 * 1.14**i for i in range(8,0,-1)]],
        anchor_ratios=[[2.45], [2.45], [2.45]],
        anchor_offset=0.5,
        proposal_nums=(800, 800,800),
        Ks=[(6, 3), (6, 3), (4, 2)],
        max_bbox_num=max_bbox_num,
        anchor_nums=(8, 8, 8),
        rpn_pos_threshold=0.6,  # does not matter when inference
        rpn_neg_threshold=0.3,  # does not matter when inference
        rpn_neg_threshold_low=0.0,  # does not matter when inference
        net_loc_threshold=0.5,  # does not matter when inference
        net_pos_threshold=0.6,  # does not matter when inference
        net_neg_threshold=0.4,  # does not matter when inference
        net_neg_threshold_low=0.0,  # does not matter when inference
        ae_up_threshold=0.6, # does not matter when inference
        ae_low_threshold=0.3, # does not matter when inference
        noise_xy = 0.,
        noise_wh = 0.,
        select_nms_th=0.7,
        rpn_mini_batch=256,  # does not matter when inference
        rfcn_mini_batch=256,  # does not matter when inference
        back_OHEM=True,
        output_box_ratio_high=999., # does not matter when train
        output_box_ratio_low=0., # does not matter when train
        out_box_num=500,
        net='resnet_50_v1'
)



































