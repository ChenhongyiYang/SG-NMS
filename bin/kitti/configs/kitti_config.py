from nets import net_class


k_width, k_height = 1242, 375

net_height, net_width = int(k_height * 1.2), int(k_width * 1.2)


##########################################################################################
# Training config
##########################################################################################
train_net_param = net_class.nets_params(
        net_shape= (net_height,net_width),
        img_shape= (k_height, k_width),
        crop_shape=(k_height, k_width),
        num_classes=2,
        layer_num=3,
        feat_shape=((1+(net_height-1)//32, 1+(net_width-1)//32),
                    (1+(net_height-1)//16, 1+(net_width-1)//16),
                    (1+(net_height-1)//8, 1+(net_width-1)//8)),
        anchor_sizes=[(300, 250, 205), (170, 135, 100), (75, 50, 25)],
        anchor_ratios=[(1/3., 1/2., 1., 1.5), (1/3., 1/2., 1., 1.5), (1/3., 1/2., 1., 1.5)],
        anchor_offset=0.5,
        proposal_nums=(750, 1250, 1500),
        Ks=[(5, 5), (5, 5), (3, 3)],
        max_bbox_num=22,
        anchor_nums=(12, 12, 12),
        rpn_pos_threshold=0.5,
        rpn_neg_threshold=0.4,
        rpn_neg_threshold_low=0.0,
        net_loc_threshold=0.5,
        net_pos_threshold=0.70,
        net_neg_threshold=0.60,
        net_neg_threshold_low=0.0,
        ae_up_threshold=0.7,
        ae_low_threshold=0.5,
        noise_xy = 0.06,
        noise_wh = 0.2,
        select_nms_th = 0.7,
        rpn_mini_batch=256,
        rfcn_mini_batch=128,
        back_OHEM=True,
        output_box_ratio_high=2., # does not matter when train
        output_box_ratio_low=1/3.6, # does not matter when train
        out_box_num=300,
        net='resnet_101_v1'
)




##########################################################################################
# Inference config
##########################################################################################
inference_net_param = net_class.nets_params(
        net_shape= (net_height,net_width),
        img_shape= (k_height, k_width),
        crop_shape=(k_height, k_width),
        num_classes=2,
        layer_num=3,
        feat_shape=((1+(net_height-1)//32, 1+(net_width-1)//32),
                    (1+(net_height-1)//16, 1+(net_width-1)//16),
                    (1+(net_height-1)//8, 1+(net_width-1)//8)),
        anchor_sizes=[(300, 250, 205), (170, 135, 100), (75, 50, 25)],
        anchor_ratios=[(1/3., 1/2., 1., 1.5), (1/3., 1/2., 1., 1.5), (1/3., 1/2., 1., 1.5)],
        anchor_offset=0.5,
        proposal_nums=(2000, 2000, 2000),
        Ks=[(5, 5), (5, 5), (3, 3)],
        max_bbox_num=22,
        anchor_nums=(12, 12, 12),
        rpn_pos_threshold=0.5,
        rpn_neg_threshold=0.4,
        rpn_neg_threshold_low=0.0,
        net_loc_threshold=0.5,
        net_pos_threshold=0.75,
        net_neg_threshold=0.7,
        net_neg_threshold_low=0.0,
        ae_up_threshold=0.7,
        ae_low_threshold=0.3,
        noise_xy = 0.08,
        noise_wh = 0.25,
        select_nms_th = 0.7,
        rpn_mini_batch=256,
        rfcn_mini_batch=256,
        back_OHEM=True,
        output_box_ratio_high=2., # does not matter when train
        output_box_ratio_low=1/3.6, # does not matter when train
        out_box_num=500,
        net='resnet_101_v1'
)

























