class FCFG_BASE:
    # 训练验证
    IS_TRAIN = True
    IS_VAL = True
    IS_VISUAL = False

    PRINT_FREQ = 1  # 打印频率 与 batch*PRINT_FREQ
    VAL_FREQ = 1  # 验证频率
    NUM_SAVE_INTERVAL = 1
    BATCH_TRAIN = 3
    BATCH_VAL = 5


    # 数据增强
    IS_MULTI_SCALE_V2 = False  # 只要用数据增强必须有
    MULTI_SCALE_VAL_V2 = [200, 300]
