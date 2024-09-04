from sacred import Experiment

ex = Experiment("ViLT", save_git_info=False)


def _loss_names(d):  # _ indicates it is not a part of public API.
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,
    }
    ret.update(d)  # input d as a dictionary to update ret.
    return ret


######## basic settings ########
@ex.config
def config():
    exp_name = None
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256

    test_ratio = None
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # missing modality config
    missing_rate = 0.5
    missing_type = {"train": None, "val": None, "test": None}
    missing_ratio = {"train": missing_rate, "val": missing_rate, "test": missing_rate}
    both_ratio = 0.5  # missing both ratio
    missing_table_root = "./datasets/missing_tables/"
    simulate_missing = True
    with_delta_infer = None

    # missing_aware_prompts config
    prompt_type = None
    prompt_length = 16
    learnt_p = None
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = None

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 0.01
    weight_decay = 0.02
    decay_power = 1
    max_epoch = 4
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    max_steps = None

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = None
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 16  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 2
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


######## dataset settings ########
@ex.named_config
def task_finetune_hateful_memes():
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.11
    weight_decay = 2e-2
    max_text_len = 128


@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
    max_text_len = 1024


######## prompt settings ########
@ex.named_config
def kronecker_prompts():
    prompt_type = "kronecker"
    learnt_p = True
    multi_layer_prompt = True


@ex.named_config
def input_prompts():
    prompt_type = "input"
    learnt_p = True
    multi_layer_prompt = True


@ex.named_config
def none_prompts():
    prompt_type = "none"
    learnt_p = False
    multi_layer_prompt = True


######## missing settings ########
@ex.named_config
def trainm_i_testm_t():
    missing_type = {"train": "image", "val": "image", "test": "text"}


@ex.named_config
def trainm_t_testm_t():
    missing_type = {"train": "text", "val": "text", "test": "text"}


@ex.named_config
def trainm_b_testm_t():
    missing_type = {"train": "both", "val": "both", "test": "text"}


@ex.named_config
def trainm_i_testm_i():
    missing_type = {"train": "image", "val": "image", "test": "image"}


@ex.named_config
def trainm_t_testm_i():
    missing_type = {"train": "text", "val": "text", "test": "image"}


@ex.named_config
def trainm_b_testm_i():
    missing_type = {"train": "both", "val": "both", "test": "image"}


@ex.named_config
def trainm_i_testm_b():
    missing_type = {"train": "image", "val": "image", "test": "both"}


@ex.named_config
def trainm_t_testm_b():
    missing_type = {"train": "text", "val": "text", "test": "both"}


@ex.named_config
def trainm_b_testm_b():
    missing_type = {"train": "both", "val": "both", "test": "both"}
