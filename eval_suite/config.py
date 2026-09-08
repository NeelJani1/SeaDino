"""Global constants, paths, and normalization parameters."""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENTHIC_MEAN = [0.359, 0.413, 0.386]
BENTHIC_STD = [0.219, 0.215, 0.209]

DEFAULT_PATHS = {
    "benthic_img_root": "/home/njan320/Neel/BenthicNet/01_BenthicNet/images/labelled/full_labelled_512px/compiled_labelled_512pix",
    "substrate_csv": "/home/njan320/SeaDino/substrate_depth_2_spatial_split.csv",
    "german_bank_csv": "/home/njan320/SeaDino/german_bank_2010_spatial_split.csv",
    "biota_csv": "/home/njan320/Neel/BenthicNet/csvs/finalized_csvs/trainable/benthicnet_nn.csv",
    "coralmask_dir": "/home/njan320/Neel/CoralMaskv1/CoralMask",
    "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
}

# Official class labels matching Fig 7 & Fig 8 in the BenthicNet paper
SUBSTRATE_CLASS_NAMES = ["Boulders", "Cobbles", "Rock", "Pebble/Gravel", "Sand/Mud (<2mm)"]
GERMAN_BANK_CLASS_NAMES = ["silt/mud", "silt with bedforms", "reef", "glacial till", "sand with bedforms"]
